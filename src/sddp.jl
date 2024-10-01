module Jl_SDDP

using SDDP
using HiGHS
using Clp
using JuMP
using Base
using Xpress
using PythonCall

# Define types for structured data
struct Reservoir
    capacity::Float64
    efficiency::Float64
    max_pumping::Vector{Float64}
    max_generating::Vector{Float64}
    upper_level::Vector{Float64}
    lower_level::Vector{Float64}
    upper_curve_penalty::Float64
    lower_curve_penalty::Float64
    spillage_penalty::Float64
    level_init::Float64
    inflows::Matrix{Float64}
end

struct LinInterp
    inputs::Matrix{Float64}
    base_costs::Vector{Float64}
    slopes::Matrix{Float64}
end

struct Normalizer
    euro::Float64
    energy::Float64
    price::Float64
end

# Function to convert Python data to Julia structures
function formater(n_weeks, n_scenarios, reservoirs_data, costs_approx_data, norm_euros::Float64=1e7, norm_enrgy::Float64=1e5)
    round_energy = 4#4#3
    round_euro = 8#2
    round_price = 5#4
    norm_price = norm_euros / norm_enrgy # € / MWh
    norms = Normalizer(norm_euros, norm_enrgy, norm_price)

    # divisor = 1e6
    reservoirs = [Reservoir(
                    round(pyconvert(Float64, r["capacity"])             /norm_enrgy, digits=round_energy), #MWh
                    round(pyconvert(Float64, r["efficiency"])                      , digits=round_energy), #MWh / MWh
                    round.(pyconvert(Vector, r["max_pumping"])          /norm_enrgy, digits=round_energy), #MWh
                    round.(pyconvert(Vector, r["max_generating"])       /norm_enrgy, digits=round_energy), #MWh
                    round.(pyconvert(Vector, r["upper_level"])          /norm_enrgy, digits=round_energy), #MWh
                    round.(pyconvert(Vector, r["lower_level"])          /norm_enrgy, digits=round_energy), #Mwh 
                    round(pyconvert(Float64, r["upper_curve_penalty"])  /norm_price, digits=round_price), # €/MWh
                    round(pyconvert(Float64, r["lower_curve_penalty"])  /norm_price, digits=round_price), # €/MWh
                    round(pyconvert(Float64, r["spillage_penalty"])     /norm_price, digits=round_price), # €/MWh
                    round(pyconvert(Float64, r["level_init"])           /norm_enrgy, digits=round_energy), # MWh
                    round.(pyconvert(Matrix, r["inflows"])              /norm_enrgy, digits=round_energy), # MWh
                ) for r in reservoirs_data]
                
    size_ca_data = size(costs_approx_data)
    costs_approx = [[
        LinInterp(
            round.(pyconvert(Matrix, costs_approx_data[i,j]["inputs"])  /norm_enrgy, digits=round_energy), # MWh
            round.(pyconvert(Vector, costs_approx_data[i,j]["costs"])   /norm_euros, digits=round_euro), # €
            round.(pyconvert(Matrix, costs_approx_data[i,j]["duals"])   /norm_price, digits=round_price), # € / MWh
            ) for j in 1:size_ca_data[2]] for i in 1:size_ca_data[1]]
            
    return (n_weeks, n_scenarios, reservoirs, costs_approx, norms)
end


function generate_model(n_weeks::Int, n_scenarios::Int, reservoirs::Vector{Main.Jl_SDDP.Reservoir}, costs_approx::Vector{Vector{Main.Jl_SDDP.LinInterp}}, norms::Normalizer)
    n_reservoirs = size(reservoirs)[1]
    model = SDDP.LinearPolicyGraph(
        stages = 3*n_weeks,
        sense = :Min,
        lower_bound = 0.0,
        optimizer = Xpress.Optimizer,
        # optimizer = HiGHS.Optimizer,
        # cut_oracle = SDDP.LevelOneCutOracle()
    ) do subproblem, stage

        # Declaring the state variable 
        @variable(
            subproblem,
            0<=level[r=1:n_reservoirs]<=maximum([reservoirs[r].capacity for r in 1:n_reservoirs]),
            SDDP.State,
            initial_value = 0
        )
        
        @variables(subproblem, begin
            control[r=1:n_reservoirs]
            ξ[r=1:n_reservoirs]  # Inflows as noise.
            Ξ[s=1:n_scenarios]  # Current scenario as noise
            over_upper[r=1:n_reservoirs] >= 0
            below_lower[r=1:n_reservoirs] >= 0
            spillage[r=1:n_reservoirs] >= 0
            cost >= 0
        end)
        
        # Add constraints for control bounds
        modulo_stage = (stage-1)%n_weeks + 1
        modulo_prev_stage = (stage-2+n_weeks)%n_weeks + 1
        for r in 1:n_reservoirs
            @constraint(subproblem, control[r] >= -reservoirs[r].max_pumping[modulo_stage] * reservoirs[r].efficiency)
            @constraint(subproblem, control[r] <= reservoirs[r].max_generating[modulo_stage])
        end
        
        @constraints(subproblem, begin
            demand_constraint[r=1:n_reservoirs],  level[r].out == (level[r].in + (stage==1)*reservoirs[r].level_init) - control[r] + ξ[r] - spillage[r] 
            [r=1:n_reservoirs], level[r].in <= reservoirs[r].upper_level[modulo_prev_stage] + over_upper[r]
            [r=1:n_reservoirs], level[r].in >= reservoirs[r].lower_level[modulo_prev_stage] - below_lower[r]
            [r=1:n_reservoirs], level[r].out <= reservoirs[r].capacity
            [r=1:n_reservoirs], level[r].in >= 0
            [r=1:n_reservoirs], level[r].in <= reservoirs[r].capacity
        end)
        
        # Define scenarios for inflows
        # inflows = [
        #     [reservoirs[r]["inflows"][stage][scenario] for r in 1:n_reservoirs]
        #     for scenario in 1:n_scenarios
        # ]
        Ω = [
            (
                inflows = [reservoirs[r].inflows[modulo_stage, scenario] for r in 1:n_reservoirs],
                next_scenario_state = scenario,
                hyperps_selec = [s != scenario + 0 for s in 1:n_scenarios],
            )
            for scenario in 1:n_scenarios
        ]

        P = ones(n_scenarios)/n_scenarios
                
        SDDP.parameterize(subproblem, Ω, P) do ω
            for r in 1:n_reservoirs
                JuMP.fix(ξ[r], ω.inflows[r])
            end
            for s in 1:n_scenarios
                JuMP.fix(Ξ[s], s != ω.hyperps_selec[s])
            end
        end
        
        @stageobjective(subproblem, cost + sum((over_upper[r]) * reservoirs[r].upper_curve_penalty + (below_lower[r]) * reservoirs[r].lower_curve_penalty + spillage[r] * reservoirs[r].spillage_penalty 
        for r in 1:n_reservoirs))
            
        COST_UB = 1e10/norms.euro #Beware as too high a value WILL create numerical stability and generate problems labelled as INFEASIBLE
        for scenario in 1:n_scenarios #Fixer pour faire correspondre avec les inflows #DONE
            hyperps = costs_approx[modulo_stage][scenario]
            for (i, base_cost) in enumerate(hyperps.base_costs)
                @constraint(subproblem, cost >= base_cost - sum(hyperps.slopes[i,:].*(hyperps.inputs[i,:] - control)) - COST_UB*Ξ[scenario])
            end
        end
    end
    return model
end

function stability_report(model)
    return SDDP.numerical_stability_report(model)
end

function reinit_cuts(n_weeks::Int, n_scenarios::Int, reservoirs::Vector{Main.Jl_SDDP.Reservoir}, costs_approx::Vector{Vector{Main.Jl_SDDP.LinInterp}}, norms::Normalizer)
    model = generate_model(n_weeks, n_scenarios, reservoirs, costs_approx, norms)
    SDDP.write_cuts_to_file(model, "C:/Users/trenquierelo/Desktop/antares-water-values/notebooks/sddp_current_cuts")
end

function manage_reservoirs(n_weeks::Int, n_scenarios::Int, reservoirs::Vector{Main.Jl_SDDP.Reservoir}, costs_approx::Vector{Vector{Main.Jl_SDDP.LinInterp}}, norms::Normalizer)
    model = generate_model(n_weeks, n_scenarios, reservoirs, costs_approx, norms)
    SDDP.read_cuts_from_file(model, "C:/Users/trenquierelo/Desktop/antares-water-values/notebooks/sddp_current_cuts")
    # Training the model
    SDDP.train(model, stopping_rules = [SDDP.BoundStalling(40, 1e1)], iteration_limit = 700, cut_type = SDDP.MULTI_CUT)
    SDDP.write_cuts_to_file(model, "C:/Users/trenquierelo/Desktop/antares-water-values/notebooks/sddp_current_cuts")
    
    #Simulating
    simulation_results = get_trajectory(n_weeks, n_scenarios, reservoirs, model, norms)
    return simulation_results, model
    # #Getting the usage values
    # n_disc = 101
    # VU, costs = get_usage_values(n_weeks, n_scenarios, reservoirs, model, norms, n_disc)
    # return VU, costs, simulation_results
end

function get_trajectory(n_weeks::Int, n_scenarios::Int, reservoirs::Vector{Main.Jl_SDDP.Reservoir}, model, norms::Normalizer)
    n_reservoirs=size(reservoirs)[1]
    simulations = []
    for scenario in 1:n_scenarios
        sampling_scheme = SDDP.OutOfSampleMonteCarlo(
            model;
            use_insample_transition = true
            ) do node
                stage = node
                modulo_stage = (stage-1)%n_weeks + 1
                return [SDDP.Noise((
                    inflows=[reservoirs[r].inflows[modulo_stage, scenario] for r in 1:n_reservoirs],
                    hyperps_selec = [s != scenario + 0 for s in 1:n_scenarios]
                    ), 1.0)]
            end
        simulation_result = SDDP.simulate(model, 1, sampling_scheme=sampling_scheme, custom_recorders= Dict{Symbol, Function}(
            :control => (sp::JuMP.Model) -> [JuMP.value(sp[:control][r]) * norms.energy for r in 1:n_reservoirs],
            :level_in => (sp::JuMP.Model) -> [JuMP.value(sp[:level][r].in) * norms.energy for r in 1:n_reservoirs],
            :level_out => (sp::JuMP.Model) -> [JuMP.value(sp[:level][r].out) * norms.energy for r in 1:n_reservoirs],
            :cost => (sp::JuMP.Model) -> JuMP.value(sp[:cost]) * norms.euro,
            :spillage => (sp::JuMP.Model) -> [JuMP.value(sp[:spillage][r]) * norms.energy for r in 1:n_reservoirs]
        ))
        append!(simulations, simulation_result)
    end
    return simulations
end

function get_usage_values(n_weeks::Int, n_scenarios::Int, reservoirs::Vector{Main.Jl_SDDP.Reservoir}, model, norms::Normalizer, discretization::Int=101)
    n_disc=discretization
    n_reservoirs = size(reservoirs)[1]
    VU = zeros(n_weeks, n_disc, n_reservoirs)
    costs = zeros(n_weeks, n_disc)
    for scen = 1:n_scenarios
        traj = get_trajectory(n_weeks, n_scenarios, reservoirs, model, norms)[scen]
        for week = 1:n_weeks
            V = SDDP.ValueFunction(model; node=week+n_weeks) #Added n_weeks, because of effect of fixing initial_level
            level_outs = traj[week][:level_out]
            for d = 1:n_disc
                tot_cost=0
                for res in 1:n_reservoirs
                    cost, water_vals = SDDP.evaluate(V, Dict("level[$r]" => ((r==res) * reservoirs[r].capacity*d/n_disc + (r!=res)*level_outs[r]/norms.energy) for r in 1:n_reservoirs))
                    tot_cost += cost
                    # Store the results
                    VU[week, d, res] += -water_vals[Symbol("level[$res]")]/n_scenarios * norms.price  # Store the dual values (water values)
                end
                costs[week, d] += tot_cost / (n_reservoirs*n_scenarios) * norms.euro  # Store the cost
            end
        end
    end
    return VU, costs
end

export manage_reservoirs, get_usage_values, stability_report, reinit_cuts
end
# Call the function
# results = manage_reservoirs(n_weeks, n_scenarios, reservoirs, costs_approx)

# Print the results
# println(results)
