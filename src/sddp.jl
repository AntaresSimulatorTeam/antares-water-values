using SDDP
using HiGHS
using JuMP
  
function manage_reservoirs(n_weeks, n_scenarios, reservoirs, costs_approx)
    n_reservoirs = length(reservoirs)
    model = SDDP.LinearPolicyGraph(
        stages = n_weeks,
        sense = :Min,
        optimizer = HiGHS.Optimizer
    ) do subproblem, stage
        @variable(
            subproblem,
            0 <= level[r=1:n_reservoirs] <= [reservoirs[r]["capacity"] for r in 1:n_reservoirs],
            SDDP.State,
            initial_value = [reservoirs[r]["level_init"] for _ in 1:n_reservoirs]
        )
        @variables(subproblem, begin
            -[reservoirs[r]["max_pumping"][stage] * reservoirs[r]["efficiency"] for r in 1:n_reservoirs] <= control[r=1:n_reservoirs] <= [reservoirs[r]["max_generating"][stage] for r in 1:n_reservoirs]
            ξ[r=1:n_reservoirs]  # Inflows as noise.
            over_upper[r=1:n_reservoirs] >= 0
            below_lower[r=1:n_reservoirs] >= 0
            spillage[r=1:n_reservoirs] >= 0
            cost >= 0
        end)
        
        @constraints(subproblem, begin
            [r=1:n_reservoirs], level[r].out == level[r].in - control[r] + ξ[r] - spillage[r]
            [r=1:n_reservoirs], level[r].out <= reservoirs[r]["upper_level"] + over_upper[r]
            [r=1:n_reservoirs], level[r].out >= reservoirs[r]["lower_level"] - below_lower[r]
        end)
        
        # Define scenarios for inflows
        inflow_scenarios = [
            [reservoirs[r]["inflows"][stage][scenario] for r in 1:n_reservoirs]
            for scenario in 1:n_scenarios
        ]

        SDDP.parameterize(subproblem, inflow_scenarios) do ω
            for r in 1:n_reservoirs
                JuMP.fix(ξ[r], ω[r])
            end
        end
        
        @stageobjective(subproblem, cost + sum(over_upper[r] * reservoirs[r]["upper_curve_penalty"] + below_lower[r] * reservoirs[r]["lower_curve_penalty"] + spillage[r] * reservoirs[r]["spillage_penalty"] 
            for r in 1:n_reservoirs))
        
        for scenario in 1:n_scenarios
            for r in 1:n_reservoirs
                piecewise_approx = costs_approx[stage][scenario][r]
                for (input, base_cost, slopes) in piecewise_approx
                    @constraint(subproblem, cost >= base_cost + dot(slopes, input - control[r]))
                end
            end
        end
    end

    SDDP.train(model, iteration_limit = 1000, log_frequency = 100)

    simulation_results = SDDP.simulate(model, 5000)
    results = []
    for simulation in simulation_results
        stage_results = []
        for (t, data) in enumerate(simulation)
            stage_result = Dict()
            stage_result[:control] = data[:control]
            stage_result[:level_in] = [data[:level][r].in for r in 1:n_reservoirs]
            stage_result[:level_out] = [data[:level][r].out for r in 1:n_reservoirs]
            stage_result[:over_upper] = data[:over_upper]
            stage_result[:below_lower] = data[:below_lower]
            stage_result[:spillage] = data[:spillage]
            push!(stage_results, stage_result)
        end
        push!(results, stage_results)
    end

    return results
end

# Call the function
# results = manage_reservoirs(n_weeks, n_scenarios, reservoirs, costs_approx)

# Print the results
# println(results)
