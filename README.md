# antares-water-values
Water values computation for Antares (https://antares-doc.readthedocs.io/en/latest/)

## Installation
```bash
git clone https://github.com/AntaresSimulatorTeam/antares-water-values.git
```
## Requirements
* Python 3.9
* Antares Simulator
* [Python libraries requirements](requirements.txt)

## Usage

```python
# Define time and scenario parameters
param = TimeScenarioParameter(len_week=5, len_scenario=1)

# Define reservoir parameters
reservoir = Reservoir("test_data/one_node", "area")
reservoir_management = ReservoirManagement(
    reservoir=reservoir,
    penalty_bottom_rule_curve=3000,
    penalty_upper_rule_curve=3000,
    penalty_final_level=3000,
    force_final_level=False,
)

# Define discretization of stock level
xNsteps = 20
X = np.linspace(0, reservoir.capacity, num=xNsteps)

# Generate mps files 
generate_mps_file(study_path=study_path,antares_path=antares_path)

# Compute Bellman values directly
vb = calculate_bellman_value_directly(
    param=param,
    reservoir_management=reservoir_management,
    output_path="test_data/one_node",
    X=X,
)

# or with precalulated reward
vb, G = calculate_bellman_value_with_precalculated_reward(
        len_controls=20,
        param=param,
        reservoir_management=reservoir_management,
        output_path="test_data/one_node",
        X=X,
    )

# or with iterative algorithm
vb, G, _, _, controls_upper, traj = itr_control(
        param=param,
        reservoir_management=reservoir_management,
        output_path="test_data/one_node",
        X=X,
        N=3,
        tol_gap=1e-4,
    )
```
## Structure

The repository consists in:
- [src](./src):
  python package to launch simulations and compute Bellman values based on the result
- [tests](./tests):
  python tests illustrating the use and behaviour of the concepts
- [test_data](./test_data): data for running tests
