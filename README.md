# Particle Filter for Nonlinear Systems (PFNN)

This repository contains a Python implementation of a Particle Filter for Nonlinear Systems (PFNN). The PFNN class is designed to estimate the state of a nonlinear system using a particle filter approach. This method is particularly useful for systems where the state dynamics are complex and cannot be easily modeled.

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib

## Installation

1. Clone this repository
2. Navigate to the cloned directory
3. Install the required Python packages:
```
pip install numpy scipy matplotlib
```

## Functionality

The Particle Filter Neural Network (PFNN) algorithm I've implemented combines particle filtering and neural networks to estimate the state of a dynamic system. It operates through several key steps:

1. **Initialization**: Sets up the state space and weights for the particle filter, representing possible states and their likelihoods.

2. **Update Method**: Updates the state space and weights based on observations and control inputs. This involves applying the state transition model to update the state space and the observation model to update the weights, reflecting the likelihood of each state.

3. **Evaluation Method**: Provides a quantitative measure of the model's performance by calculating the Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) between the predicted and actual observations.

4. **Plotting Method**: Offers a qualitative measure of the model's performance by visualizing the predicted vs. actual values.

5. **Integration of Particle Filtering and Neural Networks**: Integrates particle filtering and neural networks to estimate the state of the system. The particle filter updates the state space and weights based on the observations and control inputs, while the neural network models the system's dynamics, including the state transition and observation models.

This integration allows the PFNN algorithm to adapt to the complexities of the system being modeled, providing a robust and flexible approach to state estimation. The effectiveness of the PFNN algorithm depends on the accuracy of the state transition and observation models, as well as the appropriateness of the particle filter parameters.


## Class Methods

- `__init__(self, state_dim, state_bounds, state_model, obs_model, particle_num)`: Initializes the PFNN class with the given parameters.
- `update(self, obs, ctrl)`: Updates the state space and weights based on the observation and control inputs.
- `eval_model(self, obs, ctrl)`: Evaluates the model by calculating the Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) between the predicted and actual observations.
- `plot(self, obs, ctrl)`: Plots the predicted vs actual observations.
