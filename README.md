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
## Usage

The PFNN class can be used to estimate the state of a nonlinear system. The dataset need to be imported.

## Class Methods

- `__init__(self, state_dim, state_bounds, state_model, obs_model, particle_num)`: Initializes the PFNN class with the given parameters.
- `update(self, obs, ctrl)`: Updates the state space and weights based on the observation and control inputs.
- `eval_model(self, obs, ctrl)`: Evaluates the model by calculating the Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) between the predicted and actual observations.
- `plot(self, obs, ctrl)`: Plots the predicted vs actual observations.
