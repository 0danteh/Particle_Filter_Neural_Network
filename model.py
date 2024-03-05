import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

class PFNN:
    def __init__(self, state_dim, state_bounds, state_model, obs_model, particle_num):
        self.state_dim=state_dim
        self.state_bounds=state_bounds
        self.state_model=state_model
        self.obs_model=obs_model
        self.particle_num=particle_num
        self.state_space=np.zeros((particle_num,state_dim))
        self.weights=np.ones(particle_num)/particle_num
        for i in range(state_dim):
            self.state_space[:,i]=np.random.uniform(state_bounds[i][0], state_bounds[i][0], particle_num)
    
    def update(self,obs,ctrl):
        # Update the state space through the state transition model
        if isinstance(self.state_model, np.ndarray):
            noise = np.random.multivariate_normal(np.zeros(self.state_dim), np.eye(self.state_dim), self.particle_num)
            self.state_space = self.state_space @ self.state_model.T + ctrl @ self.state_model.T + noise
        else: 
            for i in range(self.particle_num):
                self.state_space[i] = self.state_model(self.state_space[i], ctrl, np.random.randn(self.state_dim))
        # Update the weights using the observation model and the observation likelihood
        if isinstance(self.obs_model, np.ndarray):
            obs_pred = self.state_space @ self.obs_model.T
            obs_mean = np.mean(obs_pred, axis=0)
            obs_likelihood = stats.multivariate_normal.pdf(obs_pred, mean=obs_mean, cov=np.eye(self.state_dim))
        else:
            obs_pred = obs_pred = np.zeros((self.particle_num, self.state_dim))
            for i in range(self.particle_num):
                obs_pred[i] = self.obs_model(self.state_space[i], np.random.randn(self.state_dim))
            obs_mean = np.mean(obs_pred, axis=0)
            obs_likelihood = stats.multivariate_normal.pdf(obs_pred, mean=obs_mean, cov=np.eye(self.state_dim))
        # Update the weights using the observation likelihood and the prior distribution
        prior = stats.uniform.pdf(self.state_space, loc=self.state_bounds[:, 0], scale=self.state_bounds[:, 1] - self.state_bounds[:, 0])
        prior = np.prod(prior, axis=1)
        self.weights = self.weights * obs_likelihood * prior
        self.weights = (self.weights + 1e-9) / np.sum(self.weights + 1e-9)
        # Handling NaN values
        nan_indices = np.isnan(self.weights)
        self.weights = self.weights[~nan_indices]
        self.state_space = self.state_space[~nan_indices]
        self.particle_num = len(self.weights)
        return self.state_space, self.weights, self.particle_num

    def eval_model(self,obs,ctrl):
        pred = self.update(obs,ctrl)
        diff_abs = np.abs(obs-pred[0])
        diff_sq = np.square(obs-pred[0])
        MAE = np.mean(diff_abs)
        RMSE = np.mean(diff_sq)
        RMSE = np.sqrt(RMSE)
        print("MAE:",MAE)
        print("RMSE:",RMSE)
    
    def plot(self,obs,ctrl):
        pred = self.update(obs,ctrl)
        plt.plot(pred[0][:,0], pred[0][:,1], 'bo', label='Predicted')
        plt.plot(obs[:,0], obs[:,1], 'rx', label='Actual')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Predicted vs Actual')
        plt.legend()
        plt.show()
