import gym
from gym import spaces
import numpy as np

#create env for algorithm.py
class PortfolioEnv(gym.Env):
    def __init__(self, data, window_size=9, transaction_cost=0.002, slippage=0.005):
        super(PortfolioEnv, self).__init__()
        
        self.data = data
        self.window_size = window_size
        self.n_assets = data.shape[1]
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
       
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        
       
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, self.n_assets), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        self.current_step = self.window_size
        self.portfolio = np.zeros(self.n_assets)
        for i in range(self.n_assets):
            self.portfolio[i] = 1 / self.n_assets

        return self._get_state()
    
    def _get_state(self):
        price_data = self.data[self.current_step - self.window_size:self.current_step]
        return price_data
    
    def step(self, action):
        prev_portfolio = self.portfolio
        self.portfolio = action / np.sum(action)  
        
        reward = self._calculate_reward(prev_portfolio, self.portfolio)
        
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        return np.array(self._get_state().tolist() + [self.portfolio.tolist()]), reward, done, {}
    
    def _calculate_reward(self, prev_portfolio, new_portfolio):
        close_prices = self.data[self.current_step]
        prev_close_prices = self.data[self.current_step -1]
        returns = (close_prices/ prev_close_prices)            
        returns = np.transpose([returns])
        portfolio_return = np.dot(new_portfolio, returns)
        if portfolio_return < 1:
            portfolio_return = -1
            return float(portfolio_return)
        else:
            return float(portfolio_return)

    def render(self, mode='human', close=False):
        pass