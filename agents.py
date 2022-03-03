import numpy as np


class Agent():

    def __init__(self, N_bandits, T, policy):
        self.N_bandits = N_bandits
        self.T = T
        self.policy = policy
        self.reset()

    def __str__(self):
        return f'{str(self.policy)}'

    def reset(self):
        """
        Reset the agent
        :return:
        """
        self.t = 0
        self.last_action = None
        self._last_actions = np.zeros(shape=(self.N_bandits,self.T))
        self._last_rewards = np.full(self.T, np.nan)

    def select_action(self):
        """
        Select an action
        :return:
        """
        action = self.policy.select_action()
        return action

    @property
    def value_estimates(self):
        return self._value_estimates[:self.t]

class BetaAgent(Agent):
    """
    An agent using Thomspon Sampling
    """
    def __init__(self, N_bandits, T, policy):
        super(BetaAgent, self).__init__(N_bandits, T, policy)
        self.alpha = np.zeros(N_bandits)
        self.beta = np.zeros(N_bandits)

    def update(self, action, reward, t):
        self.alpha[action] += reward
        self.beta[action] += (1-reward)
        self._last_actions[action, t] = action
        self._last_rewards[t] = reward
        self.t += 1

