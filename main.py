import numpy as np
import pandas as pd
from bandits import BinomialBandit
from agents import BetaAgent
from policies import ThompsonSampling


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    Q_bandits = [0.4, 0.8]
    N_bandits = len(Q_bandits)

    bandits = []
    for i in range(N_bandits):
        bandits.append(BinomialBandit(Q_bandits[i]))

    T = 1000
    ts_sampling_policy = ThompsonSampling()
    agent = BetaAgent(N_bandits, T, ts_sampling_policy)

    for t in range(T):
        action = agent.select_action()
        reward = bandits[action].sample()
        agent.update(action, reward, t)
