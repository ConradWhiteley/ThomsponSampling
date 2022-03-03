import numpy as np

class Bandit():
    """
    A Bandit
    """

    def __init__(self):
        pass


class BinomialBandit(Bandit):
    """
    A Bernoulli Bandit
    """
    def __init__(self, q):
        super(BinomialBandit, self).__init__()
        self.q = q

    def sample(self):
        return np.random.binomial(1, self.q)