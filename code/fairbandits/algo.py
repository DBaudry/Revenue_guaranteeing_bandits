import numpy as np

class GeneralAlgo():
    """
    General class that subsume any Bandit or Fairness allocation algorithm
    """
    def __init__(self, lambdas, mu_tilde_estimate) -> None:
        """
        Parameters
        -----------
        lambas: np array of shape K
            Fairness parameters
        mu_tilde_estimate: function of t, N, muhat 
            Function that estimates mu_tilde
        """
        K = len(lambdas)
        self.lambdas = lambdas
        self.muhat = np.ones(K)  # Mean estimate
        self.N =  np.zeros(K) # Number of times each arm has been observed
        self.mu_tilde_estimate = mu_tilde_estimate
        self.t = 0 # Current timestep

    def play(self):
        pass

    def update(self, k, r):
        self.muhat[k] = (self.N[k] * self.muhat[k] + r) / (self.N[k] + 1)
        self.N[k] = self.N[k] + 1
        self.t += 1


class Fair(GeneralAlgo):
    def play(self):
        mu_tilde = self.mu_tilde_estimate(self.t, self.N, self.muhat)
        if np.sum(mu_tilde == 0) > 0:
            K = len(self.lambdas)
            q = np.zeros(K)
            q[mu_tilde == 0] = self.lambdas[mu_tilde == 0] / np.sum(self.lambdas[mu_tilde == 0])
            return q
        else:
            return (self.lambdas / mu_tilde) / max(np.sum(self.lambdas / mu_tilde), 1)
        
class Bandit(GeneralAlgo):
    def play(self):
        mu_tilde = self.mu_tilde_estimate(self.t, self.N, self.muhat)
        return np.argmax(mu_tilde)

class FairBandit():
    def __init__(self, fairalgo, bandit) -> None:
        self.bandit = bandit
        self.fairalgo = fairalgo

    def play(self):
        q = self.fairalgo.play() # fair allocation
        k = self.bandit.play() # bandit play
        p = np.copy(q)
        p[k] = q[k] + (1 - np.sum(q))
        return p

    def update(self, k, r):
        self.fairalgo.update(k, r)
        self.bandit.update(k, r)


def greedy(t, N, muhat):
    return muhat

def mab_opt(mus, lambdas):
    p = lambdas / mus
    k = np.argmax(mus)
    p[k] = p[k] + (1 - np.sum(p))
    return p
