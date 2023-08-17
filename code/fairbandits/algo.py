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
        self.muhat = np.zeros(K)  # Mean estimate
        self.N =  np.zeros(K) # Number of times each arm has been observed
        self.mu_tilde_estimate = mu_tilde_estimate
        self.t = 0 # Current timestep

    def play(self):
        pass

    def update(self, k, r):
        # print(self.t)
        # print(self.muhat, k, self.N[k], r)
        self.muhat[k] = (self.N[k] * self.muhat[k] + r) / (self.N[k] + 1)
        self.N[k] = self.N[k] + 1
        # print(self.muhat, self.N[k])
        # print("---------------")
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
        K = len(self.bandit.lambdas)
        t = self.bandit.t
        if t < K:
            p = np.zeros(K)
            p[t] = 1
            return p
        else:
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

def clip(x):
    return np.minimum(np.maximum(x, 0), 1)

def ucb(t, N, muhat):
    f = lambda x: 1 + x * np.log(x)**2
    return clip(muhat + np.sqrt(2 * np.log(f(t)) / N ))

def lcb(t, N, muhat):
    f = lambda x: 1 + x * np.log(x)**2
    return clip(muhat - np.sqrt(2 * np.log(f(t)) / N ))

def mu_opt(mus):
    return lambda x, y, z: mus

def mab_opt(mus, lambdas):
    p = lambdas / mus
    k = np.argmax(mus)
    p[k] = p[k] + (1 - np.sum(p))
    return p
