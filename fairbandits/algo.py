import numpy as np


class GeneralAlgo:
    """
    General class that subsume any Bandit or Fairness allocation algorithm
    """

    def __init__(self, lambdas, mu_tilde_estimate) -> None:
        """
        Parameters
        -----------
        lambdas: np array of shape K
            Fairness parameters
        mu_tilde_estimate: function of t, N, muhat
            Function that estimates mu_tilde
        """
        K = len(lambdas)
        # This trick is to deal with zero division
        self.lambdas = np.maximum(lambdas, np.ones(K) * 1e-15)
        self.muhat = np.zeros(K)  # Mean estimate
        self.N = np.zeros(K)  # Number of times each arm has been observed
        self.mu_tilde_estimate = mu_tilde_estimate
        self.t = 0  # Current timestep

    def play(self):
        pass

    def update(self, k, r):
        self.muhat[k] = (self.N[k] * self.muhat[k] + r) / (self.N[k] + 1)
        self.N[k] = self.N[k] + 1
        self.t += 1


class Fair(GeneralAlgo):
    def play(self):
        mu_tilde = self.mu_tilde_estimate(self.t, self.N, self.muhat, self.lambdas)
        return (self.lambdas / mu_tilde) / max(np.sum(self.lambdas / mu_tilde), 1)


class Bandit(GeneralAlgo):
    def play(self):
        mu_tilde = self.mu_tilde_estimate(self.t, self.N, self.muhat, self.lambdas)
        return np.argmax(mu_tilde)


class FairBandit:
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
            q = self.fairalgo.play()  # fair allocation
            k = self.bandit.play()  # bandit play
            p = np.copy(q)
            p[k] = q[k] + (1 - np.sum(q))
        return p

    def update(self, k, r):
        self.fairalgo.update(k, r)
        self.bandit.update(k, r)


def greedy(t, N, muhat, lambdas):
    return np.maximum(muhat, lambdas)


def clip(x, lambdas):
    return np.minimum(np.maximum(x, lambdas), 1)


def ucb(t, N, muhat, lambdas):
    return clip(muhat + np.sqrt(2 * np.log(t) / N), lambdas)


def lcb(t, N, muhat, lambdas):
    return clip(muhat - np.sqrt(2 * np.log(t) / N), lambdas)


def mu_opt(mus):
    return lambda x, y, z: mus


def mab_opt(mus, lambdas):
    p = lambdas / mus
    k = np.argmax(mus)
    p[k] = p[k] + (1 - np.sum(p))
    return p


def kl(p, q):
    if p == 0:
        res = -np.log(1 - q)
        return res

    if p == 1:
        res = -np.log(q)
        return res

    res = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    return res


def dichotomy(lower, upper, f, target, precision=1e-6):
    """Dichotomy to approach x_star such that f(x_star) = target for f
    monotone on interval [lower, upper].
    Returns x such that f(x) > f(x_start) and |x - x_star| <= precision.
    """
    if lower == upper:
        return lower

    mid = (lower + upper) / 2
    lmid = (lower + mid) / 2
    umid = (upper + mid) / 2
    increasing = f(lmid) < f(umid)

    n_iter = 0
    while lower + precision < upper:
        mid = (lower + upper) / 2
        if f(mid) > target:
            if increasing:
                upper = mid
            else:
                lower = mid
        else:
            if increasing:
                lower = mid
            else:
                upper = mid
        n_iter += 1
        if n_iter > 1e3:
            raise ValueError(
                "Convergence issue (%f, %f, %f, %f, %f)"
                % (lower, upper, f(lower), f(upper), target)
            )

    if increasing:
        return upper
    else:
        return lower


def kl_ucb(t, N, muhat, lambdas, precision=1e-6):
    """Solving kl(muhat, q) = 2 log(t)/N_k(t) for q greater than p"""
    return clip(
        np.array(
            [
                dichotomy(muhat[k], 1, lambda x: kl(muhat[k], x), 2 * np.log(t) / N[k])
                for k in range(len(N))
            ]
        ),
        lambdas,
    )


def kl_lcb(t, N, muhat, lambdas, precision=1e-6):
    """Solving kl(muhat, q) = 2 log(t)/t for q greater than p"""
    return clip(
        np.array(
            [
                dichotomy(0, muhat[k], lambda x: kl(muhat[k], x), 2 * np.log(t) / N[k])
                for k in range(len(N))
            ]
        ),
        lambdas,
    )
