import numpy as np
import warnings

class BanditQ:
    def __init__(self, lambdas, T) -> None:
        self.lambdas = lambdas
        d = len(lambdas)
        self.Q = np.zeros(d)
        self.p = np.ones(d) / d
        self.cum_r =np.zeros(d)
        self.eta = d
        self.S = 1
        self.V = np.sqrt(T)
        self.t = 0


    def play(self):
        """
        return p (np.ndarray of size K): p_i is the proba of playing arm i
        """
        d = len(self.Q)
        if self.t == 0:
            gamma = 1/2
        else:
            gamma = min(1/2, np.sqrt(d/self.t))

        x = (1- gamma) * self.p + gamma / (d * np.ones(d))
        return x

    def update(self, k, r, x):
        """
        Parameters
        k (int): chosen arm
        r (float): obtained reward
        x (np.ndarray): previous play
        """
        d = len(self.Q)
        self.Q = self.Q + self.lambdas
        self.Q[k] = self.Q[k] - r
        self.Q = np.maximum(self.Q, 0)
        self.cum_r[k] += (self.Q[k] + self.V)* r / x[k]
        p = self.p
        q = maxlog(r - 1/p)
        breg = - np.sum(np.log(q)) + np.sum(np.log(p)) + np.sum((q - p) / p)
        self.S = self.S + self.eta * (np.dot(r, q - p) - breg)
        eta = d / self.S
        self.p = maxlog(self.cum_r * eta)
        self.t += 1


def maxlog(r, tol=1e-8, max_iter=100):
    """
    Solves for x in the d-dimensional probability simplex
    max sum_{i=1}^d log(x_i) + sum_{i=1}^d r_i x_i
    """
    mu = -np.max(r) - 1
    err = np.abs(-np.sum(1/(r + mu)) - 1)
    for i in range(max_iter):
        inv = -1/(r + mu)
        num = -(np.sum(inv) - 1)
        denum = np.sum(inv ** 2)
        mu = mu + num/denum
        err = np.abs(-np.sum(1/(r + mu)) - 1)
        if err < tol:
            return -1 / (r + mu)
    warnings.warn("MaxLog did not converge, current error is %f" % err)
    return -1 / (r + mu)


class ETC:
    def __init__(self, lambdas, T, f, c, w):
        self.lambdas = lambdas
        self.T = T
        self.phase = 1
        self.t = 0
        self.c = c
        self.f = f
        self.w = w
        self.muhat = np.zeros_like(lambdas)  # Mean estimate
        self.muhat1 = np.zeros_like(lambdas)  # Mean estimate
        self.muhat2 = np.zeros_like(lambdas)  # Mean estimate
        self.mu_estimate1 = np.zeros_like(lambdas)
        self.mu_estimate2 = np.zeros_like(lambdas)
        self.N = np.zeros_like(lambdas)
        self.N1 = np.zeros_like(lambdas)
        self.N2 = np.zeros_like(lambdas)
        self.N3 = np.zeros_like(lambdas)

    def play(self):
        K = len(self.lambdas)
        T = self.T
        lambdas = self.lambdas
        c = self.c
        f = self.f
        w = self.w

        self.phase = 1
        for k in range(K):
            if self.N1[k] <  self.f[k] * T:
                p = np.zeros(K)
                p[k] = 1
                return p

        self.phase = 2
        for k in range(K):
            if self.N2[k] < c * lambdas[k]/self.mu_estimate1[k]* T:
                p = np.zeros(K)
                p[k] = 1
                return p

        self.phase = 3
        for k in range(K):
            condk = lambdas[k] / self.mu_estimate2[k] 
            condk -= c * lambdas[k] / self.mu_estimate1[k]
            condk -= f[k]
            condk -=  w[k] * max((np.sum(lambdas/self.mu_estimate2) - 1), 0)
            condk = max(condk, 0) * T
            if self.N3[k] < condk:
                p = np.zeros(K)
                p[k] = 1
                return p

        self.phase = 4
        p = np.zeros(K)
        i = np.argmax(kl_ucb(self.t, self.N, self.muhat, self.lambdas))
        p[i] = 1
        return p


    def update(self, k, r, x=None):
        self.muhat[k] = (self.N[k] * self.muhat[k] + r) / (self.N[k] + 1)

        self.N[k] = self.N[k] + 1
        self.t += 1
        if self.phase == 1:
            self.muhat1 = self.muhat
            self.mu_estimate1 = clip(self.muhat1, self.lambdas)
            self.N1[k] += 1

        if self.phase == 2:
            self.muhat2[k] = (self.N2[k] * self.muhat2[k] + r) / (self.N2[k] + 1)
            self.mu_estimate2 = clip(self.muhat2, self.lambdas)
            self.N2[k] += 1

        if self.phase == 3:
            self.N3[k] += 1


class ETCAnytime:
    def __init__(self, lambdas, T, rng:np.random.RandomState):
        self.lambdas = lambdas
        self.T = T
        self.N = np.zeros_like(lambdas)
        self.N1 = np.zeros_like(lambdas)
        self.muhat = np.zeros_like(lambdas)
        self.muhat1 = np.zeros_like(lambdas)
        self.mu_estimate1 = clip(self.muhat1, self.lambdas)
        self.t = 0
        self.rng = rng
        self.i = 0

    def play(self):
        lambdas = self.lambdas
        K = len(lambdas)

        proba = np.zeros(3)
        proba[0] = np.sum(lambdas)

        if np.sum(lambdas / self.mu_estimate1) <= 1:
            proba[1] = np.sum(lambdas / self.mu_estimate1 - lambdas)
        else:
            proba[1] = np.sum(lambdas / self.mu_estimate1 - lambdas)  / np.sum(lambdas / self.mu_estimate1 - lambdas) * (1 - np.sum(lambdas))
        proba[2] = 1 - np.sum(proba[:2])


        i = self.rng.choice([0, 1, 2], p=proba)
        self.i = i

        if i == 0:
            p = lambdas
        if i == 1:
            p = lambdas / self.mu_estimate1 - lambdas
        if i == 2:
            p = np.zeros(K)
            k = np.argmax(kl_ucb(self.t, self.N, self.muhat, self.lambdas))
            p[k] = 1
        return p / np.sum(p)


    def update(self, k, r, x=None):
        self.muhat[k] = (self.N[k] * self.muhat[k] + r) / (self.N[k] + 1)
        self.N[k] = self.N[k] + 1
        self.t += 1
        if self.i == 1:
            self.muhat1[k] = (self.N1[k] * self.muhat1[k] + r) / (self.N1[k] + 1)
            self.mu_estimate1 = clip(self.muhat1, self.lambdas)
            self.N1[k] += 1


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
        self.lambdas = lambdas
        self.muhat = np.zeros(K)  # Mean estimate
        self.N = np.zeros(K)  # Number of times each arm has been observed
        self.mu_tilde_estimate = mu_tilde_estimate
        self.t = 0  # Current timestep

    def play(self):
        pass

    def update(self, k, r, x=None):
        self.muhat[k] = (self.N[k] * self.muhat[k] + r) / (self.N[k] + 1)
        self.N[k] = self.N[k] + 1
        self.t += 1


class Fair(GeneralAlgo):
    def play(self):
        mu_tilde = self.mu_tilde_estimate(self.t, self.N, self.muhat, self.lambdas)
        I = self.lambdas != 0
        p = np.zeros_like(mu_tilde)
        p[I] =  self.lambdas[I] / mu_tilde[I]
        if np.sum(p) > 1:
            return p / np.sum(p)
        else:
            return p


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

    def update(self, k, r, x=None):
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
    return lambda w, x, y, z: mus


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
