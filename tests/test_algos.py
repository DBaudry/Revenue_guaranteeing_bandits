# MAB experiment

from fairbandits.algo import FairBandit, Fair, Bandit, greedy, ucb, lcb, mab_opt, mu_opt, kl_ucb, kl, kl_lcb, maxlog, LagrangeBwK
from fairbandits.environment import mab_environment
from joblib import Parallel, delayed
import numpy as np
from fairbandits.algo import BanditQ

def do_exp(seed, lambdas, mus, T, mab_algo):
    """
    Compute the cumulative fairness regret, cumulative regret and absolute cumulative deviation
    """
    p_opt = mab_opt(mus, lambdas)
    rng = np.random.RandomState()
    cum_regrets = []
    cum_fairness = []
    abs_cum_deviation = []
    cum_regret_t = 0
    cum_fairness_t = 0
    cum_deviation_t = 0
    iterations = []
    for t in range(T):
        p_t = mab_algo.play()
        k_t, r_t = mab_environment(p_t, mus, rng)
        mab_algo.update(k_t, r_t, p_t)
        cum_regret_t += np.sum((p_opt - p_t) * mus)
        cum_fairness_t += np.max(lambdas - p_t * mus)
        cum_deviation_t += np.abs(p_t - p_opt)
        cum_regrets.append(cum_regret_t)
        cum_fairness.append(cum_fairness_t)
        abs_cum_deviation.append(np.copy(cum_deviation_t))
        iterations.append(t)
    return iterations, cum_fairness, cum_regrets, abs_cum_deviation, p_t

def test_lagrangian():
    T = 10000
    cum_fairnesses = []
    cum_regret = []
    ts = []
    # for T in np.arange(1, 6) * 10000:
    for Tt in np.arange(1, 10) * 10000:
        T  = int(Tt)
        print(T)
        feasibility_gap = 0.95
        mus = np.array([0.7, 0.9, 0.8])
        lambdas = mus / len(mus) * (1 - feasibility_gap)
        lambdas = np.array([0.25, 0.25, 0.25])
        print(np.sum(lambdas/mus))
        cum_regret_t = 0
        cum_fairness_t = 0
        algo = LagrangeBwK(lambdas, T)
        rng = np.random.RandomState(None)
        p_opt = mab_opt(mus, lambdas)
        print("OPT", p_opt)
        p_avg  = np.zeros_like(p_opt)
        etas = []
        for t in range(T):
            p_t = algo.play()
            k_t, r_t = mab_environment(p_t, mus, rng)
            algo.update(k_t, r_t, p_t)
            cum_regret_t += np.sum((p_opt - p_t) * mus)
            cum_fairness_t += lambdas - p_t * mus
            etas.append(algo.eta)
            p_avg += p_t
        print(p_avg / T)
        cum_fairnesses.append(cum_fairness_t/t)
        cum_regret.append(cum_regret_t / t)
        ts.append(T)
            

    import matplotlib.pyplot as plt
    f, axes = plt.subplots(nrows=2)
    axes[0].set_title("Fairness")
    axes[0].plot(ts, cum_fairnesses, marker="*")
    axes[0].axhline(0)
    axes[1].set_title("Regret")
    axes[1].plot(ts, cum_regret, marker="*")
    axes[1].axhline(0)
    plt.legend()
    plt.show()

    print(p_avg, p_opt)
    print(cum_regret_t)

def test_banditq():
    T = 5000
    feasibility_gap = 0.1
    mus = np.array([0.8, 0.9, 0.7])
    lambdas = mus / len(mus) * (1 - feasibility_gap)
    cum_regret_t = 0
    cum_fairness_t = 0
    algo = BanditQ(lambdas, T)
    rng = np.random.RandomState(None)
    p_opt = mab_opt(mus, lambdas)
    p_avg  = np.zeros_like(p_opt)
    etas = []
    cum_fairnesses = []
    ts = []
    for t in range(T):
        p_t = algo.play()
        k_t, r_t = mab_environment(p_t, mus, rng)
        algo.update(k_t, r_t, p_t)
        cum_regret_t += np.sum((p_opt - p_t) * mus)
        cum_fairness_t += lambdas - p_t * mus
        etas.append(algo.eta)
        p_avg += p_t
        if t % (T // 1000) == 0:
            cum_fairnesses.append(cum_fairness_t/t)
            ts.append(t)
    p_avg = p_avg / T

    import matplotlib.pyplot as plt
    plt.plot(ts, cum_fairnesses)
    plt.axhline(0)
    plt.show()

    print(p_avg, p_opt)
    print(cum_regret_t)




def test_opt():
    seed = 5
    lambdas = np.array([0.2, 0.1, 0.25])
    mus = np.array([0.6, 0.7, 0.5])
    K = len(lambdas)
    T = int(1e2)
    mab_algo = FairBandit(Fair(lambdas, mu_opt(mus)), Bandit(lambdas, mu_opt(mus)))
    _, cum_fairness, cum_regrets, abs_cum_deviation, _ = do_exp(seed, lambdas, mus, T, mab_algo )
    # To remove RR steps
    cum_fairness = cum_fairness[4:] - cum_fairness[4]
    cum_regrets = cum_regrets[4:] - cum_regrets[4]
    abs_cum_deviation = abs_cum_deviation[4:] - abs_cum_deviation[4]
    np.testing.assert_equal(np.linalg.norm(cum_regrets), 0)
    np.testing.assert_equal(np.linalg.norm(cum_fairness), 0)
    np.testing.assert_equal(np.linalg.norm(abs_cum_deviation), 0)


def test_klucb():
    N = np.array([56, 230, 34])
    mu = np.array([0.3, 0.5, 0.9])
    t = np.sum(N)
    X = np.array([np.random.binomial(N[k], mu[k]) for k in range(len(N))])
    mu_hat = X / N
    mu_UCB = kl_ucb(t, N, mu_hat, np.zeros(3))
    for k in range(len(mu)):
        assert mu_UCB[k] > mu_hat[k]
        assert kl(mu_hat[k], mu_UCB[k]) >  2*np.log(t) / N[k]
        assert kl(mu_hat[k], mu_UCB[k] - 2e-6) <  2*np.log(t) / N[k]


def test_kllcb():
    N = np.array([56, 230, 34])
    mu = np.array([0.3, 0.5, 0.9])
    t = np.sum(N)
    X = np.array([np.random.binomial(N[k], mu[k]) for k in range(len(N))])
    mu_hat = X / N
    mu_LCB = kl_lcb(t, N, mu_hat, np.zeros(3))
    for k in range(len(mu)):
        assert mu_LCB[k] < mu_hat[k]
        assert kl(mu_hat[k], mu_LCB[k]) >  2*np.log(t) / N[k]
        assert kl(mu_hat[k], mu_LCB[k] + 2e-6) <  2*np.log(t) / N[k]

def test_banditQ():
    mus = np.array([0.335, 0.203, 0.781])
    lambdas = np.array([0, 0, 0])
    T = int(1e3)
    mab = BanditQ(lambdas, T)
    p_t = do_exp(1, lambdas, mus, T, mab)[-1]
    assert np.argmax(p_t) == np.argmax(mus)



def test_maxlog():
    max_iter=10
    r = np.array([1, 5])

    def val(x):
        return np.sum(np.log(x)) + r.dot(x)

    x =  maxlog(r)
    for _ in range(10):
        eps = np.random.randn(2) * 1e-3
        x_test = (x + eps)
        x_test = np.maximum(x_test, 0)
        x_test = x_test/ np.sum(x_test)
        assert val(x_test) < val(x)
