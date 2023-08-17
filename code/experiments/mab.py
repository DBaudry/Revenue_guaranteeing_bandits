# MAB experiment

import numpy as np
from fairbandits.algo import FairBandit, Fair, Bandit, greedy, ucb, lcb, mab_opt
from fairbandits.environment import mab_environment
from joblib import Parallel, delayed


def do_expe(seed, lambdas, mus, T, mu_estimate):
    mab_algo = FairBandit(Fair(lambdas, mu_estimate), Bandit(lambdas, mu_estimate))
    p_opt = mab_opt(mus, lambdas)
    rng = np.random.RandomState(seed)
    cum_regrets = []
    cum_regrets_ub = []
    cum_fairness = []
    abs_cum_deviation = []
    cum_deviation_q = []
    cum_regret_t = 0
    cum_regret_ub_t = 0
    cum_fairness_t = 0
    cum_deviation_t = 0
    cum_deviation_q_t = 0
    for t in range(T):
        p_t = mab_algo.play()
        muhat = mab_algo.fairalgo.muhat
        k_t, r_t = mab_environment(p_t, mus, rng)
        # print(k_t, p_t, p_opt, mab_algo.fairalgo.muhat, mus)
        mab_algo.update(k_t, r_t)
        if t <= K:
            continue
        cum_regret_t  +=  np.sum((p_opt - p_t) * mus)
        cum_fairness_t  +=  np.max(lambdas - p_t * mus)
        cum_deviation_t  +=  p_t - p_opt
        if np.sum(muhat == 0) == 0:
            # cum_deviation_q_t  +=  np.sum(lambdas / muhat - lambdas / mus)
            cum_deviation_q_t  +=  np.sum(muhat - mus)
        # print(p_t, p_opt)
        cum_regrets.append(cum_regret_t)
        cum_fairness.append(cum_fairness_t)
        abs_cum_deviation.append(np.min(np.abs(cum_deviation_t)))
        cum_deviation_q.append(cum_deviation_q_t)
    return cum_fairness, cum_regrets, abs_cum_deviation, cum_deviation_q


seeds = 100
lambdas = np.array([0.2, 0.1, 0.25])
mus = np.array([0.6, 0.7, 0.5])
mu_estimates = [greedy, ucb, lcb]
K = len(lambdas)
T = int(1e5)

res = np.array(Parallel(n_jobs=-1)(delayed(do_expe)(seed, lambdas, mus,T, mu_estimate) for seed in range(seeds) for mu_estimate in mu_estimates))

