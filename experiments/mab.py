# MAB experiment

import numpy as np
from fairbandits.algo import FairBandit, Fair, Bandit, greedy, ucb, lcb, mab_opt
from fairbandits.environment import mab_environment
from joblib import Parallel, delayed, dump
from pdb import set_trace


def do_exp(seed, lambdas, mus, T, mu_estimate1, mu_estimate2):
    """
    Compute the cumulative fairness regret, cumulative regret and absolute cumulative deviation
    """
    mab_algo = FairBandit(Fair(lambdas, mu_estimate1), Bandit(lambdas, mu_estimate2))
    p_opt = mab_opt(mus, lambdas)
    rng = np.random.RandomState(seed)
    cum_regrets = []
    cum_fairness = []
    abs_cum_deviation = []
    cum_regret_t = 0
    cum_fairness_t = 0
    cum_deviation_t = 0
    for t in range(T):
        # print(mab_algo.bandit.play())
        p_t = mab_algo.play()
        k_t, r_t = mab_environment(p_t, mus, rng)
        mab_algo.update(k_t, r_t)
        cum_regret_t += np.sum((p_opt - p_t) * mus)
        cum_fairness_t += np.max(lambdas - p_t * mus)
        cum_deviation_t += p_t - p_opt
        cum_regrets.append(cum_regret_t)
        cum_fairness.append(cum_fairness_t)
        abs_cum_deviation.append(np.min(np.abs(cum_deviation_t)))
    return cum_fairness, cum_regrets, abs_cum_deviation


seeds = 200
mus = np.array([0.6, 0.7, 0.5])
feasability_gap = 0.5
lambdas = np.array([0.6, 0.7, 0.5]) / len(mus) * feasability_gap
mu_estimates = [greedy, ucb, lcb]
names_estimates = ["Greedy", "UCB", "LCB"]
K = len(lambdas)
T = int(1e4)

res = np.array(Parallel(n_jobs=-1)(delayed(do_exp)(seed, lambdas, mus,T, mu_estimate, mu_estimate) for seed in range(seeds) for mu_estimate in mu_estimates))
dump([seeds, lambdas, mus, names_estimates, K, T, res.reshape(seeds, len(mu_estimates), -1, T)],"data/mab_same_estimates")


res = np.array(Parallel(n_jobs=-1)(delayed(do_exp)(seed, lambdas, mus,T, mu_estimate, ucb) for seed in range(seeds) for mu_estimate in mu_estimates))
dump([seeds, lambdas, mus, "UCB", names_estimates, K, T, res.reshape(seeds, len(mu_estimates), -1, T)],"data/mab_ucb_bandit_all_fair")


res = np.array(Parallel(n_jobs=-1)(delayed(do_exp)(seed, lambdas, mus,T, mu_estimate, greedy) for seed in range(seeds) for mu_estimate in mu_estimates))
dump([seeds, lambdas, mus, "Greedy", names_estimates, K, T, res.reshape(seeds, len(mu_estimates), -1, T)],"data/mab_greedy_bandit_all_fair")
