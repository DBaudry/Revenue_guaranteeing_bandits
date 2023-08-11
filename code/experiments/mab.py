# MAB experiment

import numpy as np
from fairbandits.algo import FairBandit, Fair, Bandit, greedy, mab_opt
from fairbandits.environment import mab_environment
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed


seeds = 50
def do_expe(seed, lambdas, mus, T):
    mab_algo = FairBandit(Fair(lambdas, greedy), Bandit(lambdas, greedy))
    p_opt = mab_opt(mus, lambdas)
    rng = np.random.RandomState(seed)
    cum_regrets = []
    cum_fairness = []
    abs_cum_deviation = []
    cum_regret_t = 0
    cum_fairness_t = 0
    cum_deviation_t = 0
    for t in range(T):
        p_t = mab_algo.play()
        k_t, r_t = mab_environment(p_t, mus, rng)
        # print(k_t, p_t, p_opt, mab_algo.fairalgo.muhat, mus)
        mab_algo.update(k_t, r_t)
        cum_regret_t  +=  np.sum((p_opt - p_t) * mus)
        cum_fairness_t  +=  np.max(lambdas - p_t * mus)
        cum_deviation_t  +=  np.sum(p_t - p_opt)
        # print(p_t, p_opt)
        cum_regrets.append(cum_regret_t)
        cum_fairness.append(cum_fairness_t)
        abs_cum_deviation.append(np.abs(cum_deviation_t))
    return cum_fairness, cum_regrets, abs_cum_deviation


lambdas = np.array([0.1, 0.1, 0.3])
mus = np.array([0.3, 0.4, 1])
K = len(lambdas)
T = int(1e4)

res = np.array(Parallel(n_jobs=-1)(delayed(do_expe)(seed, lambdas, mus, T) for seed in range(seeds)))

res_med = np.mean(res, axis=0)
res_high = np.quantile(res, q=0.9, axis=0)
res_low = np.quantile(res, q=0.1, axis=0)

fig, axes = plt.subplots(nrows=3)
plt.subplots_adjust(hspace=0.5)
axes[0].plot(np.arange(T), res_med[0], color="red")
axes[0].fill_between(np.arange(T), res_low[0], res_high[0], alpha=0.3)
axes[0].set_title("Fairness")
axes[1].plot(np.arange(T),res_med[1], color="red")
axes[1].fill_between(np.arange(T),res_low[1], res_high[1], alpha=0.3)
axes[1].set_title("Regret")
axes[2].plot(np.arange(T),res_med[2], color="red")
axes[2].fill_between(np.arange(T),res_low[2], res_high[2], alpha=0.3)
axes[2].set_title("Deviation")
plt.savefig("../figures/mab.pdf")
plt.close()


# np.save("../data/mab_regret", cum_regrets)
# np.save("../data/mab_fairness", cum_regrets)
