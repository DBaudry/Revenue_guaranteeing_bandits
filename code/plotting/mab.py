# Plot mab experiment
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from fairbandits.algo import greedy, ucb, lcb

seeds = 100
lambdas = np.array([0.2, 0.1, 0.25])
mus = np.array([0.6, 0.7, 0.5])
mu_estimates = [greedy, ucb, lcb]
K = len(lambdas)
T = int(1e5)

all_res = np.load("data/mab.npy")
all_res = all_res.reshape(seeds, len(mu_estimates), -1, T - (K + 1))

for n, name in enumerate(["Greedy", "UCB", "LCB"]):
    res = all_res[:, n]
    seeds, _, T = res.shape 
    res_med = np.mean(res, axis=0)
    res_high = np.mean(res, axis=0) + np.std(res, axis=0) / np.sqrt(seeds)
    res_low = np.mean(res, axis=0) - np.std(res, axis=0) / np.sqrt(seeds)

    measures = ["Fairness", "Regret"]
    fig, axes = plt.subplots(nrows=len(measures))
    plt.subplots_adjust(hspace=0.5)
    for i, measure in enumerate(measures):
        axes[i].plot(np.arange(T), res_med[i], color="red")
        axes[i].fill_between(np.arange(T), res_low[i], res_high[i], alpha=0.3)
        axes[i].set_title(measure)
        axes[i].set_xscale("symlog")
        axes[i].set_yscale("symlog")
        axes[i].grid()
    plt.savefig("figures/mab_%s.pdf" % name, bbox_inches="tight")
    plt.savefig("figures/mab_%s.png" % name)
    plt.close()
