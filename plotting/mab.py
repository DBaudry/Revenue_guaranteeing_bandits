# Plot mab experiment
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from fairbandits.algo import greedy, ucb, lcb
from joblib import load

seeds, lambdas, mus, mu_estimates, K, T, all_res = load("data/mab_same_estimates")

for n, name in enumerate(mu_estimates):
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
    plt.savefig("figures/mab_%s_%s.pdf" % (name, name), bbox_inches="tight")
    plt.savefig("figures/mab_%s_%s.png" % (name, name), bbox_inches="tight")
    plt.close()



seeds, lambdas, mus, mu_estimate2, mu_estimates, K, T, all_res = load("data/mab_ucb_bandit_all_fair")

for n, name in enumerate(mu_estimates):
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
    plt.savefig("figures/mab_%s_%s.pdf" % (name, mu_estimate2), bbox_inches="tight")
    plt.savefig("figures/mab_%s_%s.png" % (name, mu_estimate2), bbox_inches="tight")
    plt.close()


seeds, lambdas, mus, mu_estimate2, mu_estimates, K, T, all_res = load("data/mab_greedy_bandit_all_fair")

for n, name in enumerate(mu_estimates):
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
    plt.savefig("figures/mab_%s_%s.pdf" % (name, mu_estimate2), bbox_inches="tight")
    plt.savefig("figures/mab_%s_%s.png" % (name, mu_estimate2), bbox_inches="tight")
    plt.close()
