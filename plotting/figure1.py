# Main paper experiments

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from fairbandits.algo import greedy, ucb, lcb
from joblib import load
from glob import glob
import os
from matplotlib.ticker import ScalarFormatter

cmap = plt.get_cmap("tab10")

rc = {
    "pdf.fonttype": 42,
    "text.usetex": True,
    "font.size": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "text.usetex": True,
    "font.family": "serif",
}
plt.rcParams.update(rc)

COLORS = {
    "greedy": cmap(6),
    "KL-UCB": cmap(3),
    "KL-LCB": cmap(7),
    "KL-LCB-UCB": cmap(2),
    "greedy-UCB": cmap(1),
    "ETC": cmap(4),
    "BanditQ": cmap(5),
}


LABELS = {
    "greedy": "GOC (with normalization)",
    "KL-UCB": "DOC",
    "KL-LCB": "POC (with normalization)",
    "KL-LCB-UCB": "SPOC",
    "greedy-UCB": "SGOC",
    "ETC": "P-SGOC (Section 4.2)",
    "BanditQ": "BanditQ",
}


MARKERS = {
    "greedy": None,
    "greedy-UCB": None,
    "KL-LCB-UCB": None,
    "KL-UCB": None,
    "KL-LCB": None,
    "ETC": "*",
    "BanditQ": "*",
}


LINESTYLE = {
    "greedy": "solid",
    "greedy-UCB": "solid",
    "KL-LCB-UCB": "dotted",
    "KL-UCB": "dashed",
    "KL-LCB": "solid",
    "ETC": "solid",
    "BanditQ": "solid",
}


# Experiments 5: Reproducing bandit Q setting
exp = "main_paper_exp5"
print(exp)
T = 100000
mus = np.array([0.335, 0.203, 0.241, 0.781, 0.617])
lambdas = np.array([0.167, 0.067, 0, 0, 0])
K = len(mus)
paths = [
    "./figures/%s_fairness" % (exp),
    "./figures/%s_bandit" % (exp),
    "./figures/%s_lb" % (exp),
]
cond = True
plot = load("../plotting/data/%s" % exp)


# %%

fig, axes = plt.subplots(ncols=4, figsize=(14, 2), sharex=True, frameon=False)


algos_pr = ["greedy-UCB", "KL-UCB",  "KL-LCB-UCB", "BanditQ"]
algos_all = ["greedy-UCB", "KL-UCB",  "KL-LCB-UCB", "ETC", "BanditQ"]

for algo in algos_pr:
    iterations = plot[algo][-1]
    axes[0].fill_between(
        iterations,
        plot[algo][5][0],
        plot[algo][5][1],
        color=COLORS[algo],
        alpha=0.1,
    )
    axes[0].plot(
        iterations,
        plot[algo][5][2],
        color=COLORS[algo],
        label=LABELS[algo],
        marker=MARKERS[algo],
        linestyle=LINESTYLE[algo],
    )
    axes[0].set_xlabel("Horizon (T)")
    axes[0].set_title(r"$\mathcal{R}_T$")
    axes[0].set_ylim([-100, 3500])

    axes[1].fill_between(
        iterations,
        plot[algo][4][0],
        plot[algo][4][1],
        color=COLORS[algo],
        alpha=0.1,
    )
    axes[1].plot(
        iterations,
        plot[algo][4][2],
        color=COLORS[algo],
        label=LABELS[algo],
        marker=MARKERS[algo],
        linestyle=LINESTYLE[algo],
    )
    axes[1].set_xlabel("Horizon (T)")
    axes[1].set_title("$\mathcal{V}_T$")
    axes[1].set_ylim([-100, 2300])

for algo in algos_all:
    iterations = plot[algo][-1]
    axes[2].fill_between(
        iterations,
        np.maximum(plot[algo][2][0], 0),
        np.maximum(plot[algo][2][1], 0),
        color=COLORS[algo],
        alpha=0.1,
    )
    axes[2].plot(
        iterations,
        np.maximum(plot[algo][2][2], 0),
        color=COLORS[algo],
        label=LABELS[algo],
        marker=MARKERS[algo],
        linestyle=LINESTYLE[algo],
    )
    axes[2].set_xlabel("Horizon (T)")
    axes[2].set_title("$\mathcal{R}_T^{LT}$")
    axes[3].fill_between(
        iterations,
        plot[algo][1][0],
        plot[algo][1][1],
        color=COLORS[algo],
        alpha=0.1,
    )
    axes[3].plot(
        iterations,
        plot[algo][1][2],
        color=COLORS[algo],
        label=LABELS[algo],
        marker=MARKERS[algo],
        linestyle=LINESTYLE[algo],
    )
    axes[3].set_xlabel("Horizon (T)")
    axes[3].set_title("$\mathcal{V}_T^{LT}$")

for ax in axes:
    ax.ticklabel_format(axis='both', style='sci', scilimits=(3,3))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.subplots_adjust(wspace=0.1)
plt.legend(ncols=5, loc="upper right", bbox_to_anchor=(0.5, 1.6))
plt.savefig("../figures/figure1.pdf", bbox_inches="tight")
