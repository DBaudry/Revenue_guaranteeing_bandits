# Appendix Figure 2: Long term results with various feasability gaps

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


exp = "main_paper_exp1"
plots = []
for feasibility_gap in [0, 0.1, 0.5, 0.9]:
    mus = np.array([0.8, 0.9, 0.7])
    K = len(mus)
    plots.append(load("data/%s_%s" % (exp, str(feasibility_gap))))



# %%

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(14, 12), sharex=True, frameon=False)


algos_pr = ["greedy-UCB", "KL-UCB",  "KL-LCB-UCB", "BanditQ"]
algos_all = ["greedy-UCB", "KL-UCB",  "KL-LCB-UCB", "ETC", "BanditQ"]

for algo in algos_pr:
    for z, plot in enumerate(plots):
        if z == 3:
            label = LABELS[algo]
        else:
            label = None
        iterations = plot[algo][-1]
        axes[z, 0].fill_between(
            iterations,
            plot[algo][5][0],
            plot[algo][5][1],
            color=COLORS[algo],
            alpha=0.1,
        )
        axes[z, 0].plot(
            iterations,
            plot[algo][5][2],
            color=COLORS[algo],
            label=label,
            marker=MARKERS[algo],
            linestyle=LINESTYLE[algo],
        )

        axes[z, 1].fill_between(
            iterations,
            plot[algo][4][0],
            plot[algo][4][1],
            color=COLORS[algo],
            alpha=0.1,
        )
        axes[z, 1].plot(
            iterations,
            plot[algo][4][2],
            color=COLORS[algo],
            label=label,
            marker=MARKERS[algo],
            linestyle=LINESTYLE[algo],
        )

for algo in algos_all:
    for z, plot in enumerate(plots):
        if z == 3:
            label = LABELS[algo]
        else:
            label = None
        iterations = plot[algo][-1]
        axes[z, 2].fill_between(
            iterations,
            np.maximum(plot[algo][2][0], 0),
            np.maximum(plot[algo][2][1], 0),
            color=COLORS[algo],
            alpha=0.1,
        )
        axes[z, 2].plot(
            iterations,
            np.maximum(plot[algo][2][2], 0),
            color=COLORS[algo],
            label=label,
            marker=MARKERS[algo],
            linestyle=LINESTYLE[algo],
        )
        axes[z, 3].fill_between(
            iterations,
            plot[algo][1][0],
            plot[algo][1][1],
            color=COLORS[algo],
            alpha=0.1,
        )
        axes[z, 3].plot(
            iterations,
            plot[algo][1][2],
            color=COLORS[algo],
            label=label,
            marker=MARKERS[algo],
            linestyle=LINESTYLE[algo],
        )

axes[0, 0].set_title(r"$\mathcal{R}_T$")
axes[0, 1].set_title("$\mathcal{V}_T$")
axes[0, 2].set_title("$\mathcal{R}_T^{LT}$")
axes[0, 3].set_title("$\mathcal{V}_T^{LT}$")

for i in range(4):
    axes[3, i].set_xlabel("Horizon (T)")

for i in range(4):
    for j in range(4):
        axes[i, j].ticklabel_format(axis='both', style='sci', scilimits=(3,3))
        axes[i, j].spines['top'].set_visible(False)
        axes[i, j].spines['right'].set_visible(False)

axes[0, 0].set_ylim([-10, 200])
axes[0, 1].set_ylim([-10, 3000])
axes[0, 2].set_ylim([-10, 1000])

axes[0, 0].set_ylabel(r"$\rho_\lambda=0$")
axes[1, 0].set_ylabel(r"$\rho_\lambda=0.1$")
axes[2, 0].set_ylabel(r"$\rho_\lambda=0.5$")
axes[3, 0].set_ylabel(r"$\rho_\lambda=0.9$")

plt.subplots_adjust(wspace=0.1)
plt.legend(ncols=5, loc="upper right", bbox_to_anchor=(0.6, 5.1))
plt.savefig("../figures/figure3.pdf", bbox_inches="tight")
