# Main paper experiments

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from fairbandits.algo import greedy, ucb, lcb
from joblib import load
from glob import glob
import os

cmap = plt.get_cmap("tab10")

COLORS = {
    "greedy": cmap(1),
    "KL-UCB": cmap(2),
    "KL-LCB": cmap(3),
    "KL-LCB-UCB": cmap(4),
    "greedy-UCB": cmap(5),
    "ETC": cmap(6),
    "BanditQ": cmap(7),
}


LABELS = {
    "greedy": "greedy",
    "KL-UCB": "KL-UCB (with normalization)",
    "KL-LCB": "KL-LCB (with normalization)",
    "KL-LCB-UCB": "KL-LCB (with UCB switch)",
    "greedy-UCB": "greedy (with UCB switch)",
    "ETC": "ETC",
    "BanditQ": "BanditQ",
}


MARKERS = {
    "greedy": None,
    "greedy-UCB": "+",
    "KL-LCB-UCB": "x",
    "KL-UCB": None,
    "KL-LCB": None,
    "ETC": "*",
    "BanditQ": "*",
}


def do_plot(algos, plot, subopt_arms, fairness_path, bandit_path, lb_path):

    plt.figure()
    for algo in algos:
        iterations = plot[algo][-1]
        plt.fill_between(
            iterations,
            plot[algo][5][0],
            plot[algo][5][1],
            color=COLORS[algo],
            alpha=0.1,
        )
        plt.plot(iterations, plot[algo][5][2], color=COLORS[algo], label=LABELS[algo], marker=MARKERS[algo])
    plt.legend()
    plt.xlabel("Horizon (T)")
    plt.ylabel("Bandit regret")
    plt.savefig(bandit_path + "pr.pdf", bbox_inches="tight")
    plt.savefig(bandit_path + "pr.png", bbox_inches="tight")
    plt.close()


    plt.figure()
    for algo in algos:
        iterations = plot[algo][-1]
        plt.fill_between(
            iterations,
            plot[algo][4][0],
            plot[algo][4][1],
            color=COLORS[algo],
            alpha=0.1,
        )
        plt.plot(iterations, plot[algo][4][2], color=COLORS[algo], label=LABELS[algo], marker=MARKERS[algo])
    plt.legend()
    plt.xlabel("Horizon (T)")
    plt.ylabel("Fairness regret")
    plt.savefig(fairness_path + "pr.pdf", bbox_inches="tight")
    plt.savefig(fairness_path + "pr.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    for algo in algos:
        iterations = plot[algo][-1]
        plt.fill_between(
            iterations,
            plot[algo][1][0],
            plot[algo][1][1],
            color=COLORS[algo],
            alpha=0.1,
        )
        plt.plot(iterations, plot[algo][1][2], color=COLORS[algo], label=LABELS[algo], marker=MARKERS[algo])
    plt.legend()
    plt.xlabel("Horizon (T)")
    plt.ylabel("Long term Fairness regret")
    plt.savefig(fairness_path + "sum.pdf", bbox_inches="tight")
    plt.savefig(fairness_path + "sum.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    for algo in algos:
        iterations = plot[algo][-1]
        plt.fill_between(
            iterations,
            plot[algo][0][0],
            plot[algo][0][1],
            color=COLORS[algo],
            alpha=0.1,
        )
        plt.plot(iterations, plot[algo][0][2], color=COLORS[algo], label=LABELS[algo], marker=MARKERS[algo])
    plt.legend()
    plt.xlabel("Horizon (T)")
    plt.ylabel("Long term Fairness regret")
    plt.savefig(fairness_path + ".pdf", bbox_inches="tight")
    plt.savefig(fairness_path + ".png", bbox_inches="tight")
    plt.close()

    plt.figure()
    for algo in algos:
        iterations = plot[algo][-1]
        plt.fill_between(
            iterations,
            plot[algo][2][0],
            plot[algo][2][1],
            color=COLORS[algo],
            alpha=0.1,
        )
        plt.plot(iterations, plot[algo][2][2], color=COLORS[algo], label=LABELS[algo], marker=MARKERS[algo])
    plt.legend()
    plt.xlabel("Horizon (T)")
    plt.ylabel("Long term Bandit regret")
    plt.savefig(bandit_path + ".pdf", bbox_inches="tight")
    plt.savefig(bandit_path + ".png", bbox_inches="tight")
    plt.close()

    plt.figure()
    for algo in algos:
        print(algo)
        iterations = plot[algo][-1]
        for i in subopt_arms:
            plt.fill_between(
                iterations,
                plot[algo][3][0][:, i],
                plot[algo][3][1][:, i],
                color=COLORS[algo],
                alpha=0.1,
            )
            if i == subopt_arms[0]:
                plt.plot(
                    iterations,
                    plot[algo][3][2][:, i],
                    color=COLORS[algo],
                    label=LABELS[algo], marker=MARKERS[algo],
                )
            else:
                plt.plot(iterations, plot[algo][3][2][:, i], color=COLORS[algo])

    plt.legend()
    plt.xlabel("Horizon (T)")
    plt.ylabel("Cumulative abs(pt - popt)")
    plt.savefig(lb_path + ".pdf", bbox_inches="tight")
    plt.savefig(lb_path + ".png", bbox_inches="tight")
    plt.close()


algos = ["greedy", "greedy-UCB", "KL-LCB-UCB", "KL-UCB", "KL-LCB", "ETC", "BanditQ"]
redraw = True
# Experiments 1: study of the feasibility gap
exp = "main_paper_exp1"
print(exp)
for feasibility_gap in [0, 0.1, 0.5, 0.9]:
    mus = np.array([0.8, 0.9, 0.7])
    K = len(mus)
    paths = [
        "./figures/%s_fairness_feasability_%f" % (exp, feasibility_gap),
        "./figures/%s_bandit_feasability_%f" % (exp, feasibility_gap),
        "./figures/%s_lb_feasability_%f" % (exp, feasibility_gap),
    ]
    cond = True
    for path in paths:
        plot = load("plotting/data/%s_%s" % (exp, str(feasibility_gap)))
        do_plot(
            algos,
            plot,
            [0, 2],
            "./figures/%s_fairness_feasability_%f" % (exp, feasibility_gap),
            "./figures/%s_bandit_feasability_%f" % (exp, feasibility_gap),
            "./figures/%s_lb_feasability_%f" % (exp, feasibility_gap),
        )



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
for path in paths:
    plot = load("plotting/data/%s" % exp)
    do_plot(
        algos,
        plot,
        [0, 2],
        "./figures/%s_fairness" % (exp),
        "./figures/%s_bandit" % (exp),
        "./figures/%s_lb" % (exp),
    )


"""Experiments 6
Very small mus except one
"""
exp = "main_paper_exp6"
print(exp)
T = 5000
mus = np.array([0.8, 0.9, 0.7]) * 1 / np.sqrt(T)
feasibility_gap = 0
K = len(mus)
lambdas = mus / len(mus) * (1 - feasibility_gap)
mus[1] = mus[1] * np.sqrt(T)

paths = [
    "./figures/%s_fairness" % (exp),
    "./figures/%s_bandit" % (exp),
    "./figures/%s_lb" % (exp),
]
cond = True
for path in paths:
    plot = load("plotting/data/%s" % exp)
    do_plot(
        algos,
        plot,
        [0, 2],
        "./figures/%s_fairness" % (exp),
        "./figures/%s_bandit" % (exp),
        "./figures/%s_lb" % (exp),
    )


# """Experiemnts 7
# A hard one feasibility gap is zero
# lambda=1/4, mu=1, lambda=3/4*mu et mu=2*T^{-1/3}
# """
# print("Experiment 7")
# T = 5000
# exp = "exp7"
# mus = np.array([1, 2 * T ** (-1 / 3)])
# K = len(mus)
# lambdas = np.array([0.25, 3 / 4 * mus[1]])
# paths = [
#     "./figures/%s_fairness" % (exp),
#     "./figures/%s_bandit" % (exp),
#     "./figures/%s_lb" % (exp),
# ]
# cond = True
# for path in paths:
#     cond = cond and os.path.exists(path + ".pdf") and os.path.exists(path + ".png") and (not redraw)
# if  not cond:
#     plot = load_plotting(seeds=200, mus=mus, lambdas=lambdas, K=K, T=5000, algos=algos)
#     do_plot(
#         algos,
#         plot,
#         [1],
#         "./figures/%s_fairness" % (exp),
#         "./figures/%s_bandit" % (exp),
#         "./figures/%s_lb" % (exp),
#     )
