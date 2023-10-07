# Main paper experiments

import numpy as np
from fairbandits.algo import greedy, ucb, lcb
from joblib import load, dump
from glob import glob
import os

def load_plotting(seeds, mus, lambdas, K, T, algos):
    plot = {}
    for algo in algos:
        print(
            "./data/mab_%i_%s_%s_%s_%i_%i_*"
            % (
                seeds,
                "-".join(map(str, lambdas)),
                "-".join(map(str, mus)),
                algo,
                K,
                T,
            )
        )
        path = glob(
            "./data/mab_%i_%s_%s_%s_%i_%i_*"
            % (
                seeds,
                "-".join(map(str, lambdas)),
                "-".join(map(str, mus)),
                algo,
                K,
                T,
            )
        )
        if len(path) != 1:
            raise ValueError("More than more paths: " + str(path))
        path = path[0]
        print(path)
        res = load(path)[-1]
        cum_fairness = []
        cum_regret = []
        pr_cum_regret = []
        pr_cum_fairness = []
        abs_cum_deviation = []
        iterations = np.array(res[0][0])
        for seed in range(seeds):
            cum_fairness.append(res[seed][1])
            cum_regret.append(res[seed][2])
            abs_cum_deviation.append(res[seed][3])
            pr_cum_fairness.append(res[seed][4])
            pr_cum_regret.append(res[seed][5])

        cum_fairness = np.array(cum_fairness)
        cum_regret = np.array(cum_regret)
        pr_cum_fairness = np.array(pr_cum_fairness)
        pr_cum_regret = np.array(pr_cum_regret)
        abs_cum_deviation = np.array(abs_cum_deviation)

        high_fairness = np.quantile(cum_fairness, q=0.75, axis=0)
        low_fairness = np.quantile(cum_fairness, q=0.25, axis=0)
        mean_fairness = np.mean(cum_fairness, axis=0)

        high_fairness_max = np.max(high_fairness, axis=1)
        low_fairness_max = np.max(low_fairness, axis=1)
        mean_fairness_max = np.max(mean_fairness, axis=1)

        high_fairness_sum = np.sum(np.maximum(high_fairness, 0), axis=1)
        low_fairness_sum = np.sum(np.maximum(low_fairness, 0), axis=1)
        mean_fairness_sum = np.sum(np.maximum(mean_fairness, 0), axis=1)

        high_regret = np.quantile(cum_regret, q=0.75, axis=0)
        low_regret = np.quantile(cum_regret, q=0.25, axis=0)
        mean_regret = np.mean(cum_regret, axis=0)


        pr_high_regret = np.quantile(pr_cum_regret, q=0.75, axis=0)
        pr_low_regret = np.quantile(pr_cum_regret, q=0.25, axis=0)
        pr_mean_regret = np.mean(pr_cum_regret, axis=0)

        high_deviation = np.quantile(abs_cum_deviation, q=0.75, axis=0)
        low_deviation = np.quantile(abs_cum_deviation, q=0.25, axis=0)
        mean_deviation = np.mean(abs_cum_deviation, axis=0)

        pr_high_fairness = np.quantile(pr_cum_fairness, q=0.75, axis=0)
        pr_low_fairness = np.quantile(pr_cum_fairness, q=0.25, axis=0)
        pr_mean_fairness = np.mean(pr_cum_fairness, axis=0)

        pr_high_fairness_sum = np.sum(np.maximum(pr_high_fairness, 0), axis=1)
        pr_low_fairness_sum = np.sum(np.maximum(pr_low_fairness, 0), axis=1)
        pr_mean_fairness_sum = np.sum(np.maximum(pr_mean_fairness, 0), axis=1)

        plot[algo] = (
            (high_fairness_max, low_fairness_max, mean_fairness_max),
            (high_fairness_sum, low_fairness_sum, mean_fairness_sum),
            (high_regret, low_regret, mean_regret),
            (high_deviation, low_deviation, mean_deviation),
            (pr_high_fairness_sum, pr_low_fairness_sum, pr_mean_fairness_sum),
            (pr_high_regret, pr_low_regret, pr_mean_regret),
            iterations,
        )
    return plot


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
        cond = cond and os.path.exists(path + ".pdf") and os.path.exists(path + ".png") and (not redraw)
    if  cond:
        continue
    plot = load_plotting(
        seeds=200,
        mus=mus,
        lambdas=mus / len(mus) * (1 - feasibility_gap),
        K=K,
        T=2000,
        algos=algos,
    )
    dump(plot, "plotting/data/%s_%s" % (exp, str(feasibility_gap)))

    # do_plot(
    #     algos,
    #     plot,
    #     [0, 2],
    #     "./figures/%s_fairness_feasability_%f" % (exp, feasibility_gap),
    #     "./figures/%s_bandit_feasability_%f" % (exp, feasibility_gap),
    #     "./figures/%s_lb_feasability_%f" % (exp, feasibility_gap),
    # )



# Experiments 5: Reproducing bandit Q setting
exp = "main_paper_exp5"
print(exp)
T = 2000
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
    cond = cond and os.path.exists(path + ".pdf") and os.path.exists(path + ".png") and (not redraw)
if  not cond:
    plot = load_plotting(seeds=200, mus=mus, lambdas=lambdas, K=K, T=T, algos=algos)
    dump(plot, "plotting/data/%s" % exp)


"""Experiments 6
Very small mus except one
"""
exp = "main_paper_exp6"
print(exp)
T = 1000
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
    cond = cond and os.path.exists(path + ".pdf") and os.path.exists(path + ".png") and (not redraw)
if  not cond:
    plot = load_plotting(seeds=200, mus=mus, lambdas=lambdas, K=K, T=T, algos=algos)
    dump(plot, "plotting/data/%s" % exp)


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
