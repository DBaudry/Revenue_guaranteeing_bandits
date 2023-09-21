# Plot all experiments

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from fairbandits.algo import greedy, ucb, lcb
from joblib import load
from glob import glob

COLORS = {
    "greedy": "blue",
    "KL-UCB": "green",
    "KL-LCB": "red",
}


LABELS = {
    "greedy": "greedy",
    "KL-UCB": "KL-UCB",
    "KL-LCB": "KL-LCB",
}

def load_plotting(seeds, mus, lambdas, K, T, algos):
    plot = {}
    for algo in algos:
        path = "./data/mab_%i_%s_%s_%s_%i_%i" % (
            seeds,
            "-".join(map(str, lambdas)),
            "-".join(map(str, mus)),
            algo,
            K,
            T,
        )
        print(path)
        res = load(path)[-1]
        cum_fairness = []
        cum_regret = []
        abs_cum_deviation = []
        iterations = np.array(res[0][0])
        for seed in range(seeds):
            cum_fairness.append(res[seed][1])
            cum_regret.append(res[seed][2])
            abs_cum_deviation.append(res[seed][3])
        cum_fairness = np.array(cum_fairness)
        cum_regret = np.array(cum_regret)
        abs_cum_deviation = np.array(abs_cum_deviation)

        high_fairness = np.quantile(cum_fairness, q=0.75, axis=0)
        low_fairness = np.quantile(cum_fairness, q=0.25, axis=0)
        mean_fairness = np.mean(cum_fairness, axis=0)

        high_regret = np.quantile(cum_regret, q=0.75, axis=0)
        low_regret = np.quantile(cum_regret, q=0.25, axis=0)
        mean_regret = np.mean(cum_regret, axis=0)

        high_deviation = np.quantile(abs_cum_deviation, q=0.75, axis=0)
        low_deviation = np.quantile(abs_cum_deviation, q=0.25, axis=0)
        mean_deviation = np.mean(abs_cum_deviation, axis=0)

        plot[algo] = (
            (high_fairness, low_fairness, mean_fairness),
            (high_regret, low_regret, mean_regret),
            (high_deviation, low_deviation, mean_deviation),
        )

    return plot, iterations 

def do_plot(algos, subopt_arms, fairness_path, bandit_path, lb_path):
    plt.figure()
    for algo in algos:
        plt.fill_between(
            iterations, plot[algo][0][0], plot[algo][0][1], color=COLORS[algo], alpha=0.1
        )
        plt.plot(iterations, plot[algo][0][2], color=COLORS[algo], label=LABELS[algo])
    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Fairness regret")
    plt.savefig(fairness_path + ".pdf", bbox_inches="tight")
    plt.savefig(fairness_path + ".png", bbox_inches="tight")
    plt.close()


    plt.figure()
    for algo in algos:
        plt.fill_between(
            iterations, plot[algo][1][0], plot[algo][1][1], color=COLORS[algo], alpha=0.1
        )
        plt.plot(iterations, plot[algo][1][2], color=COLORS[algo], label=LABELS[algo])
    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Bandit regret")
    plt.savefig(bandit_path + ".pdf", bbox_inches="tight")
    plt.savefig(bandit_path + ".png", bbox_inches="tight")
    plt.close()

    plt.figure()
    for algo in algos:
        for i in subopt_arms:
            plt.fill_between(
                iterations, plot[algo][2][0][:, i], plot[algo][2][1][:, i], color=COLORS[algo], alpha=0.1
            )
            if i == subopt_arms[0]:
                plt.plot(iterations, plot[algo][2][2][:, i], color=COLORS[algo], label=LABELS[algo])
            else:
                plt.plot(iterations, plot[algo][2][2][:, i], color=COLORS[algo])

    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Cumulative abs(pt - popt)")
    plt.savefig(lb_path + ".pdf", bbox_inches="tight")
    plt.savefig(lb_path + ".png", bbox_inches="tight")
    plt.close()


algos = ["greedy", "KL-UCB", "KL-LCB"]
# Experiments 1: study of the feasibility gap
exp = "exp1"
for feasibility_gap in [0, 0.1, 0.5, 0.9]:
    mus = np.array([0.8, 0.9, 0.7])
    K = len(mus)
    plot, iterations = load_plotting(
        seeds = 200, 
        mus = mus,
        lambdas = mus / len(mus) * (1 - feasibility_gap),
        K = K,
        T = 5000,
        algos=algos
        )
    do_plot(algos, [0, 2], 
            "./figures/%s_fairness_feasability_%f" % (exp, feasibility_gap), 
            "./figures/%s_bandit_feasability_%f" % (exp, feasibility_gap), 
            "./figures/%s_lb_feasability_%f" % (exp, feasibility_gap), 
        )



# Experiments 2: study of the feasibility gap
exp="exp2"
for feasibility_gap in [0, 0.1, 0.5, 0.9]:
    mus = np.array([0.8, 0.9, 0.7])
    lambdas = (1 - feasibility_gap) * np.ones(len(mus))  / np.sum(1 / mus)
    K = len(lambdas)
    plot, iterations = load_plotting(
        seeds = 200, 
        mus = mus,
        lambdas = lambdas,
        K = K,
        T = 5000,
        algos=algos
        )
    do_plot(algos, [0, 2], 
            "./figures/%s_fairness_feasability_%f" % (exp, feasibility_gap), 
            "./figures/%s_bandit_feasability_%f" % (exp, feasibility_gap), 
            "./figures/%s_lb_feasability_%f" % (exp, feasibility_gap), 
        )

# Experiments 3: small lambdas
exp="exp3"
T = 5000
mus = np.array([0.8, 0.9, 0.7])
feasibility_gap = 0.5
K = len(mus)
lambdas = 1 / np.sqrt(T) * np.ones(len(mus))
plot, iterations = load_plotting(
    seeds = 200, 
    mus = mus,
    lambdas = lambdas,
    K = K,
    T = 5000,
    algos=algos
    )
do_plot(algos, [0, 2], 
        "./figures/%s_fairness" % (exp), 
        "./figures/%s_bandit" % (exp), 
        "./figures/%s_lb" % (exp), 
    )


# Experiments 4: small mus
exp="exp4"
T = 5000
mus = np.array([0.8, 0.9, 0.7]) * 1 / np.sqrt(T)
feasibility_gap = 0.5
K = len(mus)
lambdas = mus / len(mus) * (1 - feasibility_gap)
plot, iterations = load_plotting(
    seeds = 200, 
    mus = mus,
    lambdas = lambdas,
    K = K,
    T = 5000,
    algos=algos
    )
do_plot(algos, [0, 2], 
        "./figures/%s_fairness" % (exp), 
        "./figures/%s_bandit" % (exp), 
        "./figures/%s_lb" % (exp), 
    )


# Experiments 5: Reproducing bandit Q setting
exp="exp5"
T = 5000
mus = np.array([0.335, 0.203, 0.241, 0.781, 0.617])
lambdas = np.array([0.167, 0.067, 0, 0, 0])
K = len(mus)
plot, iterations = load_plotting(
    seeds = 200, 
    mus = mus,
    lambdas = lambdas,
    K = K,
    T = 5000,
    algos=algos
    )
do_plot(algos, [0, 2], 
        "./figures/%s_fairness" % (exp), 
        "./figures/%s_bandit" % (exp), 
        "./figures/%s_lb" % (exp), 
    )


# Experiments 4: small mus
exp="exp6"
T = 5000
mus = np.array([0.8, 0.9, 0.7]) * 1 / np.sqrt(T)
feasibility_gap = 0.5
K = len(mus)
lambdas = mus / len(mus) * (1 - feasibility_gap)
mus[1] = mus[1] * np.sqrt(T)
plot, iterations = load_plotting(
    seeds = 200, 
    mus = mus,
    lambdas = lambdas,
    K = K,
    T = 5000,
    algos=algos
    )
do_plot(algos, [0, 2], 
        "./figures/%s_fairness" % (exp), 
        "./figures/%s_bandit" % (exp), 
        "./figures/%s_lb" % (exp), 
    )
