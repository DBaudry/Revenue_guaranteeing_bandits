# MAB experiment

import numpy as np
from fairbandits.algo import FairBandit, Fair, Bandit, greedy, kl_ucb, kl_lcb, mab_opt, ETC, ETCAnytime, BanditQ, FairnessBaseline
from fairbandits.environment import mab_environment
from joblib import Parallel, delayed, dump
import os

def do_exp(seed, lambdas, mus, T, name):
    """
    Compute the cumulative fairness regret, cumulative regret and absolute cumulative deviation
    """
    p_opt = mab_opt(mus, lambdas)
    rng = np.random.RandomState(seed)

    # For non anytime algos let's just record 10 points
    if name in ["ETC", "BanditQ"]:
        iterations = []
        cum_regrets = []
        pr_regrets = []
        cum_fairness = []
        pr_fairness = []
        abs_cum_deviation = []
        for i in range(10):
            R = int(np.floor(T/10) * (i+1))
            if name == "ETC":
                mab_algo = ETC(lambdas, R, lambdas/4, 1/6, 1/K*np.ones(K))
            elif name == "BanditQ":
                mab_algo = BanditQ(lambdas, R)
            cum_regret_t = 0
            cum_fairness_t = np.zeros(len(lambdas))
            cum_deviation_t = 0
            pr_regret_t = 0
            pr_fairness_t = 0
            for r in range(R):
                p_t = mab_algo.play()
                k_t, r_t = mab_environment(p_t, mus, rng)
                mab_algo.update(k_t, r_t, p_t)
                cum_regret_t += np.sum((p_opt - p_t) * mus)
                pr_regret_t += np.maximum(np.sum((p_opt - p_t) * mus), 0)
                cum_fairness_t +=lambdas - p_t * mus
                cum_deviation_t += np.abs(p_t - p_opt)
                pr_fairness_t += np.maximum(lambdas - p_t * mus, 0)
            cum_regrets.append(np.copy(cum_regret_t))
            pr_regrets.append(np.copy(pr_regret_t))
            cum_fairness.append(np.copy(cum_fairness_t))
            pr_fairness.append(np.copy(pr_fairness_t))
            abs_cum_deviation.append(np.copy(cum_deviation_t))
            iterations.append(r)
        return iterations, cum_fairness, cum_regrets, abs_cum_deviation, pr_fairness, pr_regrets

    # Otherwise, we record 200 points
    if name == "greedy-UCB":
        mab_algo = FairBandit(Fair(lambdas, greedy, normalize=False), Bandit(lambdas, kl_ucb))
    elif name == "KL-LCB-UCB":
        mab_algo = FairBandit(Fair(lambdas, kl_lcb, normalize=False), Bandit(lambdas, kl_ucb))
    elif name == "greedy":
        mab_algo = FairBandit(Fair(lambdas, greedy), Bandit(lambdas, kl_ucb))
    elif name == "KL-UCB":
        mab_algo = FairBandit(Fair(lambdas, kl_ucb), Bandit(lambdas, kl_ucb))
    elif name == "KL-LCB":
        mab_algo = FairBandit(Fair(lambdas, kl_lcb), Bandit(lambdas, kl_ucb))
    elif name == "Baseline":
        mab_algo = FairnessBaseline(lambdas)
    else:
        raise ValueError("%s does not exists" % name)

    cum_regrets = []
    pr_regrets = []
    pr_fairness = []
    pr_fairness_t = 0
    cum_fairness = []
    abs_cum_deviation = []
    cum_regret_t = 0
    pr_regret_t = 0
    cum_fairness_t = np.zeros(len(lambdas))
    cum_deviation_t = 0
    iterations = []
    for t in range(T):
        p_t = mab_algo.play()
        k_t, r_t = mab_environment(p_t, mus, rng)
        mab_algo.update(k_t, r_t, p_t)
        cum_regret_t += np.sum((p_opt - p_t) * mus)
        cum_fairness_t +=lambdas - p_t * mus
        cum_deviation_t += np.abs(p_t - p_opt)
        pr_regret_t += np.maximum( np.sum((p_opt - p_t) * mus), 0)
        pr_fairness_t += np.maximum(lambdas - p_t * mus, 0)

        if t % (T // 200) == 0:
            cum_regrets.append(np.copy(cum_regret_t))
            pr_regrets.append(np.copy(pr_regret_t))
            cum_fairness.append(np.copy(cum_fairness_t))
            abs_cum_deviation.append(np.copy(cum_deviation_t))
            iterations.append(t)
            pr_fairness.append(np.copy(pr_fairness_t))

    return iterations, cum_fairness, cum_regrets, abs_cum_deviation, pr_fairness, pr_regrets

seeds = 200
T = 100000
algo_names = ["greedy","KL-UCB", "KL-LCB", "KL-LCB-UCB", "greedy-UCB", "ETC", "BanditQ", "Baseline"]

"""Experiments 1
Different feasibility gaps lambdas proportional to mus
"""
print("Experiment 1")
for feasibility_gap in [0, 0.1, 0.5, 0.9]:
    for algo_name in algo_names:
        print(algo_name)
        mus = np.array([0.8, 0.9, 0.7])
        lambdas = mus / len(mus) * (1 - feasibility_gap)
        K = len(lambdas)
        path = "data/mab_%i_%s_%s_%s_%i_%i_exp1" % (
            seeds,
            "-".join(map(str, lambdas)),
            "-".join(map(str, mus)),
            algo_name,
            K,
            T,
        )
        if not os.path.exists(path):
            res = Parallel(n_jobs=-1, verbose=True)(
                    delayed(do_exp)(seed, lambdas, mus, T, algo_name)
                    for seed in range(seeds)
                )
            dump(
                [seeds, lambdas, mus, algo_name, K, T, res],
                path,
            )
            del res

# # """Experiments 2
# # Different feasibility gaps lambdas constant
# # """
# # print("Experiment 2")
# # for feasibility_gap in [0, 0.1, 0.5, 0.9]:
# #     for algo_name in ["greedy","KL-UCB", "KL-LCB", "ETC", "BanditQ"]:
# #         mus = np.array([0.8, 0.9, 0.7])
# #         lambdas = (1 - feasibility_gap) * np.ones(len(mus))  / np.sum(1 / mus)
# #         K = len(lambdas)
# #         path = "data/mab_%i_%s_%s_%s_%i_%i_exp2" % (
# #             seeds,
# #             "-".join(map(str, lambdas)),
# #             "-".join(map(str, mus)),
# #             algo_name,
# #             K,
# #             T,
# #         )
# #         if not os.path.exists(path):
# #             res = Parallel(n_jobs=-1, verbose=True)(
# #                     delayed(do_exp)(seed, lambdas, mus, T, algo_name)
# #                     for seed in range(seeds)
# #                 )
            
# #             dump(
# #                 [seeds, lambdas, mus, algo_name, K, T, res],
# #                 path,
# #             )
# #             del res


# """Experiments 3
# Very small lambdas
# """
# print("Experiment 3")
# for algo_name in algo_names:
#     mus = np.array([0.8, 0.9, 0.7])
#     feasibility_gap = 0.5
#     K = len(mus)
#     lambdas = 1 / np.sqrt(T) * np.ones(len(mus))
#     path = "data/mab_%i_%s_%s_%s_%i_%i_exp3" % (
#         seeds,
#         "-".join(map(str, lambdas)),
#         "-".join(map(str, mus)),
#         algo_name,
#         K,
#         T,
#     )
#     if not os.path.exists(path):
#         res = Parallel(n_jobs=-1, verbose=True)(
#                 delayed(do_exp)(seed, lambdas, mus, T, algo_name)
#                 for seed in range(seeds)
#             )
        
#         dump(
#             [seeds, lambdas, mus, algo_name, K, T, res],
#             path,
#         )
#         del res

  
# """Experiments 4
# Very small mus
# """
# print("Experiment 4")
# for algo_name in algo_names:
#     mus = np.array([0.8, 0.9, 0.7]) * 1 / np.sqrt(T)
#     feasibility_gap = 0.5
#     K = len(mus)
#     lambdas = mus / len(mus) * (1 - feasibility_gap)
#     path = "data/mab_%i_%s_%s_%s_%i_%i_exp4" % (
#         seeds,
#         "-".join(map(str, lambdas)),
#         "-".join(map(str, mus)),
#         algo_name,
#         K,
#         T,
#     )
#     if not os.path.exists(path):
#         res = Parallel(n_jobs=-1, verbose=True)(
#                 delayed(do_exp)(seed, lambdas, mus, T, algo_name)
#                 for seed in range(seeds)
#             )
        
#         dump(
#             [seeds, lambdas, mus, algo_name, K, T, res],
#             path,
#         )
#         del res

"""Experiments 5
Reproducing the experimental setting of BanditQ
"""
print("Experiment 5")
for algo_name in algo_names:
    print(algo_name)
    mus = np.array([0.335, 0.203, 0.241, 0.781, 0.617])
    lambdas = np.array([0.167, 0.067, 0, 0, 0])
    K = len(mus)
    path = "data/mab_%i_%s_%s_%s_%i_%i_exp5" % (
        seeds,
        "-".join(map(str, lambdas)),
        "-".join(map(str, mus)),
        algo_name,
        K,
        T,
    )
    if not os.path.exists(path):
        res = Parallel(n_jobs=-1, verbose=True)(
                delayed(do_exp)(seed, lambdas, mus, T, algo_name)
                for seed in range(seeds)
            )
        
        dump(
            [seeds, lambdas, mus, algo_name, K, T, res],
            path,
        )
        del res



"""Experiments 6
Very small mus except one
"""
T = 5000
print("Experiment 6")
for algo_name in algo_names:
    print(algo_name)
    mus = np.array([0.8, 0.9, 0.7]) * 1 / np.sqrt(T)
    feasibility_gap = 0
    K = len(mus)
    lambdas = mus / len(mus) * (1 - feasibility_gap)
    mus[1]  = mus[1] * np.sqrt(T)
    path = "data/mab_%i_%s_%s_%s_%i_%i_exp6" % (
        seeds,
        "-".join(map(str, lambdas)),
        "-".join(map(str, mus)),
        algo_name,
        K,
        T,
    )
    if not os.path.exists(path):
        res = Parallel(n_jobs=-1, verbose=True)(
                delayed(do_exp)(seed, lambdas, mus, T, algo_name)
                for seed in range(seeds)
            )
        
        dump(
            [seeds, lambdas, mus, algo_name, K, T, res],
            path,
        )
        del res

# """Experiemnts 7
# A hard one feasibility gap is zero
# lambda=1/4, mu=1, lambda=4/3*mu et mu=T^{1/3} + log(T) légèrement au-dessus du threshold où on regarde pas (T^{1/3} théoriquement)
# """
# print("Experiment 7")
# for algo_name in algo_names:
#     print(algo_name)
#     mus = np.array([1, 2 * T**(-1/3)])
#     K = len(mus)
#     lambdas = np.array([0.25, 3/4 * mus[1]])
#     path = "data/mab_%i_%s_%s_%s_%i_%i_exp7" % (
#         seeds,
#         "-".join(map(str, lambdas)),
#         "-".join(map(str, mus)),
#         algo_name,
#         K,
#         T,
#     )
#     if not os.path.exists(path):
#         res = Parallel(n_jobs=-1, verbose=True)(
#                 delayed(do_exp)(seed, lambdas, mus, T, algo_name)
#                 for seed in range(seeds)
#             )
        
#         dump(
#             [seeds, lambdas, mus, algo_name, K, T, res],
#             path,
#         )
#         del res

