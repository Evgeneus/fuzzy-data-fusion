from collections import defaultdict
import numpy as np
from mv import majority_voting
from em import expectation_maximization
from mcmc import mcmc
from f_mcmc import f_mcmc
from generator import synthesize
import pandas as pd
from sums import sums
from average_log import average_log
from investment import investment
from pooled_investment import pooled_investment
from util import prob_binary_convert, accu_G

n_runs = 10

work_dir = '../../data/'

def adapter_input(Psi):
    Psi_new = {}
    for obj_ind, obj_data in enumerate(Psi):
        obj_s, obj_v = [], []
        for s, v in obj_data:
            obj_s.append(s)
            obj_v.append(v)
        obj_data_new = {obj_ind: [obj_s, obj_v]}
        Psi_new.update(obj_data_new)
    return Psi_new


def adapter_output(belief, data):
    val_p = []
    for obj_ind in sorted(belief.keys()):
        possible_values = sorted(list(set(data[obj_ind][1])))
        obj_p = map(lambda x: 0.0 if x != 1. else x, belief[obj_ind])
        val_p.append(defaultdict(int, zip(possible_values, obj_p)))
    return val_p


def accuracy():
    """
    Vary the confusion probability on synthetic data.
    """
    # number of sources
    N = 30
    # number of objects
    M = 5000
    # number of values per object
    V = 50
    # synthetically generated observations
    density = 0.5
    # TO DO

    mcmc_params = {'N_iter': 10, 'burnin': 1, 'thin': 2, 'FV': 0}
    conf_probs = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    res = {'sums': [], 'mv': [], 'em': [], 'mcmc': [],
           'sums_f': [], 'mv_f': [], 'em_f': [], 'mcmc_f': [],
           'conf_probs': conf_probs, 'avlog': [], 'avlog_f': [],
           'inv': [], 'inv_f': [], 'pinv': [], 'pinv_f': []}
    for conf_prob in conf_probs:
        GT, GT_G, Cl, Psi = synthesize(N, M, V, density, 1-conf_prob, None)

        mv_accu, em_accu, mcmc_accu, sums_accu, avlog_accu, inv_accu, \
        pinv_accu = [], [], [], [], [], [], []
        mv_accu_f, em_accu_f, mcmc_accu_f, sums_accu_f, avlog_accu_f, \
        inv_accu_f, pinv_accu_f = [], [], [], [], [], [], []
        for run in range(n_runs):
            Psi_fussy = f_mcmc(N, M, Psi, Cl, mcmc_params)[1]

            # MV
            mv_p = majority_voting(Psi)
            mv_b = prob_binary_convert(mv_p)
            mv_pf = majority_voting(Psi_fussy)
            mv_bf = prob_binary_convert(mv_pf)

            # EM
            em_A, em_p = expectation_maximization(N, M, Psi)
            em_b = prob_binary_convert(em_p)
            em_Af, em_pf = expectation_maximization(N, M, Psi_fussy)
            em_bf = prob_binary_convert(em_pf)

            # MCMC
            mcmc_A, mcmc_p = mcmc(N, M, Psi, mcmc_params)
            mcmc_b = prob_binary_convert(mcmc_p)
            mcmc_Af, mcmc_pf = mcmc(N, M, Psi_fussy, mcmc_params)
            mcmc_bf = prob_binary_convert(mcmc_pf)


            data = adapter_input(Psi)
            data_f = adapter_input(Psi_fussy)
            # SUMS
            sums_belief = sums(N, data)
            sums_b = adapter_output(sums_belief, data)

            sums_belief_f = sums(N, data_f)
            sums_bf = adapter_output(sums_belief_f, data_f)

            # AVG LOG
            avlog_belief = average_log(N, data)
            avlog_b = adapter_output(avlog_belief, data)

            avlog_belief_f = average_log(N, data_f)
            avlog_bf = adapter_output(avlog_belief_f, data_f)

            # INVESTMENT
            inv_belief = investment(N, data)
            inv_b = adapter_output(inv_belief, data)

            inv_belief_f = investment(N, data_f)
            inv_bf = adapter_output(inv_belief_f, data_f)

            # POOLED INVESTMENT
            pinv_belief = pooled_investment(N, data)
            pinv_b = adapter_output(pinv_belief, data)

            pinv_belief_f = pooled_investment(N, data_f)
            pinv_bf = adapter_output(pinv_belief_f, data_f)

            # exclude objects on which no conflicts
            obj_with_conflicts = []
            for obj_id, obj in enumerate(mv_b):
                if len(obj) > 1:
                    obj_with_conflicts.append(obj_id)

            mv_accu.append(np.average([mv_b[obj][GT[obj]] for obj in obj_with_conflicts]))
            mv_accu_f.append(np.average([mv_bf[obj][GT[obj]] for obj in obj_with_conflicts]))

            em_accu.append(np.average([em_b[obj][GT[obj]] for obj in obj_with_conflicts]))
            em_accu_f.append(np.average([em_bf[obj][GT[obj]] for obj in obj_with_conflicts]))

            mcmc_accu.append(np.average([mcmc_b[obj][GT[obj]] for obj in obj_with_conflicts]))
            mcmc_accu_f.append(np.average([mcmc_bf[obj][GT[obj]] for obj in obj_with_conflicts]))

            sums_accu.append(np.average([sums_b[obj][GT[obj]] for obj in obj_with_conflicts]))
            sums_accu_f.append(np.average([sums_bf[obj][GT[obj]] for obj in obj_with_conflicts]))

            avlog_accu.append(np.average([avlog_b[obj][GT[obj]] for obj in obj_with_conflicts]))
            avlog_accu_f.append(np.average([avlog_bf[obj][GT[obj]] for obj in obj_with_conflicts]))

            inv_accu.append(np.average([inv_b[obj][GT[obj]] for obj in obj_with_conflicts]))
            inv_accu_f.append(np.average([inv_bf[obj][GT[obj]] for obj in obj_with_conflicts]))

            pinv_accu.append(np.average([pinv_b[obj][GT[obj]] for obj in obj_with_conflicts]))
            pinv_accu_f.append(np.average([pinv_bf[obj][GT[obj]] for obj in obj_with_conflicts]))

        res['mv'].append(np.average(mv_accu))
        res['mv_f'].append(np.average(mv_accu_f))

        res['em'].append(np.average(em_accu))
        res['em_f'].append(np.average(em_accu_f))

        res['mcmc'].append(np.average(mcmc_accu))
        res['mcmc_f'].append(np.average(mcmc_accu_f))

        res['sums'].append(np.average(sums_accu))
        res['sums_f'].append(np.average(sums_accu_f))

        res['avlog'].append(np.average(avlog_accu))
        res['avlog_f'].append(np.average(avlog_accu_f))

        res['inv'].append(np.average(inv_accu))
        res['inv_f'].append(np.average(inv_accu_f))

        res['pinv'].append(np.average(pinv_accu))
        res['pinv_f'].append(np.average(pinv_accu_f))

        print('confusion probability: {}, mv: {:1.4f}, em: {:1.4f}, mcmc: {:1.4f}, '
              'sums: {:1.4f}, avlog: {:1.4f}, inv: {:1.4f}, pinv: {:1.4f}'
              .format(conf_prob,
               np.average(mv_accu),
               np.average(em_accu),
               np.average(mcmc_accu),
               np.average(sums_accu),
               np.average(avlog_accu),
               np.average(inv_accu),
               np.average(pinv_accu)
            ))
        print('confusion probability: {}, mv_f: {:1.4f}, em:_f {:1.4f}, mcmc_f: {:1.4f}, '
              'sums_f: {:1.4f}, avlog_f: {:1.4f}, inv_f: {:1.4f}, pinv_f: {:1.4f}'
              .format(conf_prob,
               np.average(mv_accu_f),
               np.average(em_accu_f),
               np.average(mcmc_accu_f),
               np.average(sums_accu_f),
               np.average(avlog_accu_f),
               np.average(inv_accu_f),
               np.average(pinv_accu_f)
            ))

    pd.DataFrame(res).to_csv('synthetic_accuracy_binary', index=False)


def convergence():
    """
    Convergence of MCMC.
    """
    # number of sources
    N = 30
    # number of objects
    M = 5000
    # number of values per object
    V = 30
    # synthetically generated observations
    density = 0.5
    accuracy = 0.9
    conf_prob = 0.2

    GT, GT_G, Cl, Psi = synthesize(N, M, V, density, 1-conf_prob, accuracy)
    res = {'G accuracy': [], 'error': [], 'number of iterations': [3, 5, 10, 30, 50, 100]}
    for p in [(3, 0, 1), (5, 0, 1), (10, 1, 2), (30, 5, 3), (50, 7, 5), (100, 10, 7)]:
        params = {'N_iter': p[0], 'burnin': p[1], 'thin': p[2], 'FV': 0}
        runs = []
        for run in range(n_runs):
            f_mcmc_G = f_mcmc(N, M, Psi, Cl, params)[0]
            G_accu = np.average(accu_G(f_mcmc_G, GT_G))
            runs.append(G_accu)
        res['G accuracy'].append(np.average(runs))
        res['error'].append(np.std(runs))

        print('p: {}, G accu: {}, std: {}'.format(p, np.average(runs), np.std(runs)))

    pd.DataFrame(res).to_csv('synthetic_convergence.csv', index=False)


def values():
    """
    Vary the number of distinct values V.
    """
    # number of sources
    N = 30
    # number of objects
    M = 5000
    # synthetically generated observations
    density = 0.5
    accuracy = 0.9
    conf_prob = 0.2
    Vs = [2, 4, 8, 16, 32, 64, 128]
    params = {'N_iter': 10, 'burnin': 1, 'thin': 2, 'FV': 0}
    res = {'G accuracy': [], 'error': [], 'number of distinct values per object': Vs}
    for V in Vs:
        GT, GT_G, Cl, Psi = synthesize(N, M, V, density, 1 - conf_prob, accuracy)
        G_accu = []
        for run in range(n_runs):
            f_mcmc_G = f_mcmc(N, M, Psi, Cl, params)[0]
            G_accu = np.average(accu_G(f_mcmc_G, GT_G))
        res['G accuracy'].append(np.average(G_accu))
        res['error'].append(np.std(G_accu))
        print('V: {}, accu: {:1.4f}'.format(V, np.average(G_accu)))

    pd.DataFrame(res).to_csv('synthetic_values.csv', index=False)


def get_acc_g():
    """
       Vary the confusion probability on synthetic data.
       """
    # number of sources
    N = 30
    # number of objects
    M = 5000
    # number of values per object
    V = 50
    # synthetically generated observations
    density = 0.5
    # TO DO

    mcmc_params = {'N_iter': 10, 'burnin': 1, 'thin': 2, 'FV': 0}
    conf_probs = [0.2, 0.3, 0.4]
    s_acc_list = [0.6, 0.7, 0.8, 0.9, 1.]
    res = {'conf_probs': [], 'acc_g': [], 'acc_g_std': [], 's_acc': []}
    for conf_prob in conf_probs:
        for s_acc in s_acc_list:
            GT, GT_G, Cl, Psi = synthesize(N, M, V, density, 1-conf_prob, s_acc)
            accu_G_list = []
            for run in range(n_runs):
                f_mcmc_G, Psi_fussy = f_mcmc(N, M, Psi, Cl, mcmc_params)
                accu_G_list.append(accu_G(f_mcmc_G, GT_G))

            res['acc_g'].append(np.mean(accu_G_list))
            res['acc_g_std'].append(np.std(accu_G_list))
            res['s_acc'].append(s_acc)
            res['conf_probs'].append(conf_prob)

            print 's_acc: {}'.format(s_acc)
            print 'conf_prob: {}'.format(conf_prob)
            print 'acc G: {}'.format(np.mean(accu_G_list))
            print '---------------'
    pd.DataFrame(res).to_csv(work_dir + 'accuracy_g.scv', index=False)


if __name__ == '__main__':
    # accuracy()
    # convergence()
    values()
    # get_acc_g()

