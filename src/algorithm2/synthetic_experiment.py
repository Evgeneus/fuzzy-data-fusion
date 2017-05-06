from collections import defaultdict
import numpy as np
from mv import majority_voting
from em import expectation_maximization
from mcmc import mcmc
from f_mcmc import f_mcmc
from generator import synthesize
import pandas as pd

work_dir = '/home/bykau/Dropbox/Fuzzy/'
n_runs = 50


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


def accuracy():
    """
    Vary the confusion probability on synthetic data.
    """
    # number of sources
    N = 30
    # number of objects
    M = 500
    # number of values per object
    V = 50
    # synthetically generated observations
    density = 0.4
    # TO DO
    accuracy = 0.8

    mcmc_params = {'N_iter': 10, 'burnin': 1, 'thin': 2, 'FV': 0}
    conf_probs = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    res = {'mv': [], 'em': [], 'mcmc': [],
           'em_std': [], 'mcmc_std': [],
           'mv_f': [], 'em_f': [], 'mcmc_f': [],
           'em_std_f': [], 'mcmc_std_f': [], 'conf_probs': conf_probs}
    for conf_prob in conf_probs:
        GT, GT_G, Cl, Psi = synthesize(N, M, V, density, accuracy, 1-conf_prob)

        mv_accu, em_accu, mcmc_accu, sums_accu = [], [], [], []
        mv_accu_f, em_accu_f, mcmc_accu_f, sums_accu_f = [], [], [], []
        for run in range(n_runs):
            Psi_fussy = f_mcmc(N, M, Psi, Cl, mcmc_params)

            mv_p = majority_voting(Psi)
            mv_pf = majority_voting(Psi_fussy)

            em_A, em_p = expectation_maximization(N, M, Psi)
            em_Af, em_pf = expectation_maximization(N, M, Psi_fussy)

            mcmc_A, mcmc_p = mcmc(N, M, Psi, mcmc_params)
            mcmc_Af, mcmc_pf = mcmc(N, M, Psi_fussy, mcmc_params)

            # exclude objects on which no conflicts
            obj_with_conflicts = []
            for obj_id, obj in enumerate(mv_p):
                if len(obj) > 1:
                    obj_with_conflicts.append(obj_id)

            mv_accu.append(np.average([mv_p[obj][GT[obj]] for obj in obj_with_conflicts]))
            mv_accu_f.append(np.average([mv_pf[obj][GT[obj]] for obj in obj_with_conflicts]))

            em_accu.append(np.average([em_p[obj][GT[obj]] for obj in obj_with_conflicts]))
            em_accu_f.append(np.average([em_pf[obj][GT[obj]] for obj in obj_with_conflicts]))

            mcmc_accu.append(np.average([mcmc_p[obj][GT[obj]] for obj in obj_with_conflicts]))
            mcmc_accu_f.append(np.average([mcmc_pf[obj][GT[obj]] for obj in obj_with_conflicts]))

        res['mv'].append(np.average(mv_accu))
        res['mv_f'].append(np.average(mv_accu_f))

        res['em'].append(np.average(em_accu))
        res['em_std'].append(np.std(em_accu))
        res['em_f'].append(np.average(em_accu_f))
        res['em_std_f'].append(np.std(em_accu_f))


        res['mcmc'].append(np.average(mcmc_accu))
        res['mcmc_std'].append(np.std(mcmc_accu))
        res['mcmc_f'].append(np.average(mcmc_accu_f))
        res['mcmc_std_f'].append(np.std(mcmc_accu_f))

        print('confusion probability: {}, mv: {:1.4f}, em: {:1.4f}, mcmc: {:1.4f}'.format(conf_prob,
                                                                                       np.average(mv_accu),
                                                                                       np.average(em_accu),
                                                                                       np.average(mcmc_accu)))
        print('confusion probability: {}, mv_f: {:1.4f}, em:_f {:1.4f}, mcmc_f: {:1.4f}'.format(conf_prob,
                                                                                             np.average(mv_accu_f),
                                                                                             np.average(em_accu_f),
                                                                                             np.average(mcmc_accu_f)))
    pd.DataFrame(res).to_csv('synthetic_accuracy_prob.csv', index=False)


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
    density = 0.3
    accuracy = 0.8
    conf_prob = 0.8

    GT, GT_G, Cl, Psi = synthesize(N, M, V, density, accuracy, conf_prob)
    res = {'accuracy': [], 'error': [], 'number of iterations': [5, 10, 30, 50, 100]}
    for p in [(3, 0, 1), (5, 0, 1), (10, 1, 2), (30, 5, 3), (50, 7, 5), (100, 10, 7)]:
        params = {'N_iter': p[0], 'burnin': p[1], 'thin': p[2], 'FV': 0}
        runs = []
        for run in range(n_runs):
            f_mcmc_A, f_mcmc_p, f_mcmc_G = f_mcmc(N, M, Psi, Cl, params)
            f_mcmc_accu = np.average([f_mcmc_p[obj][GT[obj]] for obj in GT.keys()])
            runs.append(f_mcmc_accu)
        res['accuracy'].append(np.average(runs))
        res['error'].append(np.std(runs))

        print('p: {}, accu: {}, std: {}'.format(p, np.average(runs), np.std(runs)))

    pd.DataFrame(res).to_csv(work_dir + 'synthetic_convergence.csv', index=False)


def values():
    """
    Vary the number of distinct values V.
    """
    # number of sources
    N = 30
    # number of objects
    M = 5000
    # synthetically generated observations
    density = 0.3
    accuracy = 0.8
    conf_prob = 0.8
    Vs = [2, 4, 8, 16, 32, 64, 128]
    params = {'N_iter': 10, 'burnin': 1, 'thin': 2, 'FV': 0}
    res = {'accuracy': [], 'std': [], 'number of distinct values per object': Vs}
    for V in Vs:
        GT, GT_G, Cl, Psi = synthesize(N, M, V, density, accuracy, conf_prob)
        f_mcmc_accu = []
        for run in range(n_runs):
            f_mcmc_A, f_mcmc_p, f_mcmc_G = f_mcmc(N, M, Psi, Cl, params)
            f_mcmc_accu.append(np.average([f_mcmc_p[obj][GT[obj]] for obj in GT.keys()]))
        res['accuracy'].append(np.average(f_mcmc_accu))
        res['std'].append(np.std(f_mcmc_accu))
        print('V: {}, accu: {:1.4f}'.format(V, np.average(f_mcmc_accu)))

    pd.DataFrame(res).to_csv(work_dir + 'synthetic_values.csv', index=False)


if __name__ == '__main__':
    accuracy()
    #convergence()
    #values()
