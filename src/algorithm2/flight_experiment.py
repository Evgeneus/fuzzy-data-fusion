import time
from copy import deepcopy
import numpy as np
import pandas as pd
from util import prob_binary_convert, accu_G
from mv import majority_voting
from em import expectation_maximization
from mcmc import mcmc
from f_mcmc import f_mcmc
from sums import sums
from average_log import average_log
from investment import investment
from pooled_investment import pooled_investment
from synthetic_experiment import adapter_input, adapter_output


work_dir = '../../data/flights/'

n_runs = 10


def confuse(Psi, conf_prob, GT, custom_Nc = None):
    M = len(Psi)
    gt_obj = list(GT.keys())
    if custom_Nc is None:
        Nc = len(gt_obj)
    else:
        Nc = custom_Nc
    Cl = {}
    for i in range(Nc/2):
        Cl[gt_obj[i]] = {'id': i, 'other': gt_obj[Nc/2+i]}
        Cl[gt_obj[Nc/2+i]] = {'id': i, 'other': gt_obj[i]}
    c_Psi = [[] for obj in range(M)]
    GT_G = {}
    for obj in Cl.keys():
        GT_G[obj] = {}

    for obj in range(M):
        if obj in Cl:
            for s, val in Psi[obj]:
                # check that a confused source hasn't voted already on the 'other' object
                if np.random.rand() >= conf_prob and s not in [x[0] for x in Psi[Cl[obj]['other']]] and val == GT[obj]:
                    c_Psi[Cl[obj]['other']].append((s, val))
                    GT_G[Cl[obj]['other']][s] = 0
                else:
                    c_Psi[obj].append((s, val))
                    GT_G[obj][s] = 1
        else:
            for s, val in Psi[obj]:
                c_Psi[obj].append((s, val))

    return GT_G, Cl, c_Psi


def load_dataset():
    Psi = []
    M = 0
    Ns = []
    with open(work_dir + 'data/data.txt') as f:
        for line in f:
            obj_votes = []
            vals = line.strip().split('\t')
            N = len(vals)
            Ns.append(N)
            for s in range(N):
                if vals[s] != '':
                    obj_votes.append((s, 'O'+str(M)+'_'+vals[s]))
            if len(obj_votes) == 0:
                obj_votes = [(1, 'O'+str(M)+'_0'), (2, 'O'+str(M)+'_0')]
            Psi.append(obj_votes)
            M += 1
    # there is a varying number of sources per object (apparently, a data quality issue), so we choose the max number of
    # sources.
    N = max(Ns)

    GT = {}
    with open(work_dir + 'data/truth_sample.txt') as f:
        for line in f:
            vals = line.strip().split('\t')
            if len(vals) == 2:
                GT[int(vals[0])] = 'O'+vals[0]+'_'+vals[1]
    return N, M, Psi, GT


def properties():
    """
    Print the flight dataset properties.
    """
    N, M, Psi, GT = load_dataset()
    print('# of sources: {}'.format(N))
    print('# of object: {}'.format(M))
    obs_n = 0
    V = [0 for obj in range(M)]
    for obj in range(M):
        obs_n += len(Psi[obj])
        V[obj] = len(set([val for s, val in Psi[obj]]))
    print('# of observations: {}'.format(obs_n))
    print('average # of values per object: {:1.3f}'.format(np.average(V)))
    print('min # of values per object: {:1.3f}'.format(min(V)))
    print('max # of values per object: {:1.3f}'.format(max(V)))


def accuracy():
    """
    Vary the confusion probability on real data.
    """
    N, M, Psi_t, GT = load_dataset()

    # inject confusions
    conf_probs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    mcmc_params = {'N_iter': 10, 'burnin': 1, 'thin': 2, 'FV': 3}
    res = {'sums': [], 'mv': [], 'em': [], 'mcmc': [],
           'sums_f': [], 'mv_f': [], 'em_f': [], 'mcmc_f': [],
           'conf_probs': conf_probs, 'avlog': [], 'avlog_f': [],
           'inv': [], 'inv_f': [], 'pinv': [], 'pinv_f': [],
           'mv_p': [], 'em_p': [], 'mcmc_p': [],
           'mv_f_p': [], 'em_f_p': [], 'mcmc_f_p': []}
    for conf_prob in conf_probs:
        mv_accu, em_accu, mcmc_accu, sums_accu, avlog_accu, inv_accu, \
        pinv_accu, mv_accu_p, em_accu_p, mcmc_accu_p = [], [], [], [], [], [], [], [], [], []
        mv_accu_f, em_accu_f, mcmc_accu_f, sums_accu_f, avlog_accu_f, \
        inv_accu_f, pinv_accu_f, mv_accu_f_p, em_accu_f_p, mcmc_accu_f_p = [], [], [], [], [], [], [], [], [], []
        for run in range(n_runs):
            GT_G, Cl, Psi = confuse(Psi_t, 1-conf_prob, GT)

            # MV
            mv_p = majority_voting(Psi)
            mv_b = prob_binary_convert(mv_p)

            # EM
            em_A, em_p = expectation_maximization(N, M, Psi)
            em_b = prob_binary_convert(em_p)

            # MCMC
            mcmc_A, mcmc_p = mcmc(N, M, Psi, mcmc_params)
            mcmc_b = prob_binary_convert(mcmc_p)

            data = adapter_input(Psi)
            # SUMS
            sums_belief = sums(N, data)
            sums_b = adapter_output(sums_belief, data)

            # AVG LOG
            avlog_belief = average_log(N, data)
            avlog_b = adapter_output(avlog_belief, data)

            # INVESTMENT
            inv_belief = investment(N, data)
            inv_b = adapter_output(inv_belief, data)

            # POOLED INVESTMENT
            pinv_belief = pooled_investment(N, data)
            pinv_b = adapter_output(pinv_belief, data)

            # FUZZY FUSION Psi
            # From now Psi is the same as Psi_fussy due to Python
            Psi_fussy = f_mcmc(N, M, deepcopy(Psi), Cl, {'N_iter': 30, 'burnin': 5, 'thin': 3, 'FV': 4})[1]
            data_f = adapter_input(Psi_fussy)

            mv_pf = majority_voting(Psi_fussy)
            mv_bf = prob_binary_convert(mv_pf)

            em_Af, em_pf = expectation_maximization(N, M, Psi_fussy)
            em_bf = prob_binary_convert(em_pf)

            mcmc_Af, mcmc_pf = mcmc(N, M, Psi_fussy, mcmc_params)
            mcmc_bf = prob_binary_convert(mcmc_pf)

            sums_belief_f = sums(N, data_f)
            sums_bf = adapter_output(sums_belief_f, data_f)

            avlog_belief_f = average_log(N, data_f)
            avlog_bf = adapter_output(avlog_belief_f, data_f)

            inv_belief_f = investment(N, data_f)
            inv_bf = adapter_output(inv_belief_f, data_f)

            pinv_belief_f = pooled_investment(N, data_f)
            pinv_bf = adapter_output(pinv_belief_f, data_f)

            # exclude objects on which no conflicts
            obj_with_conflicts = []
            for obj_id, obj in enumerate(mv_b):
                if len(obj) > 1 and obj_id in GT.keys():
                    obj_with_conflicts.append(obj_id)

            # Binary output
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

            # Probability as output
            mv_accu_p.append(np.average([mv_p[obj][GT[obj]] for obj in obj_with_conflicts]))
            mv_accu_f_p.append(np.average([mv_pf[obj][GT[obj]] for obj in obj_with_conflicts]))

            em_accu_p.append(np.average([em_p[obj][GT[obj]] for obj in obj_with_conflicts]))
            em_accu_f_p.append(np.average([em_pf[obj][GT[obj]] for obj in obj_with_conflicts]))

            mcmc_accu_p.append(np.average([mcmc_p[obj][GT[obj]] for obj in obj_with_conflicts]))
            mcmc_accu_f_p.append(np.average([mcmc_pf[obj][GT[obj]] for obj in obj_with_conflicts]))

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

        res['mv_p'].append(np.average(mv_accu_p))
        res['mv_f_p'].append(np.average(mv_accu_f_p))

        res['em_p'].append(np.average(em_accu_p))
        res['em_f_p'].append(np.average(em_accu_f_p))

        res['mcmc_p'].append(np.average(mcmc_accu_p))
        res['mcmc_f_p'].append(np.average(mcmc_accu_f_p))

        print 'BINARY:'
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
        print 'PROBABILISTIC:'
        print('confusion probability: {}, mv_p: {:1.4f}, em_p: {:1.4f}, mcmc_p: {:1.4f}'
              .format(conf_prob,
                      np.average(mv_accu_p),
                      np.average(em_accu_p),
                      np.average(mcmc_accu_p)
                      ))
        print('confusion probability: {}, mv_f_p: {:1.4f}, em:_f_p {:1.4f}, mcmc_f_p: {:1.4f}'
              .format(conf_prob,
                      np.average(mv_accu_f_p),
                      np.average(em_accu_f_p),
                      np.average(mcmc_accu_f_p)
                      ))
    pd.DataFrame(res).to_csv(work_dir + 'experiments_results/flight_accuracy.csv', index=False)


def efficiency():
    """
    Efficiency as the number of clusters growing.
    """
    N, M, Psi, GT = load_dataset()

    # inject confusions
    Ncs = [10, 100, 1000, 10000]
    # mcmc_params = {'N_iter': 10, 'burnin': 1, 'thin': 2, 'FV': 3}
    mcmc_params = {'N_iter': 30, 'burnin': 5, 'thin': 3, 'FV': 4}
    res = {'mv': [],
           'mv std': [],
           'em': [],
           'em std': [],
           'mcmc': [],
           'mcmc std': [],
           'f_mcmc': [],
           'f_mcmc std': [],
           'sums': [],
           'sums std': [],
           'avlog': [],
           'avlog std': [],
           'inv': [],
           'inv std': [],
           'pinv': [],
           'pinv std': [],
           'number of objects with confusions': Ncs}
    for Nc in Ncs:
        times = [[], [], [], [], [], [], [], [], []]
        for run in range(n_runs):
            GT_G, Cl, cPsi = confuse(Psi, 0.8, GT, Nc)

            start = time.time()
            majority_voting(cPsi)
            times[0].append(time.time() - start)

            start = time.time()
            expectation_maximization(N, M, cPsi)
            times[1].append(time.time() - start)

            start = time.time()
            mcmc(N, M, cPsi, mcmc_params)
            times[2].append(time.time() - start)

            start = time.time()
            f_mcmc(N, M, cPsi, Cl, mcmc_params)
            times[3].append(time.time() - start)

            data = adapter_input(cPsi)
            start = time.time()
            sums(N, data)
            times[4].append(time.time() - start)

            start = time.time()
            average_log(N, data)
            times[5].append(time.time() - start)

            start = time.time()
            investment(N, data)
            times[6].append(time.time() - start)

            start = time.time()
            pooled_investment(N, data)
            times[7].append(time.time() - start)

        res['mv'].append(np.average(times[0]))
        res['em'].append(np.average(times[1]))
        res['mcmc'].append(np.average(times[2]))
        res['f_mcmc'].append(np.average(times[3]))
        res['sums'].append(np.average(times[4]))
        res['avlog'].append(np.average(times[5]))
        res['inv'].append(np.average(times[6]))
        res['pinv'].append(np.average(times[7]))

        res['mv std'].append(np.std(times[0]))
        res['em std'].append(np.std(times[1]))
        res['mcmc std'].append(np.std(times[2]))
        res['f_mcmc std'].append(np.std(times[3]))
        res['sums std'].append(np.std(times[4]))
        res['avlog std'].append(np.std(times[5]))
        res['inv std'].append(np.std(times[6]))
        res['pinv std'].append(np.std(times[7]))

        print('{}\tmv: {:1.4f}\tem: {:1.4f}\tmcmc: {:1.4f}\tf_mcmc: {:1.4f}\t{}'
              '\tsums: {:1.4f}\tavlog: {:1.4f}\tinv: {:1.4f}\tpinv: {:1.4f}'.format(Nc,
                                                                              np.average(times[0]),
                                                                              np.average(times[1]),
                                                                              np.average(times[2]),
                                                                              np.average(times[3]),
                                                                              np.average(times[4]),
                                                                              np.average(times[5]),
                                                                              np.average(times[6]),
                                                                              np.average(times[7]),
                                                                              )
              )

    pd.DataFrame(res).to_csv(work_dir + 'experiments_results/flight_efficiency.csv', index=False)


if __name__ == '__main__':
    # accuracy()
    efficiency()
    # properties()



