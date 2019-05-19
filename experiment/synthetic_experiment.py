from collections import defaultdict
from copy import deepcopy
import random
import numpy as np
import pandas as pd
from src.algorithm.mv import majority_voting
from src.algorithm.em import expectation_maximization
from src.algorithm.mcmc import mcmc
from src.algorithm.f_mcmc import f_mcmc
from src.algorithm.generator import synthesize
from src.algorithm.sums import sums
from src.algorithm.average_log import average_log
from src.algorithm.investment import investment
from src.algorithm.pooled_investment import pooled_investment
from src.algorithm.util import prob_binary_convert, accu_G, adapter_psi_dawid, adapter_prob_dawid
from src.algorithm.dawid_skene import dawid_skene

n_runs = 10

work_dir = '../../data/results/'


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
    M = 100
    # number of values per object
    V = 50
    # synthetically generated observations
    density = 0.5
    crowd_accuracy = 0.9
    mcmc_params = {'N_iter': 10, 'burnin': 2, 'thin': 3, 'FV': 0}
    conf_probs = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    res = {'sums': [], 'mv': [], 'em': [], 'mcmc': [],
           'sums_f': [], 'mv_f': [], 'em_f': [], 'mcmc_f': [],
           'conf_probs': conf_probs, 'avlog': [], 'avlog_f': [],
           'inv': [], 'inv_f': [], 'pinv': [], 'pinv_f': [], 'ds': [], 'ds_f': []}
    print('Crowd Accuracy: {}, Num of classes: {}'.format(crowd_accuracy, V))
    for conf_prob in conf_probs:
        GT, GT_G, Cl, Psi = synthesize(N, M, V, density, 1-conf_prob, crowd_accuracy)

        mv_accu, em_accu, mcmc_accu, sums_accu, avlog_accu, inv_accu, \
        pinv_accu = [], [], [], [], [], [], []
        mv_accu_f, em_accu_f, mcmc_accu_f, sums_accu_f, avlog_accu_f, \
        inv_accu_f, pinv_accu_f = [], [], [], [], [], [], []
        ds_accu, ds_accu_f = [], []
        for run in range(n_runs):
            # MV
            mv_p = majority_voting(Psi)
            mv_b = prob_binary_convert(mv_p)

            # EM
            em_A, em_p = expectation_maximization(N, M, Psi)
            em_b = prob_binary_convert(em_p)

            # Dawis and Skene
            Psi_dawid = adapter_psi_dawid(Psi)
            values_prob, _, classes = dawid_skene(Psi_dawid, tol=0.001, max_iter=50)
            ds_p = adapter_prob_dawid(values_prob, classes)
            ds_b = prob_binary_convert(ds_p)

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
            _, Psi_fussy, _, _ = f_mcmc(N, M, deepcopy(Psi), Cl, {'N_iter': 30, 'burnin': 5, 'thin': 3, 'FV': 4})
            data_f = adapter_input(Psi_fussy)

            mv_pf = majority_voting(Psi_fussy)
            mv_bf = prob_binary_convert(mv_pf)

            em_Af, em_pf = expectation_maximization(N, M, Psi_fussy)
            em_bf = prob_binary_convert(em_pf)

            Psi_dawid_f = adapter_psi_dawid(Psi_fussy)
            values_prob_f, _, classes = dawid_skene(Psi_dawid_f, tol=0.001, max_iter=50)
            ds_p_f = adapter_prob_dawid(values_prob_f, classes)
            ds_b_f = prob_binary_convert(ds_p_f)

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
                if len(obj) > 1:
                    obj_with_conflicts.append(obj_id)

            mv_accu.append(np.average([mv_b[obj][GT[obj]] for obj in obj_with_conflicts]))
            mv_accu_f.append(np.average([mv_bf[obj][GT[obj]] for obj in obj_with_conflicts]))

            em_accu.append(np.average([em_b[obj][GT[obj]] for obj in obj_with_conflicts]))
            em_accu_f.append(np.average([em_bf[obj][GT[obj]] for obj in obj_with_conflicts]))

            ds_accu.append(np.average([ds_b[obj][GT[obj]] for obj in obj_with_conflicts]))
            ds_accu_f.append(np.average([ds_b_f[obj][GT[obj]] for obj in obj_with_conflicts]))

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

        res['ds'].append(np.average(ds_accu))
        res['ds_f'].append(np.average(ds_accu_f))

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

        print('ORG|conf prob: {}, mv: {:1.3f}, em: {:1.3f}, D&S: {:1.3f}, mcmc: {:1.3f}, '
              'sums: {:1.3f}, avlog: {:1.3f}, inv: {:1.3f}, pinv: {:1.3f}'
              .format(conf_prob,
               np.average(mv_accu),
               np.average(em_accu),
               np.average(ds_accu),
               np.average(mcmc_accu),
               np.average(sums_accu),
               np.average(avlog_accu),
               np.average(inv_accu),
               np.average(pinv_accu)
            ))
        print('COR|conf prob: {}, mv: {:1.3f}, em: {:1.3f}, D&S: {:1.3f}, mcmc: {:1.3f}, '
              'sums: {:1.3f}, avlog: {:1.3f}, inv: {:1.3f}, pinv: {:1.3f}'
              .format(conf_prob,
               np.average(mv_accu_f),
               np.average(em_accu_f),
               np.average(ds_accu_f),
               np.average(mcmc_accu_f),
               np.average(sums_accu_f),
               np.average(avlog_accu_f),
               np.average(inv_accu_f),
               np.average(pinv_accu_f)
            ))
        print('-----------------------')

    pd.DataFrame(res).to_csv(work_dir + 'synthetic_accuracy_binary.csv', index=False)


def convergence():
    """
    Convergence of MCMC.
    """
    # number of sources
    N = 30
    # number of objects
    M = 500
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
            f_mcmc_G, _, _, _ = f_mcmc(N, M, Psi, Cl, params)
            G_accu = np.average(accu_G(f_mcmc_G, GT_G))
            runs.append(G_accu)
        res['G accuracy'].append(np.average(runs))
        res['error'].append(np.std(runs))

        print('p: {}, G accu: {}, std: {}'.format(p, np.average(runs), np.std(runs)))

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
    density = 0.5
    accuracy = 0.9
    conf_prob = 0.2
    Vs = [2, 4, 8, 16, 32, 64, 128]
    params = {'N_iter': 30, 'burnin': 5, 'thin': 3, 'FV': 4}
    res = {'G accuracy': [], 'error': [], 'number of distinct values per object': Vs}
    for V in Vs:
        GT, GT_G, Cl, Psi = synthesize(N, M, V, density, 1 - conf_prob, accuracy)
        G_accu = []
        for run in range(n_runs):
            f_mcmc_G, _, _ = f_mcmc(N, M, Psi, Cl, params)
            G_accu = np.average(accu_G(f_mcmc_G, GT_G))
        res['G accuracy'].append(np.average(G_accu))
        res['error'].append(np.std(G_accu))
        print('V: {}, accu: {:1.4f}'.format(V, np.average(G_accu)))

    pd.DataFrame(res).to_csv(work_dir + 'synthetic_values.csv', index=False)


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

    mcmc_params = {'N_iter': 30, 'burnin': 5, 'thin': 3, 'FV': 4}
    conf_probs = [0.2, 0.3, 0.4]
    s_acc_list = [0.6, 0.7, 0.8, 0.9, 1.]
    res = {'conf_probs': [], 'acc_g': [], 'acc_g_std': [], 's_acc': []}
    for conf_prob in conf_probs:
        for s_acc in s_acc_list:
            accu_G_list = []
            for run in range(n_runs):
                GT, GT_G, Cl, Psi = synthesize(N, M, V, density, 1 - conf_prob, s_acc)
                f_mcmc_G, _, _ = f_mcmc(N, M, Psi, Cl, mcmc_params)
                accu_G_list.append(accu_G(f_mcmc_G, GT_G))

            res['acc_g'].append(np.mean(accu_G_list))
            res['acc_g_std'].append(np.std(accu_G_list))
            res['s_acc'].append(s_acc)
            res['conf_probs'].append(conf_prob)

            print 's_acc: {}'.format(s_acc)
            print 'conf_prob: {}'.format(conf_prob)
            print 'acc G: {}'.format(np.mean(accu_G_list))
            print '---------------'
    pd.DataFrame(res).to_csv(work_dir + 'accuracy_g.csv', index=False)


def cluster_detection_bimodality_check():
    """
    Vary the confusion probability on synthetic data.
    """
    # number of items
    item_num = 50
    # number of classes
    V = 10
    # number of votes per item
    votes_item = 5
    # number of votes per worker
    votes_worker = 25
    crowd_accuracy = [0.7, 0.9]
    # confusing classes
    conf_class = {0: 1, 1: 0}
    # % of workers who does confusions
    conf_workers_prop_list = [0.5, 0.1, 0.2, 0.3, 0.4, 0.5]
    print('Crowd Accuracy: {}, Num of classes: {}'.format(crowd_accuracy, V))
    for conf_workers_prop in conf_workers_prop_list:
        Psi = generate_data_bimodality_check(item_num, V, conf_workers_prop, votes_item, votes_worker, crowd_accuracy, conf_class)

        # Dawis and Skene
        Psi_dawid = adapter_psi_dawid(Psi)
        values_prob, ErrM, classes = dawid_skene(Psi_dawid, tol=0.001, max_iter=50)
        pass


def generate_data_bimodality_check(items_num, V, conf_workers_prop, votes_item, votes_worker, acc_range, conf_class):
    workers_num = (items_num * votes_item) // votes_worker
    item_ids = list(range(items_num))
    worker_ids = {worker_id: votes_worker for worker_id in range(workers_num)}
    worker_ids_acc = {worker_id: np.random.uniform(acc_range[0], acc_range[1]) for worker_id in range(workers_num)}
    GT = {item_id: np.random.randint(0, V) for item_id in item_ids}
    conf_workers_num = int(conf_workers_prop * workers_num)
    conf_workers_id = set(np.random.choice(worker_ids.keys(), conf_workers_num, replace=False))
    Psi = []
    for item_id in item_ids:
        item_votes = []
        gt_item = GT[item_id]
        worker_ids_ = worker_ids.keys()
        random.shuffle(worker_ids_)
        for worker_id in worker_ids_[:votes_item]:
            if np.random.binomial(1, worker_ids_acc[worker_id]):
                if (gt_item in conf_class) and (worker_id in conf_workers_id):  # generate confusion
                    item_votes.append([worker_id, conf_class[gt_item]])
                else:
                    item_votes.append([worker_id, gt_item])
            else:
                wrong_vote = np.random.choice([i for i in range(V) if i != gt_item], 1)[0]
                item_votes.append([worker_id, wrong_vote])
            # reduce number of votes needed from a worker_id
            worker_ids[worker_id] -= 1
            if worker_ids[worker_id] == 0:
                del worker_ids[worker_id]
        Psi.append(item_votes)
    return Psi


if __name__ == '__main__':
    '''
    Gt_G = {
            obj_id: {source_id: g_val, source_id2: g_val2, ..},
            ...
            },
    Psi = [
           obj_id: [ [s_id1, val1], [s_id2, val2],..],
           ...
           ]
    '''
    cluster_detection_bimodality_check()
    # accuracy()
    # convergence()
    # values()
    # get_acc_g()

