import numpy as np
import pandas as pd
from copy import deepcopy
from src.algorithm.em import expectation_maximization
from src.algorithm.mv import majority_voting
from src.algorithm.mcmc import mcmc
from src.algorithm.f_mcmc import f_mcmc
from src.algorithm.util import accu_G, prob_binary_convert, precision_recall, adapter_psi_dawid, adapter_prob_dawid
from src.algorithm.sums import sums
from src.algorithm.average_log import average_log
from src.algorithm.investment import investment
from src.algorithm.pooled_investment import pooled_investment
from src.algorithm.dawid_skene import dawid_skene
from data_loader import load_data_faces, load_data_flags, load_data_plots, load_data_food, TruncaterVotesItem
from synthetic_experiment import adapter_input, adapter_output

n_runs = 50


def accuracy(load_data, Truncater=None):
    res = {'accuracy': [],
           'accuracy_std': [],
           'methods': ['mv_p', 'em_p', 'mcmc_p',
                       'mv_f_p', 'em_f_p', 'mcmc_f_p',
                       'mv_b', 'em_b', 'mcmc_b',
                       'mv_f_b', 'em_f_b', 'mcmc_f_b',
                       'sums', 'avlog', 'inv', 'pinv',
                       'sums_f', 'avlog_f', 'inv_f', 'pinv_f',
                       'mcmc_conf_p', 'mcmc_conf_b', 'D&S_p',
                       'D&S_b', 'D&S_f_p', 'D&S_f_b'
                       ]}
    runs = [[] for _ in range(26)]
    G_accu_p_list, G_accu_b_list, G_precision_list, G_recall_list = [], [], [], []
    mcmc_params = {'N_iter': 10, 'burnin': 1, 'thin': 2}
    run = 0
    while run < n_runs:
        N, M, Psi, GT, Cl, GT_G = load_data(Truncater)
        # PROBABILISTIC OUTPUT
        mv_p = majority_voting(Psi)
        em_A, em_p = expectation_maximization(N, M, Psi)
        mcmc_A, mcmc_p = mcmc(N, M, Psi, mcmc_params)

        f_mcmc_G, Psi_fussy, mcmc_conf_p = f_mcmc(N, M, deepcopy(Psi), Cl, {'N_iter': 30, 'burnin': 5, 'thin': 3, 'FV': 4})
        if [] in Psi_fussy:  # check the border case when all votes on an item considered as confused
            print('empty fussion, repeat')
            continue
        else:
            run += 1
        print(run)
        precision, recall, G_accu_b = precision_recall(f_mcmc_G, GT_G)
        G_precision_list.append(precision)
        G_recall_list.append(recall)
        G_accu_b_list.append(G_accu_b)

        # only for 'flags' dataset
        if 'flags' in load_data.__name__:
            # compute G ACCuracy only on clusters where we belief confusions might happen
            f_mcmc_G_clust, GT_G_clust = {}, {}
            num_of_clusters = 13
            for obj_id in range(num_of_clusters):
                f_mcmc_G_clust[obj_id] = f_mcmc_G[obj_id]
                f_mcmc_G_clust[obj_id + M/2] = f_mcmc_G[obj_id + M/2]
                GT_G_clust[obj_id] = GT_G[obj_id]
                GT_G_clust[obj_id + M/2] = GT_G[obj_id + M/2]
            G_accu_p_list.append(accu_G(f_mcmc_G_clust, GT_G_clust))
        else:
            G_accu_p_list.append(accu_G(f_mcmc_G, GT_G))

        mv_f_p = majority_voting(Psi_fussy)
        em_f_A, em_f_p = expectation_maximization(N, M, Psi_fussy)
        mcmc_f_A, mcmc_f_p = mcmc(N, M, Psi_fussy, mcmc_params)

        # Dawis and Skene
        Psi_dawid = adapter_psi_dawid(Psi)
        values_prob, _, classes = dawid_skene(Psi_dawid, tol=0.001, max_iter=50)
        ds_p = adapter_prob_dawid(values_prob, classes)

        Psi_dawid_f = adapter_psi_dawid(Psi_fussy)
        values_prob_f, _, classes = dawid_skene(Psi_dawid_f, tol=0.001, max_iter=50)
        ds_p_f = adapter_prob_dawid(values_prob_f, classes)

        # BINARY OUTPUT
        data = adapter_input(Psi)
        data_f = adapter_input(Psi_fussy)

        mv_b = prob_binary_convert(mv_p)
        em_b = prob_binary_convert(em_p)
        mcmc_b = prob_binary_convert(mcmc_p)
        mv_f_b = prob_binary_convert(mv_f_p)
        em_f_b = prob_binary_convert(em_f_p)
        mcmc_f_b = prob_binary_convert(mcmc_f_p)
        ds_b = prob_binary_convert(ds_p)
        ds_b_f = prob_binary_convert(ds_p_f)
        mcmc_conf_b = prob_binary_convert(mcmc_conf_p)  # our algorithm MCMC-C

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

        mv_hits = []
        em_hits = []
        mcmc_hits = []
        mv_b_hits = []
        em_b_hits = []
        mcmc_b_hits = []
        sums_hits = []
        avlog_hits = []
        inv_hits = []
        pinv_hits = []
        mv_f_hits = []
        em_f_hits = []
        mcmc_f_hits = []
        mv_f_b_hits = []
        em_f_b_hits = []
        mcmc_f_b_hits = []
        sums_f_hits = []
        avlog_f_hits = []
        inv_f_hits = []
        pinv_f_hits = []
        mcmc_conf_p_hist = []
        mcmc_conf_b_hist = []
        ds_p_hits, ds_b_hits = [], []
        ds_p_f_hits, ds_b_f_hits = [], []

        # slect objects with conflicting votes among ones are in clusters
        obj_with_conflicts = []
        for obj_id, obj in enumerate(mv_p):
            # only for 'flags' dataset
            if 'flags' in load_data.__name__:
                if len(obj) > 1 and (obj_id < num_of_clusters or M/2 <= obj_id < M/2+num_of_clusters):
                    obj_with_conflicts.append(obj_id)
            else:
                if len(obj) > 1:
                    obj_with_conflicts.append(obj_id)

        for obj in range(M):
            if obj in obj_with_conflicts:
                # PROBABILISTIC OUTPUT
                mv_hits.append(mv_p[obj][GT[obj]])
                em_hits.append(em_p[obj][GT[obj]])
                mcmc_hits.append(mcmc_p[obj][GT[obj]])
                ds_p_hits.append(ds_p[obj][GT[obj]])
                ds_p_f_hits.append(ds_p_f[obj][GT[obj]])

                mv_f_hits.append(mv_f_p[obj][GT[obj]])
                em_f_hits.append(em_f_p[obj][GT[obj]])
                mcmc_f_hits.append(mcmc_f_p[obj][GT[obj]])

                mcmc_conf_p_hist.append(mcmc_conf_p[obj][GT[obj]])

                # BINARY OUTPUT
                mv_b_hits.append(mv_b[obj][GT[obj]])
                em_b_hits.append(em_b[obj][GT[obj]])
                mcmc_b_hits.append(mcmc_b[obj][GT[obj]])
                ds_b_hits.append(ds_b[obj][GT[obj]])
                ds_b_f_hits.append(ds_b_f[obj][GT[obj]])
                mv_f_b_hits.append(mv_f_b[obj][GT[obj]])
                em_f_b_hits.append(em_f_b[obj][GT[obj]])
                mcmc_f_b_hits.append(mcmc_f_b[obj][GT[obj]])

                sums_hits.append(sums_b[obj][GT[obj]])
                sums_f_hits.append(sums_bf[obj][GT[obj]])

                avlog_hits.append(avlog_b[obj][GT[obj]])
                avlog_f_hits.append(avlog_bf[obj][GT[obj]])

                inv_hits.append(inv_b[obj][GT[obj]])
                inv_f_hits.append(inv_bf[obj][GT[obj]])

                pinv_hits.append(pinv_b[obj][GT[obj]])
                pinv_f_hits.append(pinv_bf[obj][GT[obj]])

                mcmc_conf_b_hist.append(mcmc_conf_b[obj][GT[obj]])

        runs[0].append(np.average(mv_hits))
        runs[1].append(np.average(em_hits))
        runs[2].append(np.average(mcmc_hits))
        runs[3].append(np.average(mv_f_hits))
        runs[4].append(np.average(em_f_hits))
        runs[5].append(np.average(mcmc_f_hits))
        runs[6].append(np.average(mv_b_hits))
        runs[7].append(np.average(em_b_hits))
        runs[8].append(np.average(mcmc_b_hits))
        runs[9].append(np.average(mv_f_b_hits))
        runs[10].append(np.average(em_f_b_hits))
        runs[11].append(np.average(mcmc_f_b_hits))
        runs[12].append(np.average(sums_hits))
        runs[13].append(np.average(avlog_hits))
        runs[14].append(np.average(inv_hits))
        runs[15].append(np.average(pinv_hits))
        runs[16].append(np.average(sums_f_hits))
        runs[17].append(np.average(avlog_f_hits))
        runs[18].append(np.average(inv_f_hits))
        runs[19].append(np.average(pinv_f_hits))
        runs[20].append(np.average(mcmc_f_hits))
        runs[21].append(np.average(mcmc_f_b_hits))
        runs[22].append(np.average(ds_p_hits))
        runs[23].append(np.average(ds_b_hits))
        runs[24].append(np.average(ds_p_f_hits))
        runs[25].append(np.average(ds_b_f_hits))

    print('G Accu prob: {:1.4f}+-{:1.4f}'.format(np.average(G_accu_p_list), np.std(G_accu_p_list)))
    print('G Accu bin: {:1.4f}+-{:1.4f}'.format(np.average(G_accu_b_list), np.std(G_accu_b_list)))
    print('G precision: {:1.4f}+-{:1.4f}'.format(np.average(G_precision_list), np.std(G_precision_list)))
    print('G recall: {:1.4f}+-{:1.4f}'.format(np.average(G_recall_list), np.std(G_recall_list)))
    print 'PROBABILISTIC OUTPUT'
    print('mv: {:1.4f}+-{:1.4f}'.format(np.average(runs[0]), np.std(runs[0])))
    print('mv_f: {:1.4f}+-{:1.4f}'.format(np.average(runs[3]), np.std(runs[3])))
    print('em: {:1.4f}+-{:1.4f}'.format(np.average(runs[1]), np.std(runs[1])))
    print('em_f: {:1.4f}+-{:1.4f}'.format(np.average(runs[4]), np.std(runs[4])))
    print('D&S: {:1.4f}+-{:1.4f}'.format(np.average(runs[22]), np.std(runs[22])))
    print('D&S_f: {:1.4f}+-{:1.4f}'.format(np.average(runs[24]), np.std(runs[24])))
    print('mcmc: {:1.4f}+-{:1.4f}'.format(np.average(runs[2]), np.std(runs[2])))
    print('mcmc_f: {:1.4f}+-{:1.4f}'.format(np.average(runs[5]), np.std(runs[5])))
    print('*mcmc_conf_p*: {:1.4f}+-{:1.4f}'.format(np.average(runs[20]), np.std(runs[20])))
    print 'BINARY OUTPUT'
    print('*mcmc_conf_b*: {:1.4f}+-{:1.4f}'.format(np.average(runs[21]), np.std(runs[21])))
    print('mv: {:1.4f}+-{:1.4f}'.format(np.average(runs[6]), np.std(runs[6])))
    print('mv_f: {:1.4f}+-{:1.4f}'.format(np.average(runs[9]), np.std(runs[9])))
    print('em: {:1.4f}+-{:1.4f}'.format(np.average(runs[7]), np.std(runs[7])))
    print('em_f: {:1.4f}+-{:1.4f}'.format(np.average(runs[10]), np.std(runs[10])))
    print('D&S: {:1.4f}+-{:1.4f}'.format(np.average(runs[23]), np.std(runs[23])))
    print('D&S_f: {:1.4f}+-{:1.4f}'.format(np.average(runs[25]), np.std(runs[25])))
    print('mcmc: {:1.4f}+-{:1.4f}'.format(np.average(runs[8]), np.std(runs[8])))
    print('mcmc_f: {:1.4f}+-{:1.4f}'.format(np.average(runs[11]), np.std(runs[11])))
    print('sums: {:1.4f}+-{:1.4f}'.format(np.average(runs[12]), np.std(runs[12])))
    print('sums_f: {:1.4f}+-{:1.4f}'.format(np.average(runs[16]), np.std(runs[16])))
    print('avlog: {:1.4f}+-{:1.4f}'.format(np.average(runs[13]), np.std(runs[13])))
    print('avlog_f: {:1.4f}+-{:1.4f}'.format(np.average(runs[17]), np.std(runs[17])))
    print('inv: {:1.4f}+-{:1.4f}'.format(np.average(runs[14]), np.std(runs[14])))
    print('inv_f: {:1.4f}+-{:1.4f}'.format(np.average(runs[18]), np.std(runs[18])))
    print('pinv: {:1.4f}+-{:1.4f}'.format(np.average(runs[15]), np.std(runs[15])))
    print('pinv_f: {:1.4f}+-{:1.4f}'.format(np.average(runs[19]), np.std(runs[19])))

    for run in runs:
        res['accuracy'].append(np.average(run))
        res['accuracy_std'].append(np.std(run))

    # Save results in a CSV
    # pd.DataFrame(res).to_csv('../data/results/accuracy_.csv', index=False)


if __name__ == '__main__':
    datasets = ['faces', 'flags', 'food', 'plots']

    dataset_name = datasets[2]
    if dataset_name == 'faces':
        load_data = load_data_faces
    elif dataset_name == 'flags':
        load_data = load_data_flags
    elif dataset_name == 'food':
        load_data = load_data_food
    elif dataset_name == 'plots':
        load_data = load_data_plots
    else:
        print('Dataset not selected')
        exit(1)
    print('Dataset: {}'.format(dataset_name))

    for votes_per_item in [3, 10, 'All']:
        print('Votes: ', votes_per_item)
        if votes_per_item == 'All':
            accuracy(load_data)
        else:
            Truncater = TruncaterVotesItem(votes_per_item)
            accuracy(load_data, Truncater)

