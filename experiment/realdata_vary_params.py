import os
import numpy as np
import pandas as pd
from copy import deepcopy
from src.algorithm.em import expectation_maximization
from src.algorithm.mv import majority_voting
from src.algorithm.mcmc import mcmc
from src.algorithm.f_mcmc import f_mcmc
from src.algorithm.util import accu_G, prob_binary_convert, precision_recall, conf_ranks_precision, \
    adapter_psi_dawid, adapter_prob_dawid, invert, get_ds_G, do_conf_ranks_ds, do_conf_ranks_fmcmc
from src.algorithm.sums import sums
from src.algorithm.average_log import average_log
from src.algorithm.investment import investment
from src.algorithm.pooled_investment import pooled_investment
from src.algorithm.dawid_skene import dawid_skene
from data_loader import load_data_faces, load_data_flags, load_data_plots, load_data_food, \
    TruncaterVotesItem, load_gt_conf_ranks, load_gt_conf_ranks_faces
from synthetic_experiment import adapter_input, adapter_output

n_runs = 50


def accuracy(load_data, dataset_name, votes_per_item, Truncater=None):
        ## load ground truth of clusters having confusable classes
    if dataset_name == 'faces':
        gt_conf_ranks = load_gt_conf_ranks_faces()
    elif dataset_name == 'flags':
        df1 = pd.read_csv('../data/Flags/flags1_res_postporos.csv')
        df2 = pd.read_csv('../data/Flags/flags2_res_postporos.csv')
        gt_conf_ranks = load_gt_conf_ranks(df1, df2)
    elif dataset_name == 'food':
        df1 = pd.read_csv('../data/Food/food1_res_postporos.csv')
        df2 = pd.read_csv('../data/Food/food2_res_postporos.csv')
        gt_conf_ranks = load_gt_conf_ranks(df1, df2)
    elif dataset_name == 'plots':
        df1 = pd.read_csv('../data/Plots/plots1_res_postporos.csv')
        df2 = pd.read_csv('../data/Plots/plots2_res_postporos.csv')
        gt_conf_ranks = load_gt_conf_ranks(df1, df2)
    else:
        exit(1)

    runs = [[] for _ in range(26)]
    G_accu_p_list, G_accu_b_list, G_precision_list, G_recall_list = [], [], [], []
    G_acc_b_DS, G_precision_DS_list, G_recall_DS_list = [], [], []
    ## data structure for statistics of precion in detecting clusters having confusable classes
    conf_ranks_pr_fmcmc_sum = [0] * gt_conf_ranks.shape[0]
    conf_ranks_pr_ds_sum = [0] * gt_conf_ranks.shape[0]
    mcmc_params = {'N_iter': 10, 'burnin': 1, 'thin': 2}
    run = 0
    while run < n_runs:
        N, M, Psi, GT, Cl, GT_G = load_data(Truncater)
        # PROBABILISTIC OUTPUT
        mv_p = majority_voting(Psi)
        em_A, em_p = expectation_maximization(N, M, Psi)
        mcmc_A, mcmc_p = mcmc(N, M, Psi, mcmc_params)

        f_mcmc_G, Psi_fussy, mcmc_conf_p, Cl_conf_scores = f_mcmc(N, M, deepcopy(Psi), Cl, {'N_iter': 30, 'burnin': 5, 'thin': 3, 'FV': 4})
        if [] in Psi_fussy:  # check the border case when all votes on an item considered as confused
            print('empty fussion, repeat')
            continue
        else:
            run += 1
        print(run)

        ## Dawis and Skene
        Psi_dawid = adapter_psi_dawid(Psi)
        values_prob, ErrM, classes = dawid_skene(Psi_dawid, tol=0.001, max_iter=50)
        ds_p = adapter_prob_dawid(values_prob, classes)
        ## D&S accuracy in confusion detection
        conf_ranks_ds = do_conf_ranks_ds(ErrM, classes, ds_p, Psi)  # ranked pairs of classes that might be confused
        conf_ranks_pr_ds = conf_ranks_precision(gt_conf_ranks[:, 0], conf_ranks_ds[:, 0])
        conf_ranks_pr_ds_sum += conf_ranks_pr_ds
        try:
            ds_G = get_ds_G(ErrM, classes, ds_p, Psi)
        except ValueError:
            print('VALUE ERROR')
            run -= 1
            continue

        ds_conf_precision, ds_conf_recall, ds_conf_acc = precision_recall(ds_G, GT_G)
        G_acc_b_DS.append(ds_conf_acc)
        G_precision_DS_list.append(ds_conf_recall)
        G_recall_DS_list.append(ds_conf_precision)

        Psi_dawid_f = adapter_psi_dawid(Psi_fussy)
        values_prob_f, _, classes = dawid_skene(Psi_dawid_f, tol=0.001, max_iter=50)
        ds_p_f = adapter_prob_dawid(values_prob_f, classes)

        ## cluster detection evaluation
        conf_ranks_fmcmc = do_conf_ranks_fmcmc(Cl_conf_scores, M, Cl, Psi, mcmc_conf_p)  # ranked pairs of classes that might be confused
        conf_ranks_pr_fmcmc = conf_ranks_precision(gt_conf_ranks[:, 0], conf_ranks_fmcmc[:, 0])
        conf_ranks_pr_fmcmc_sum += conf_ranks_pr_fmcmc

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

    ## confusion detection MCMC-CONF
    G_acc_p, G_acc_p_std = np.average(G_accu_p_list), np.std(G_accu_p_list)
    G_acc_b, G_acc_b_std = np.average(G_accu_b_list), np.std(G_accu_b_list)
    G_precision, G_precision_std = np.average(G_precision_list), np.std(G_precision_list)
    G_recall, G_recall_std = np.average(G_recall_list), np.std(G_recall_list)

    ## confusion detection D&S
    G_acc_ds, G_acc_ds_std = np.average(G_acc_b_DS), np.std(G_acc_b_DS)
    G_ds_precision, G_ds_precision_std = np.average(G_precision_DS_list), np.std(G_precision_DS_list)
    G_ds_recall, G_ds_recall_std = np.average(G_recall_DS_list), np.std(G_recall_DS_list)

    # ## precision in cluster detection
    # print('Precision in Cluster Detection')
    # print('MCMC-CONF: {}'.format(conf_ranks_pr_fmcmc_sum / n_runs))
    # print('D&S      : {}'.format(conf_ranks_pr_ds_sum / n_runs))

    print('Confusion Detection')
    print('MCMC-CONF')
    print('G Accu prob: {:1.4f}+-{:1.4f}'.format(G_acc_p, G_acc_ds_std))
    print('G Accu bin: {:1.4f}+-{:1.4f}'.format(G_acc_b, G_acc_b_std))
    print('G precision: {:1.4f}+-{:1.4f}'.format(G_precision, G_precision_std))
    print('G recall: {:1.4f}+-{:1.4f}'.format(G_recall, G_recall_std))
    print('D&S')
    print('G Accu bin: {:1.4f}+-{:1.4f}'.format(G_acc_ds, G_acc_b_std))
    print('G precision: {:1.4f}+-{:1.4f}'.format(G_ds_precision, G_ds_precision_std))
    print('G recall: {:1.4f}+-{:1.4f}'.format(G_ds_recall, G_ds_recall_std))
    print('\n')
    print('PROBABILISTIC OUTPUT')
    print('mv: {:1.4f}+-{:1.4f}'.format(np.average(runs[0]), np.std(runs[0])))
    print('mv_f: {:1.4f}+-{:1.4f}'.format(np.average(runs[3]), np.std(runs[3])))
    print('em: {:1.4f}+-{:1.4f}'.format(np.average(runs[1]), np.std(runs[1])))
    print('em_f: {:1.4f}+-{:1.4f}'.format(np.average(runs[4]), np.std(runs[4])))
    print('D&S: {:1.4f}+-{:1.4f}'.format(np.average(runs[22]), np.std(runs[22])))
    print('D&S_f: {:1.4f}+-{:1.4f}'.format(np.average(runs[24]), np.std(runs[24])))
    print('mcmc: {:1.4f}+-{:1.4f}'.format(np.average(runs[2]), np.std(runs[2])))
    print('mcmc_f: {:1.4f}+-{:1.4f}'.format(np.average(runs[5]), np.std(runs[5])))
    print('*mcmc_conf_p*: {:1.4f}+-{:1.4f}'.format(np.average(runs[20]), np.std(runs[20])))
    print('BINARY OUTPUT')
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

    ## *** Making dataFrame of results (precision in cluster detection) ***
    data_cl = []
    ## add MCMC-CONF
    data_cl += list(zip([votes_per_item]*gt_conf_ranks.shape[0], list(range(1, gt_conf_ranks.shape[0]+1)),
                        conf_ranks_pr_fmcmc_sum / n_runs, ['mcmc_conf']*gt_conf_ranks.shape[0]))
    ## add D&S
    data_cl += list(zip([votes_per_item] * gt_conf_ranks.shape[0], list(range(1, gt_conf_ranks.shape[0] + 1)),
                        conf_ranks_pr_ds_sum / n_runs, ['D&S'] * gt_conf_ranks.shape[0]))
    ## Save results in a CSV
    df_cl = pd.DataFrame(data_cl, columns=['votes_per_item', 'top-k', 'precision', 'method'])
    path = '../data/results/{}_cluster_detection.csv'.format(dataset_name)
    if os.path.isfile(path):
        df_prev = pd.read_csv(path)
        df_new = df_prev.append(df_cl, ignore_index=True)
        df_new.to_csv(path, index=False)
    else:
        df_cl.to_csv(path, index=False)

    ## *** Making dataFrame of results (confusion detection and correction) ***
    method_list = ['mv_p', 'truth_finder_p', 'mcmc_p',
               'mv_f_p', 'truth_finder_f_p', 'mcmc_f_p',
               'mv_b', 'truth_finder_b', 'mcmc_b',
               'mv_f_b', 'truth_finder_f_b', 'mcmc_f_b',
               'sums_b', 'avlog_b', 'inv_b', 'pinv_b',
               'sums_f_b', 'avlog_f_b', 'inv_f_b', 'pinv_f_b',
               'mcmc_conf_p', 'mcmc_conf_b', 'D&S_p',
               'D&S_b', 'D&S_f_p', 'D&S_f_b'
               ]
    method_indexes = {
        'mv': {'p': method_list.index('mv_p'), 'b': method_list.index('mv_b')},
        'mv_f': {'p': method_list.index('mv_f_p'), 'b': method_list.index('mv_f_b')},
        'truth_finder': {'p': method_list.index('truth_finder_p'), 'b': method_list.index('truth_finder_b')},
        'truth_finder_f': {'p': method_list.index('truth_finder_f_p'), 'b': method_list.index('truth_finder_f_b')},
        'mcmc': {'p': method_list.index('mcmc_p'), 'b': method_list.index('mcmc_b')},
        'mcmc_f': {'p': method_list.index('mcmc_f_p'), 'b': method_list.index('mcmc_f_b')},
        'D&S': {'p': method_list.index('D&S_p'), 'b': method_list.index('D&S_b')},
        'D&S_f': {'p': method_list.index('D&S_f_p'), 'b': method_list.index('D&S_f_b')},
        'sums': {'b': method_list.index('sums_b')},
        'sums_f': {'b': method_list.index('sums_f_b')},
        'avlog': {'b': method_list.index('avlog_b')},
        'avlog_f': {'b': method_list.index('avlog_f_b')},
        'inv': {'b': method_list.index('inv_b')},
        'inv_f': {'b': method_list.index('inv_f_b')},
        'pinv': {'b': method_list.index('pinv_b')},
        'pinv_f': {'b': method_list.index('pinv_f_b')},
        'mcmc_conf': {'p': method_list.index('mcmc_conf_b'), 'b': method_list.index('mcmc_conf_p')}
    }
    columns = ['votes_per_item', 'method', 'accuracy_conf', 'accuracy_conf_std', 'accuracy_prob_conf',
               'accuracy_prob_conf_std', 'precision_conf', 'precision_std', 'recall_conf', 'recall_conf_std',
               'accuracy', 'accuracy_std', 'accuracy_prob', 'accuracy_prob_std',
               'ds_accuracy_conf', 'ds_accuracy_conf_std', 'ds_precision_conf', 'ds_precision_std',
               'ds_recall_conf', 'ds_recall_conf_std']
    data = []
    for method, m_ in method_indexes.items():
        if method == 'mcmc_conf':
            p, b = m_['p'], m_['b']
            data.append([votes_per_item, method, G_acc_b, G_acc_b_std, G_acc_p, G_acc_p_std, G_precision, G_precision_std,
                         G_recall, G_recall_std, np.average(runs[b]), np.std(runs[b]), np.average(runs[p]), np.std(runs[p]),
                         None, None, None, None, None, None])
        elif 'p' in m_.keys():
            p, b = m_['p'], m_['b']
            if method is not 'D&S':
                data.append([votes_per_item, method, None, None, None, None, None, None, None, None,
                             np.average(runs[b]), np.std(runs[b]), np.average(runs[p]), np.std(runs[p]),
                             None, None, None, None, None, None])
            else:
                data.append([votes_per_item, method, None, None, None, None, None, None, None, None,
                             np.average(runs[b]), np.std(runs[b]), np.average(runs[p]), np.std(runs[p]),
                             G_acc_ds, G_acc_ds_std, G_ds_precision, G_ds_precision_std, G_ds_recall, G_ds_recall_std])
        else:
            b = m_['b']
            data.append([votes_per_item, method, None, None, None, None, None, None, None, None,
                         np.average(runs[b]), np.std(runs[b]), None, None, None, None, None, None, None, None])

    ## Save results in a CSV
    df = pd.DataFrame(data, columns=columns)
    votes_per_item = [len(i) for i in invert(N, M, Psi)]
    df['votes_per_worker_mean'] = np.mean(votes_per_item)
    df['votes_per_worker_std'] = np.std(votes_per_item)
    path = '../data/results/accuracy_votes_per_item.csv'
    if os.path.isfile(path):
        df_prev = pd.read_csv(path)
        df_new = df_prev.append(df, ignore_index=True)
        df_new.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


if __name__ == '__main__':
    datasets = ['faces', 'flags', 'food', 'plots']
    dataset_name = datasets[2]
    if dataset_name == 'faces':
        load_data = load_data_faces
        votes_per_item_list = [5, 'All']
    elif dataset_name == 'flags':
        load_data = load_data_flags
        votes_per_item_list = [5, 7, 9, 'All']
    elif dataset_name == 'food':
        load_data = load_data_food
        votes_per_item_list = [5, 7, 9, 'All']
    elif dataset_name == 'plots':
        load_data = load_data_plots
        votes_per_item_list = [5, 7, 9, 11, 13, 'All']
    else:
        print('Dataset not selected')
        exit(1)
    print('Dataset: {}'.format(dataset_name))

    for votes_per_item in votes_per_item_list:
        print('Votes: ', votes_per_item)
        if votes_per_item == 'All':
            accuracy(load_data, dataset_name, votes_per_item)
        else:
            Truncater = TruncaterVotesItem(votes_per_item)
            accuracy(load_data, dataset_name, votes_per_item, Truncater)

