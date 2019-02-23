import pandas as pd
import numpy as np
from copy import deepcopy
from src.algorithm.em import expectation_maximization
from src.algorithm.mv import majority_voting
from src.algorithm.mcmc import mcmc
from src.algorithm.f_mcmc import f_mcmc
from src.algorithm.util import accu_G, prob_binary_convert
from src.algorithm.sums import sums
from src.algorithm.average_log import average_log
from src.algorithm.investment import investment
from src.algorithm.pooled_investment import pooled_investment
from src.algorithm.synthetic_experiment import adapter_input, adapter_output

n_runs = 100


def load_data():
    test = [[], []]
    f1_df = pd.read_csv('../data/Flags/flags1_res.csv', delimiter=';')
    f2_df = pd.read_csv('../data/Flags/flags2_res.csv', delimiter=';')
    s_f1 = set(f1_df['_worker_id'].values)
    s_f2 = set(f2_df['_worker_id'].values)
    sources = s_f1 | s_f2
    source_dict = dict(zip(sources, range(len(sources))))
    N = len(source_dict)  # number of sources
    M = 60  # number of objects

    Cl, GT = {}, {}
    for i in range(M / 2):
        GT[i] = f1_df[f1_df['question_n'] == i+1]['gt'].values[0]
        GT[i + M/2] = f2_df[f2_df['question_n'] == i+1]['gt'].values[0]
        Cl.update({i: {'id': i, 'other': i + M/2}})
        Cl.update({i + M/2: {'id': i + M/2, 'other': i}})

    GT_G = {}
    for obj in range(M):
        GT_G[obj] = {}

    conf_counter = 0
    total_votes = 0
    Psi = [[] for _ in range(M)]
    # 13 number of clusters where confusions likely to happen
    for obj_id in range(M):
        if obj_id < M/2:
            obj_data = f1_df.loc[f1_df['question_n'] == obj_id+1]
        else:
            obj_data = f2_df.loc[f2_df['question_n'] == obj_id-M/2+1]
        for index, row in obj_data.iterrows():
            s_id = source_dict[row['_worker_id']]
            vote = row['crowd_ans']
            Psi[obj_id].append((s_id, vote))

            other_id = Cl[obj_id]['other']
            other_GT = GT[other_id]
            if vote == other_GT:
                GT_G[obj_id][s_id] = 0
                conf_counter += 1
                # print 'obj: {}, other: {}'.format(obj_id, other_id)
                test[0].append(obj_id)
                test[1].append(other_id)
            else:
                GT_G[obj_id][s_id] = 1
            total_votes += 1

    num_votes_per_object = 20
    print '#confusions: {}, {:1.1f}%'.format(conf_counter, conf_counter*100./(num_votes_per_object*26))
    print '#total votes: {}'.format(total_votes)
    return [N, M, Psi, GT, Cl, GT_G]


def accuracy():
    N, M, Psi, GT, Cl, GT_G = load_data()
    res = {'accuracy': [],
           'std': [],
           'methods': ['mv_p', 'em_p', 'mcmc_p',
                       'mv_f_p', 'em_f_p', 'mcmc_f_p',
                       'mv_b', 'em_b', 'mcmc_b',
                       'mv_f_b', 'em_f_b', 'mcmc_f_b',
                       'sums', 'avlog', 'inv', 'pinv',
                       'sums_f', 'avlog_f', 'inv_f', 'pinv_f'
                       ]}
    runs = [[] for _ in range(20)]
    accu_G_list = []
    mcmc_params = {'N_iter': 10, 'burnin': 1, 'thin': 2}
    for run in range(n_runs):
        # PROBABILISTIC OUTPUT
        mv_p = majority_voting(Psi)
        em_A, em_p = expectation_maximization(N, M, Psi)
        mcmc_A, mcmc_p = mcmc(N, M, Psi, mcmc_params)

        f_mcmc_G, Psi_fussy = f_mcmc(N, M, deepcopy(Psi), Cl, {'N_iter': 30, 'burnin': 5, 'thin': 3, 'FV': 4})


        # compute G ACCuracy only on clusters where we belief confusions might happen
        f_mcmc_G_clust, GT_G_clust = {}, {}
        num_of_clusters = 13
        for obj_id in range(num_of_clusters):
            f_mcmc_G_clust[obj_id] = f_mcmc_G[obj_id]
            f_mcmc_G_clust[obj_id + M/2] = f_mcmc_G[obj_id + M/2]
            GT_G_clust[obj_id] = GT_G[obj_id]
            GT_G_clust[obj_id + M/2] = GT_G[obj_id + M/2]
        accu_G_list.append(accu_G(f_mcmc_G_clust, GT_G_clust))

        # accu_G_list.append(accu_G(f_mcmc_G, GT_G))

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

        # slect objects with conflicting votes among ones are in clusters
        obj_with_conflicts = []
        for obj_id, obj in enumerate(mv_p):
            if len(obj) > 1 and (obj_id < num_of_clusters or M/2 <= obj_id < M/2+num_of_clusters):
                obj_with_conflicts.append(obj_id)

        for obj in range(M):
            if obj in obj_with_conflicts:
                # PROBABILISTIC OUTPUT
                mv_hits.append(mv_p[obj][GT[obj]])
                em_hits.append(em_p[obj][GT[obj]])
                mcmc_hits.append(mcmc_p[obj][GT[obj]])

                mv_f_hits.append(mv_f_p[obj][GT[obj]])
                em_f_hits.append(em_f_p[obj][GT[obj]])
                mcmc_f_hits.append(mcmc_f_p[obj][GT[obj]])

                # BINARY OUTPUT
                mv_b_hits.append(mv_b[obj][GT[obj]])
                em_b_hits.append(em_b[obj][GT[obj]])
                mcmc_b_hits.append(mcmc_b[obj][GT[obj]])
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

    print('G Accu: {:1.4f}+-{:1.4f}'.format(np.average(accu_G_list), np.std(accu_G_list)))
    print 'PROBABILISTIC OUTPUT'
    print('mv: {:1.4f}+-{:1.4f}'.format(np.average(runs[0]), np.std(runs[0])))
    print('mv_f: {:1.4f}+-{:1.4f}'.format(np.average(runs[3]), np.std(runs[3])))
    print('em: {:1.4f}+-{:1.4f}'.format(np.average(runs[1]), np.std(runs[1])))
    print('em_f: {:1.4f}+-{:1.4f}'.format(np.average(runs[4]), np.std(runs[4])))
    print('mcmc: {:1.4f}+-{:1.4f}'.format(np.average(runs[2]), np.std(runs[2])))
    print('mcmc_f: {:1.4f}+-{:1.4f}'.format(np.average(runs[5]), np.std(runs[5])))
    print 'BINARY OUTPUT'
    print('mv: {:1.4f}+-{:1.4f}'.format(np.average(runs[6]), np.std(runs[6])))
    print('mv_f: {:1.4f}+-{:1.4f}'.format(np.average(runs[9]), np.std(runs[9])))
    print('em: {:1.4f}+-{:1.4f}'.format(np.average(runs[7]), np.std(runs[7])))
    print('em_f: {:1.4f}+-{:1.4f}'.format(np.average(runs[10]), np.std(runs[10])))
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
        res['std'].append(np.std(run))

    # Save results in a CSV
    pd.DataFrame(res).to_csv('../data/results/flags_accuracy.csv', index=False)


if __name__ == '__main__':
    # if fuzzy mcmc do a crucial wrong swamp
    try:
        accuracy()
    except:
        print('empty..')
        accuracy()
