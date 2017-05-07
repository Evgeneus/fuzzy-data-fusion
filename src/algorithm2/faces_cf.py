import pandas as pd

from em import expectation_maximization
from mv import majority_voting
from mcmc import mcmc
from f_mcmc import f_mcmc
import numpy as np

n_runs = 10


def load_data():
    M = 48
    GT_df = pd.read_csv('../../data/faces_cf/gt.csv')
    GT = dict(zip(GT_df['obj_id'].values, GT_df['GT'].values))

    f1_df = pd.read_csv('../../data/faces_cf/f1_cf.csv')
    f2_df = pd.read_csv('../../data/faces_cf/f2_cf.csv')
    s_f1 = set(f1_df['_worker_id'].values)
    s_f2 = set(f2_df['_worker_id'].values)
    sources = s_f1 | s_f2
    source_dict = dict(zip(sources, range(len(sources))))
    N = len(source_dict)

    Cl = {}
    for i in range(24):
        Cl.update({i: {'id': i, 'other': i+24}})
        Cl.update({i+24: {'id': i + 24, 'other': i}})

    Psi = [[] for obj in range(M)]
    for obj_id in range(M):
        if obj_id < 24:
            obj_data = f1_df.loc[f1_df['question_n'] == obj_id+1]
        else:
            obj_data = f2_df.loc[f2_df['question_n'] == obj_id-24+1]
        for index, row in obj_data.iterrows():
            s_id = source_dict[row['_worker_id']]
            vote = row['vote']
            if vote == "I don't know":
                continue
            Psi[obj_id].append((s_id, vote))
    return N, M, Psi, GT, Cl


def accuracy():
    N, M, Psi, GT, Cl = load_data()
    res = {'accuracy': [],
           'std': [],
           'methods': ['mv', 'em', 'mcmc']}
    runs = [[], [], [], []]
    for run in range(n_runs):
        mv_p = majority_voting(Psi)
        em_A, em_p = expectation_maximization(N, M, Psi)
        mcmc_A, mcmc_p = mcmc(N, M, Psi, {'N_iter': 10, 'burnin': 1, 'thin': 2})

        Psi_fussy, f_mcmc_G = f_mcmc(N, M, Psi, Cl, {'N_iter': 30, 'burnin': 5, 'thin': 3, 'FV': 4})

        mv_hits = []
        em_hits = []
        mcmc_hits = []
        for obj in range(M):
            if len(Psi[obj]) > 0:
                mv_hits.append(mv_p[obj][GT[obj]])
                em_hits.append(em_p[obj][GT[obj]])
                mcmc_hits.append(mcmc_p[obj][GT[obj]])

        runs[0].append(np.average(mv_hits))
        runs[1].append(np.average(em_hits))
        runs[2].append(np.average(mcmc_hits))


    print('mv: {:1.4f}+-{:1.4f}'.format(np.average(runs[0]), np.std(runs[0])))
    print('em: {:1.4f}+-{:1.4f}'.format(np.average(runs[1]), np.std(runs[1])))
    print('mcmc: {:1.4f}+-{:1.4f}'.format(np.average(runs[2]), np.std(runs[2])))

    res['accuracy'].append(np.average(runs[0]))
    res['accuracy'].append(np.average(runs[1]))
    res['accuracy'].append(np.average(runs[2]))
    res['std'].append(np.std(runs[0]))
    res['std'].append(np.std(runs[1]))
    res['std'].append(np.std(runs[2]))


    # for obj in range(M):
    #     for s, val in Psi[obj]:
    #         if obj in Cl and val == GT[Cl[obj]['other']]:
    #             print(GT[obj], f_mcmc_G[obj][s])

    #pd.DataFrame(res).to_csv(work_dir + 'face_accuracy.csv', index=False)


if __name__ == '__main__':
    accuracy()
