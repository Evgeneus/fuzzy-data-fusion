import pandas as pd
import numpy as np
from collections import Counter


def compute_clusters(c_Psi):
    M = len(c_Psi)
    mv_res = [None for _ in range(M)]
    conf_label = [None for _ in range(M)]  # array with potential confused labels
    # prepare data
    for obj, votes in enumerate(c_Psi):
        votes = zip(*votes)[1]
        counter = Counter(votes)
        most_common2 = counter.most_common(2)
        # print(counter.most_common(3))
        mv_res[obj] = most_common2[0][0]
        if len(most_common2) >= 2:
            conf_label[obj] = most_common2[1][0]
        else:
            conf_label[obj] = None

    mv_res = np.array(mv_res)
    conf_label = np.array(conf_label)
    # compute clusters
    Cl = {}
    for obj in range(M):
        other_label = conf_label[obj]
        if other_label == None:
            # print("0")
            continue
        other_obj_candidates = np.where(mv_res == other_label)[0]
        # other_obj_candidates = other_obj_candidates[conf_label[other_obj_candidates] == obj]
        if len(other_obj_candidates) == 0:
            # print ('!')
            continue
        other_obj = np.random.choice(other_obj_candidates, 1, replace=False)[0]
        Cl[obj] = {'id': obj, 'other': other_obj}

    return Cl


def load_data_faces(truncater=None):
    test = [[], []]
    M = 48
    GT_df = pd.read_csv('../data/faces_crowdflower/gt.csv')
    GT = dict(zip(GT_df['obj_id'].values, GT_df['GT'].values))
    f1_df = pd.read_csv('../data/faces_crowdflower/f1_cf.csv')
    f2_df = pd.read_csv('../data/faces_crowdflower/f2_cf.csv')
    ## remove rows with "I don't know" votes
    f1_df = f1_df[f1_df['vote'] != "I don't know"]
    f2_df = f2_df[f2_df['vote'] != "I don't know"]
    if truncater is not None:
        f1_df = truncater.do_trancate(f1_df)
        f2_df = truncater.do_trancate(f2_df)
    s_f1 = set(f1_df['_worker_id'].values)
    s_f2 = set(f2_df['_worker_id'].values)
    sources = s_f1 | s_f2
    source_dict = dict(zip(sources, range(len(sources))))
    N = len(source_dict)

    Cl = {}
    for i in range(24):
        Cl.update({i: {'id': i, 'other': i+24}})
        Cl.update({i+24: {'id': i + 24, 'other': i}})

    GT_G = {}
    for obj in range(M):
        GT_G[obj] = {}

    conf_counter = 0
    total_votes = 0
    Psi = [[] for _ in range(M)]
    for obj_id in range(M):
        if obj_id < 24:
            obj_data = f1_df.loc[f1_df['question_n'] == obj_id+1]
        else:
            obj_data = f2_df.loc[f2_df['question_n'] == obj_id-24+1]
        for index, row in obj_data.iterrows():
            s_id = source_dict[row['_worker_id']]
            vote = row['vote']
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

    # print '#confusions: {}, {:1.1f}%'.format(conf_counter, conf_counter*100./total_votes)
    # print '#total votes: {}'.format(total_votes)
    # Cl = compute_clusters(Psi)

    return [N, M, Psi, GT, Cl, GT_G]


def load_data_flags(truncater=None):
    test = [[], []]
    f1_df = pd.read_csv('../data/Flags/flags1_res.csv', delimiter=';')
    f2_df = pd.read_csv('../data/Flags/flags2_res.csv', delimiter=';')
    f3_df = pd.read_csv('../data/Flags/flags3_res.csv', delimiter=';')
    ## remove rows with "I don't know" votes
    f1_df = f1_df[f1_df['crowd_ans'] != "I don't know"]
    f2_df = f2_df[f2_df['crowd_ans'] != "I don't know"]
    f3_df = f3_df[f3_df['crowd_ans'] != "I don't know"]
    if truncater is not None:
        f1_df = truncater.do_trancate(f1_df)
        f2_df = truncater.do_trancate(f2_df)
        f3_df = truncater.do_trancate(f3_df)
    s_f1 = set(f1_df['_worker_id'].values)
    s_f2 = set(f2_df['_worker_id'].values)
    s_f3 = set(f3_df['_worker_id'].values)
    sources = s_f1 | s_f2 | s_f3
    source_dict = dict(zip(sources, range(len(sources))))
    N = len(source_dict)  # number of sources
    M = 60  # number of objects df1+df2
    M2 = 40  # # number of objects df3

    Cl, GT = {}, {}
    for i in range(M / 2):
        GT[i] = f1_df[f1_df['question_n'] == i+1]['gt'].values[0]
        GT[i + M/2] = f2_df[f2_df['question_n'] == i+1]['gt'].values[0]
        Cl.update({i: {'id': i, 'other': i + M/2}})
        Cl.update({i + M/2: {'id': i + M/2, 'other': i}})
    for i in range(M, M + M2/2):
        GT[i] = f3_df[f3_df['question_n'] == i - M + 1]['gt'].values[0]
        GT[i + M2/2] = f3_df[f3_df['question_n'] == i - M + M2/2 + 1]['gt'].values[0]
        Cl.update({i: {'id': i, 'other': i + M2/2}})
        Cl.update({i + M2/2: {'id': i + M2/2, 'other': i}})

    GT_G = {}
    for obj in range(M + M2):
        GT_G[obj] = {}

    conf_counter = 0
    total_votes = 0
    Psi = [[] for _ in range(M+M2)]
    for obj_id in range(M+M2):
        if obj_id < M/2:
            obj_data = f1_df.loc[f1_df['question_n'] == obj_id+1]
        elif obj_id >= M/2 and obj_id < M:
            obj_data = f2_df.loc[f2_df['question_n'] == obj_id-M/2+1]
        else:
            obj_data = f3_df.loc[f3_df['question_n'] == obj_id-M+1]
        for index, row in obj_data.iterrows():
            s_id = source_dict[row['_worker_id']]
            vote = row['crowd_ans']
            Psi[obj_id].append((s_id, vote))

            other_id = Cl[obj_id]['other']
            other_GT = GT[other_id]
            if vote == other_GT:
                GT_G[obj_id][s_id] = 0
                conf_counter += 1
                test[0].append(obj_id)
                test[1].append(other_id)
            else:
                GT_G[obj_id][s_id] = 1
            total_votes += 1

    # print '#confusions: {}, {:1.1f}%'.format(conf_counter, conf_counter*100./total_votes)
    # print '#total votes: {}'.format(total_votes)
    # Cl = compute_clusters(Psi)
    return [N, M+M2, Psi, GT, Cl, GT_G]


def load_data_food(truncater=None):
    test = [[], []]
    f1_df = pd.read_csv('../data/Food/food1_res.csv', delimiter=';')
    f2_df = pd.read_csv('../data/Food/food2_res.csv', delimiter=';')
    f3_df = pd.read_csv('../data/Food/food3_res.csv', delimiter=';')
    ## remove rows with "I don't know" votes
    f1_df = f1_df[f1_df['crowd_ans'] != "I don't know"]
    f2_df = f2_df[f2_df['crowd_ans'] != "I don't know"]
    f3_df = f3_df[f3_df['crowd_ans'] != "I don't know"]
    if truncater is not None:
        f1_df = truncater.do_trancate(f1_df)
        f2_df = truncater.do_trancate(f2_df)
        f3_df = truncater.do_trancate(f3_df)

    s_f1 = set(f1_df['_worker_id'].values)
    s_f2 = set(f2_df['_worker_id'].values)
    s_f3 = set(f3_df['_worker_id'].values)
    sources = s_f1 | s_f2 | s_f3
    source_dict = dict(zip(sources, range(len(sources))))
    N = len(source_dict)  # number of sources
    M = 20  # number of objects 10+10
    M2 = 56  # number of objects df3: 56

    Cl, GT = {}, {}
    ## df1, df2
    for i in range(M / 2):
        GT[i] = f1_df[f1_df['question_n'] == i+1]['gt'].values[0]
        GT[i + M/2] = f2_df[f2_df['question_n'] == i+1]['gt'].values[0]
        Cl.update({i: {'id': i, 'other': i + M/2}})
        Cl.update({i + M/2: {'id': i + M/2, 'other': i}})
    ## df3
    for i in range(M, M + M2 / 2 / 2):
        GT[i] = f3_df[f3_df['question_n'] == i-M+1]['gt'].values[0]
        GT[i + M2/2/2] = f3_df[f3_df['question_n'] == i-M+1+M2/2/2]['gt'].values[0]
        Cl.update({i: {'id': i, 'other': i + M2/2/2}})
        Cl.update({i + M2/2/2: {'id': i + M2/2/2, 'other': i}})
    for i in range(M + M2 / 2, M + 3*M2/4):
        GT[i] = f3_df[f3_df['question_n'] == i-M+1]['gt'].values[0]
        GT[i + M2/2/2] = f3_df[f3_df['question_n'] == i-M+1+M2/2/2]['gt'].values[0]
        Cl.update({i: {'id': i, 'other': i + M2/2/2}})
        Cl.update({i + M2/2/2: {'id': i + M2/2/2, 'other': i}})

    GT_G = {}
    for obj in range(M+M2):
        GT_G[obj] = {}

    conf_counter = 0
    total_votes = 0
    Psi = [[] for _ in range(M+M2)]
    for obj_id in range(M+M2):
        if obj_id < M/2:
            obj_data = f1_df.loc[f1_df['question_n'] == obj_id+1]
        elif obj_id >= M/2 and obj_id < M:
            obj_data = f2_df.loc[f2_df['question_n'] == obj_id-M/2+1]
        else:
            obj_data = f3_df.loc[f3_df['question_n'] == obj_id-M+1]

        for index, row in obj_data.iterrows():
            s_id = source_dict[row['_worker_id']]
            vote = row['crowd_ans']
            Psi[obj_id].append((s_id, vote))

            other_id = Cl[obj_id]['other']
            other_GT = GT[other_id]
            if vote == other_GT:
                GT_G[obj_id][s_id] = 0
                conf_counter += 1
                test[0].append(obj_id)
                test[1].append(other_id)
            else:
                GT_G[obj_id][s_id] = 1
            total_votes += 1
    # print '#confusions: {}, {:1.1f}%'.format(conf_counter, conf_counter*100./total_votes)
    # print '#total votes: {}'.format(total_votes)
    # Cl = compute_clusters(Psi)
    return [N, M+M2, Psi, GT, Cl, GT_G]


def load_data_plots(truncater=None):
    test = [[], []]
    f1_df = pd.read_csv('../data/Plots/plots1_res.csv', delimiter=';')
    f2_df = pd.read_csv('../data/Plots/plots2_res.csv', delimiter=';')
    ## remove rows with "I don't know" votes
    f1_df = f1_df[f1_df['crowd_ans'] != "I don't know"]
    f2_df = f2_df[f2_df['crowd_ans'] != "I don't know"]
    if truncater is not None:
        f1_df = truncater.do_trancate(f1_df)
        f2_df = truncater.do_trancate(f2_df)
    s_f1 = set(f1_df['_worker_id'].values)
    s_f2 = set(f2_df['_worker_id'].values)
    sources = s_f1 | s_f2
    source_dict = dict(zip(sources, range(len(sources))))
    N = len(source_dict)  # number of sources
    M = 100  # number of objects

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
    # print '#confusions: {}, {:1.1f}%'.format(conf_counter, conf_counter*100./total_votes)
    # print '#total votes: {}'.format(total_votes)
    # Cl = compute_clusters(Psi)
    return [N, M, Psi, GT, Cl, GT_G]


def load_gt_conf_ranks(df1, df2):
    df1.dropna(subset=['gt_conf'], how='all', inplace=True)
    df2.dropna(subset=['gt_conf'], how='all', inplace=True)
    df1['conf_score'] = np.log(df1['num_votes']) * df1['num_conf'] / (df1['num_votes'])
    df1['clusters'] = df1[['gt', 'gt_conf']].apply(lambda x: '-'.join(x), axis=1)
    df2['conf_score'] = np.log(df2['num_votes']) * df2['num_conf'] / (df2['num_votes'])
    df2['clusters'] = df2[['gt', 'gt_conf']].apply(lambda x: '-'.join(x), axis=1)
    df1 = df1[df1['conf_score'] > 0.]
    df2 = df2[df2['conf_score'] > 0.]
    gt_conf_ranks = np.vstack((df1[['clusters', 'conf_score']].values, df2[['clusters', 'conf_score']].values))
    gt_conf_ranks = np.array(sorted(gt_conf_ranks, key=lambda x: x[1], reverse=True))
    return gt_conf_ranks


def load_gt_conf_ranks_faces():
    M = 48
    GT_df = pd.read_csv('../data/faces_crowdflower/gt.csv')
    GT = dict(zip(GT_df['obj_id'].values, GT_df['GT'].values))
    f1_df = pd.read_csv('../data/faces_crowdflower/f1_cf.csv')
    f2_df = pd.read_csv('../data/faces_crowdflower/f2_cf.csv')
    ## remove rows with "I don't know" votes
    f1_df = f1_df[f1_df['vote'] != "I don't know"]
    f2_df = f2_df[f2_df['vote'] != "I don't know"]
    Cl = {}
    for i in range(24):
        Cl.update({i: {'id': i, 'other': i+24}})
        Cl.update({i+24: {'id': i + 24, 'other': i}})

    nPi = [[0., 0.] for _ in range(M)]
    for obj_id in range(M):
        if obj_id < 24:
            obj_data = f1_df.loc[f1_df['question_n'] == obj_id+1]
        else:
            obj_data = f2_df.loc[f2_df['question_n'] == obj_id-24+1]
        other_id = Cl[obj_id]['other']
        other_GT = GT[other_id]
        for index, row in obj_data.iterrows():
            vote = row['vote']
            if vote == other_GT:
                nPi[obj_id][1] += 1  ## means a confusion
            else:
                nPi[obj_id][0] += 1  ## not a confusion
    Cl_conf_scores = [np.log((i + j)) * float(j) / (i + j) for i, j in nPi]
    gt_conf_ranks = []
    for obj_id, conf_score in enumerate(Cl_conf_scores):
        if conf_score == 0.:
            continue
        gt_conf_ranks.append([GT[obj_id] + '-' + GT[Cl[obj_id]['other']], conf_score])
    gt_conf_ranks = np.array(sorted(gt_conf_ranks, key=lambda x: x[1], reverse=True))
    return gt_conf_ranks


class TruncaterVotesItem:
    def __init__(self, votes_per_item):
        self.votes_per_item = votes_per_item

    def do_trancate(self, df):
        df_new = pd.DataFrame()
        for item_id in df['question_n'].unique():
            try:
                df_new = df_new.append(df.loc[df['question_n'] == item_id].sample(self.votes_per_item), ignore_index=True)
            except:
                df_new = df_new.append(df.loc[df['question_n'] == item_id], ignore_index=True)
                print('Warning: not enough votes per item!')
        return df_new


# def truncate_votes_per_worker(df, votes_per_worker):
#     df_new = pd.DataFrame()
#     for worker_id in df['_worker_id'].unique():
#         df_new = df_new.append(df.loc[df['_worker_id'] == worker_id].sample(votes_per_worker), ignore_index=True)
#     return df_new
