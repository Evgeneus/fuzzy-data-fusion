import pandas as pd

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
    return GT, Psi


if __name__ == '__main__':
    load_data()