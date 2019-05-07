import math
import numpy as np
from collections import defaultdict
from src.algorithm.PICA.Dataset import Label, Dataset
from random import randrange


def log_likelihood(GT, M, Psi, A, p):
    """
    Computes the log likelihood of the Psi using A and p.
    """
    res = 0
    for obj_id in range(M):
        for source_id, value_id in Psi[obj_id]:
            if p[obj_id][value_id] == 1.0:
                p[obj_id][value_id] = 0.9999999
            if p[obj_id][value_id] == 0.0:
                p[obj_id][value_id] = 0.0000001
            if value_id == GT[obj_id]:
                res += math.log(A[source_id] * p[obj_id][value_id])
            else:
                res += math.log((1 - A[source_id]) * (1 - p[obj_id][value_id]))
    return res


def invert(N, M, Psi):
    """
    Inverts the observation matrix. Need for performance reasons.
    :param N:
    :param M:
    :param Psi:
    :return:
    """
    inv_Psi = [[] for s in range(N)]
    for obj in range(M):
        for s, val in Psi[obj]:
            inv_Psi[s].append((obj, val))
    return inv_Psi


def fPsi(N, M, Psi, G, Cl):
    """
    Computes the observation matrix based on known confusions.
    :param N:
    :param M:
    :param Psi:
    :param G: confusions
    :param Cl: clusters
    :return:
    """
    f_Psi = [[] for x in range(M)]
    for obj in range(M):
        for s, val in Psi[obj]:
            if obj in Cl:
                if G[obj][s] == 1:
                    f_Psi[obj].append((s, val))
                else:
                    f_Psi[Cl[obj]['other']].append((s, val))
            else:
                f_Psi[obj].append((s, val))
    return f_Psi, invert(N, M, f_Psi)


def accu_G(f_mcmc_G, GT_G):
    tp = 0.0
    total = 0.0
    for obj in GT_G.keys():
        for s in GT_G[obj]:
            tp += f_mcmc_G[obj][s][GT_G[obj][s]]
            total += 1

    return tp/total


def precision_recall(f_mcmc_G, GT_G):
    # '0' IS POSITIVE, CONFUSION
    tp = tn = fp = fn = 0.
    for obj in GT_G.keys():
        for s in GT_G[obj]:
            gt = GT_G[obj][s]
            val = f_mcmc_G[obj][s].index(max(f_mcmc_G[obj][s]))
            gt = 1 - gt
            val = 1 - val
            if gt and not val:
                fn += 1
            if not gt and val:
                fp += 1
            if gt and val:
                tp += 1
            if not gt and not val:
                tn += 1
    try:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        print('ZeroDivisionError -> recall, precision, fbeta = 0., 0., 0')
        recall = precision = 0.
    return precision, recall, accuracy


def prob_binary_convert(data):
    data_b = []
    for obj in data:
        sources = obj.keys()
        probs = obj.values()
        index_max = probs.index(max(probs))
        binary = [0.] * len(probs)
        binary[index_max] = 1.
        data_b.append(defaultdict(int, zip(sources, binary)))
    return data_b


## Dawid and Skene
def adapter_psi_dawid(Psi):
    Psi_dawid = {}
    for item_id in range(len(Psi)):
        Psi_dawid[item_id] = {}
        for worker_id, val in Psi[item_id]:
            Psi_dawid[item_id][worker_id] = [val]
    return Psi_dawid


def adapter_prob_dawid(values_prob, classes):
    ds_p = []
    for item_prob in values_prob:
        ids = np.nonzero(item_prob)[0]
        d = defaultdict(int)
        for id in ids:
            d[classes[id]] = item_prob[id]
        ds_p.append(d)
    return ds_p


def get_ds_G(ErrM, classes, ds_p, Psi):
    M = sum(ErrM)
    M = M / M.sum(axis=1)[:, np.newaxis]
    ds_classes = [max(stats, key=stats.get) for stats in ds_p]
    ds_G = {}
    for obj_id, data in enumerate(Psi):
        ds_G[obj_id] = {}
        for s_id, val in data:
            ds_class = ds_classes[obj_id]
            ds_class_id = classes.index(ds_class)
            val_id = classes.index(val)
            if (ds_class_id != val_id) and M[ds_class_id][val_id] >= 0.5:
                ds_G[obj_id][s_id] = [1, 0]
            else:
                ds_G[obj_id][s_id] = [0, 1]
    return ds_G


def do_conf_ranks_ds(ErrM, classes):
    M = sum(ErrM)
    M = M / M.sum(axis=1)[:, np.newaxis]
    conf_classes = []
    for class_id, data in enumerate(M):
        data[class_id] = 0.
        argmax_id = data.argmax()
        if data[argmax_id] == 0:
            continue
        conf_classes.append(np.array([classes[class_id] + '-' + classes[argmax_id], data[argmax_id]]))
    conf_ranks = np.array(sorted(conf_classes, key=lambda x: x[1], reverse=True))
    return conf_ranks


def do_conf_ranks_fmcmc(Cl_conf_scores, M, GT, Cl):
    conf_classes = []
    for obj_id in range(M):
        conf_classes.append(np.array([GT[obj_id] + '-' + GT[Cl[obj_id]['other']], Cl_conf_scores[obj_id]]))
    conf_ranks = np.array(sorted(conf_classes, key=lambda x: x[1], reverse=True))
    return conf_ranks


def conf_ranks_acc_pr_rec(gt_conf_ranks, conf_ranks):
    pass


def adapter_psi_pica(numLabelers, numImages, Psi, GT, gamma=1, isDSM=True):
    if isDSM:
        from src.algorithm.PICA.SinkProp.Labeler import Labeler
    else:
        from src.algorithm.PICA.Stochastic.Labeler import Labeler

    alphabet = set()
    for obj_data in Psi:
        alphabet |= set(zip(*obj_data)[1])
    alphabet = list(alphabet)
    numCharacters = len(alphabet)
    label_labelID_map = dict(zip(alphabet, range(numCharacters)))
    labels = []
    for obj_id, obj_data in enumerate(Psi):
        for s_id, val in obj_data:
            labels.append(Label(obj_id, s_id, label_labelID_map[val]))
    numLabels = len(labels)
    probZ = np.zeros((numCharacters, numImages))
    priorA = np.identity(numCharacters)
    Labelers = [Labeler(priorA) for i in range(numLabelers)]

    ## Z priors
    priorZ = np.empty((numCharacters, numImages))
    for x in range(numCharacters):
        priorZ[x][:] = 1. / numCharacters

    ## transform ground truth labels
    hasGT = True
    gt = []
    for obj_id in range(numImages):
        gt.append(label_labelID_map[GT[obj_id]])  # Only store label

    # Initialize Dataset object
    return Dataset(numLabels, numLabelers, numImages, numCharacters, gamma,
                   alphabet, priorZ, labels, probZ, Labelers, hasGT, gt, isDSM)


def make_random_clusters(M):
    Cl = {}
    obj_ids = list(range(M))
    for obj_id in range(M):
        rand_obj_id = obj_id
        while rand_obj_id == obj_id:
            rand_id = randrange(len(obj_ids))
            rand_obj_id = obj_ids[rand_id]
        rand_obj_id = obj_ids.pop(rand_id)
        Cl[obj_id] = {'id': obj_id, 'other': rand_obj_id}
    return Cl
