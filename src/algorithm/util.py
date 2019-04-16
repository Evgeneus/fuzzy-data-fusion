import math
from collections import defaultdict

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
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
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
