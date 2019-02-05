from src.algorithm.PICA.Dataset import *
from copy import deepcopy
from mv import majority_voting
from em import expectation_maximization
from f_mcmc import f_mcmc
from util import prob_binary_convert, accu_G


def load_data(file_name):
    """
        :param N: number of sources
        :param M: number of objects
        :param Psi: observations, e.g., [obj_id1: [(src11, val11), (src12, val12), ...], obj_id2: [(src21, val21), ...]]
        :param Cl: clusters where observations might be confused, e.g., {obj_id1: {"id": obj_id1, "other": confused_with_obj_id1},
                                                                         confused_with_obj_id1: {"id": confused_with_obj_id1, "other": obj_id1},
                                                                         ...}
        :return: f_mcmc_A , f_mcmc_p, f_mcmc_G where f_mcmc_A - list of source accuracies [src_accu1, src_accu2, ...]
                                                     f_mcmc_p - probability of values {obj_id: {val1: prob1, val2: prob2, ...}}
                                                     f_mcmc_G - probability of confusion for given object, source {obj_id: {src1: {0: prob1, 1: prob2}, src2: {0: prob3, 1: prob4}}}
                                                                e.g.: f_mcmc_G[o1][s1][0] - prob that s1 not confused on o1
    """
    data = init_from_file("../../data/clustering_MechanicalTurk/%d.txt" % file_name, 1, True, True)
    N = data.numLabelers  # number of sources
    M = data.numImages  # number of objects
    Psi = [[] for _ in range(M)]
    for v_ in data.labels:
        Psi[v_.imageIdx].append((v_.labelerId, v_.label))
    # Cl = {0: {'other': 1, 'id': 0},
    #       1: {'other': 0, 'id': 1},
    #       2: {'other': 3, 'id': 2},
    #       3: {'other': 2, 'id': 3},
    #       4: {'other': 5, 'id': 4},
    #       5: {'other': 4, 'id': 5},
    #       }
    # Cl = {0: {'other': 5, 'id': 0},
    #       1: {'other': 4, 'id': 1},
    #       2: {'other': 3, 'id': 2},
    #       3: {'other': 2, 'id': 3},
    #       4: {'other': 1, 'id': 4},
    #       5: {'other': 0, 'id': 5},
    #       }
    Cl = {0: {'other': 4, 'id': 0},
          1: {'other': 3, 'id': 1},
          2: {'other': 5, 'id': 2},
          3: {'other': 1, 'id': 3},
          4: {'other': 0, 'id': 4},
          5: {'other': 2, 'id': 5},
          }

    return N, M, Psi, Cl, data.gt

def permutedAcc(bin_output, num_possible_values, gt):
    # Generate list of permutations of possible_values
    permutations = list(itertools.permutations(range(num_possible_values)))

    acc = 0
    best_perm = 0

    for perm in permutations:
        # Compute votes
        labels = [None]*len(bin_output)
        for obj_id, obj in enumerate(bin_output):
            labels[obj_id] = perm[max(obj, key=obj.get)]

        # Compute accuracy
        correct = 0.
        total = 0.
        for i in range(len(bin_output)):
            if labels[i] == gt[i]:
                correct += 1
            total += 1
        new_acc = correct / total

        if new_acc > acc:
            acc = new_acc
            # best_perm = perm
    # print "Transformation of labels: " + str(best_perm)
    return acc


def run(N, M, Psi, Cl, num_possible_values, gt):
    # MV
    # mv_p = majority_voting(Psi)
    # mv_b = prob_binary_convert(mv_p)

    # Fussy MCMC
    mcmc_params = {'N_iter': 40, 'burnin': 5, 'thin': 3, 'FV': 4}
    f_mcmc_G, Psi_fussy = f_mcmc(N, M, deepcopy(Psi), deepcopy(Cl), mcmc_params)



    # # MV + CONF
    # mv_f_p = majority_voting(Psi_fussy)
    # mv_f_b = prob_binary_convert(mv_f_p)

    # EM + CONF
    em_A, em_p = expectation_maximization(N, M, Psi_fussy)
    em_f_b = prob_binary_convert(em_p)


    acc = permutedAcc(em_f_b, num_possible_values, gt)

    return acc


if __name__=='__main__':
  acc_list = []
  for file_name in range(100):  # Run 100 simulations
    N, M, Psi, Cl, gt = load_data(file_name)
    num_possible_values = 3
    # if fuzzy mcmc do a crucial wrong swamp
    try:
        acc = run(N, M, deepcopy(Psi), Cl, num_possible_values, gt)
    except:
        acc = run(N, M, deepcopy(Psi), Cl, num_possible_values, gt)


    acc_list.append(acc)
  print(sum(acc_list)/len(acc_list))
