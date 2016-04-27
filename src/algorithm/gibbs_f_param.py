import numpy as np
import random
import copy
from scipy.stats import beta

max_rounds = 100
eps = 10e-5
possible_values = [0, 1]
beta1, beta2 = 10, 10
std = 0.01


def get_hparam(m, std):
    param1 = ((1-m)/std**2 - 1/m)*m**2
    param2 = param1*(1/m - 1)

    return [param1, param2]


def init_var(data):
    s_ind = sorted(data.S.drop_duplicates())
    obj_index_list = sorted(data.O.drop_duplicates())
    var_index = [obj_index_list, s_ind]
    alpha1, alpha2 = get_hparam(m=0.7, std=std)
    accuracy_list = beta.rvs(alpha1, alpha2, size=len(s_ind))
    gamma1, gamma2 = get_hparam(m=g_true, std=std)
    pi_init = beta.rvs(gamma1, gamma2, size=1)
    g_values = np.random.binomial(1, pi_init, len(data))
    obj_values = beta.rvs(beta1, beta2, size=len(ground_truth))
    pi_prob = [0.5]*(len(obj_index_list)/2)

    init_prob = []
    l = len(possible_values)
    for obj_index in obj_index_list:
        init_prob.append([1./l]*l)

    counts = []
    for s in s_ind:
        psi_ind_list = []
        counts_list = []
        for psi in data[data.S == s].iterrows():
            psi_ind = psi[0]
            psi_ind_list.append(psi_ind)
            psi = psi[1]
            if g_values[psi_ind] == 1:
                if psi.V == obj_values[psi.O]:
                    counts_list.append(1)
                else:
                    counts_list.append(0)
            else:
                if psi.O % 2 == 0:
                    if psi.V == obj_values[psi.O+1]:
                        counts_list.append(1)
                    else:
                        counts_list.append(0)
                else:
                    if psi.V == obj_values[psi.O-1]:
                        counts_list.append(1)
                    else:
                        counts_list.append(0)
        counts.append(pd.DataFrame(counts_list, columns=['c']).set_index([psi_ind_list]))

    return [var_index, g_values, obj_values, counts, init_prob, pi_prob, accuracy_list,
            alpha1, alpha2, gamma1, gamma2]


def get_o(o_ind, g_values, obj_values, counts, data, prob, accuracy_list):
    l_p = []
    l_c = []
    if o_ind % 2 == 0:
        cl = [o_ind, o_ind+1]
    else:
        cl = [o_ind-1, o_ind]
    psi_cl = data[data.O.isin(cl)]
    s_in_cluster = list(psi_cl.S.drop_duplicates())
    for v in possible_values:
        counts_v = copy.deepcopy(counts)
        l_p.append(prob[o_ind][v])
        for s in s_in_cluster:
            n_p, n_m = 0, 0
            for psi in psi_cl[psi_cl.S == s].iterrows():
                psi_ind = psi[0]
                psi = psi[1]
                c_old = counts_v[s].at[psi_ind, 'c']
                if g_values[psi_ind] == 1 and psi.O == o_ind:
                    c_new = 1 if psi.V == v else 0
                    if c_new != c_old:
                        counts_v[s].at[psi_ind, 'c'] = c_new
                    if c_new == 1:
                        n_p += 1
                    else:
                        n_m += 1
                elif psi.O != o_ind and g_values[psi_ind] == 0:
                    c_new = 1 if psi.V == v else 0
                    if c_new != c_old:
                        counts_v[s].at[psi_ind, 'c'] = c_new
                    if c_new == 1:
                        n_p += 1
                    else:
                        n_m += 1
            s_counts = [n_m, n_p]
            if any(s_counts):
                accuracy = accuracy_list[s]
                l_p[v] *= accuracy**n_p*(1-accuracy)**n_m
        l_c.append(counts_v)
    norm_const = sum(l_p)
    l_p[0] /= norm_const
    l_p[1] /= norm_const
    v_new = np.random.binomial(1, l_p[1], 1)[0]
    counts_new = l_c[v_new]

    return [v_new, counts_new, l_p]


def get_g(g_ind, g_prev, pi_prob, obj_values, accuracy_list, counts, data):
    l_p = []
    psi = data.iloc[g_ind]
    s = psi.S
    accuracy = accuracy_list[s]
    cluster = psi.O/2
    for g in possible_values:
        pr_pi = pi_prob[cluster]**g*(1-pi_prob[cluster])**(1-g)
        if g == 1:
            if psi.V == obj_values[psi.O]:
                pr_pi *= accuracy
            else:
                pr_pi *= 1-accuracy
        else:
            if psi.O % 2 == 0:
                if psi.V == obj_values[psi.O+1]:
                    pr_pi *= accuracy
                else:
                    pr_pi *= 1-accuracy
            else:
                if psi.V == obj_values[psi.O-1]:
                    pr_pi *= accuracy
                else:
                    pr_pi *= 1-accuracy
        l_p.append(pr_pi)
    norm_const = sum(l_p)
    l_p[0] /= norm_const
    l_p[1] /= norm_const
    g_new = np.random.binomial(1, l_p[1], 1)[0]
    if g_new != g_prev:
        if g_new == 1:
            if psi.V == obj_values[psi.O]:
                counts[s].at[g_ind, 'c'] = 1
            else:
                counts[s].at[g_ind, 'c'] = 0
        else:
            if psi.O % 2 == 0:
                if psi.V == obj_values[psi.O+1]:
                    counts[s].at[g_ind, 'c'] = 1
                else:
                    counts[s].at[g_ind, 'c'] = 0
            else:
                if psi.V == obj_values[psi.O-1]:
                    counts[s].at[g_ind, 'c'] = 1
                else:
                    counts[s].at[g_ind, 'c'] = 0

    return g_new


def get_pi(cl_ind, g_values, data, gamma1, gamma2):
    cl = [cl_ind*2, cl_ind*2+1]
    psi_cl_list = list(data[data.O.isin(cl)].index)
    count_p = 0
    count_m = 0
    for g_ind in psi_cl_list:
        g = g_values[g_ind]
        if g == 1:
            count_p += 1
        else:
            count_m += 1
    pi_new = beta.rvs(count_p + gamma1, count_m + gamma2, size=1)

    return pi_new


def get_a(s_counts, alpha1, alpha2):
    count_p = len(s_counts[s_counts.c == 1])
    count_m = len(s_counts[s_counts.c == 0])
    a_new = beta.rvs(count_p + alpha1, count_m + alpha2, size=1)

    return a_new


def get_dist_metric(data, truth_obj_list, prob):
    prob_gt = []
    val = []
    l = len(possible_values)
    for obj_index in range(len(data.O.drop_duplicates())):
        val.append(possible_values)
        prob_gt.append([0]*l)
    for obj_ind, v_true in enumerate(truth_obj_list):
        for v_ind, v in enumerate(val[obj_ind]):
            if v == v_true:
                prob_gt[obj_ind][v_ind] = 1
    prob_gt_vector = []
    prob_vector = []
    for i in range(len(prob_gt)):
        prob_gt_vector += prob_gt[i]
        prob_vector += prob[i]
    dist_metric = np.dot(prob_gt_vector, prob_vector)
    dist_metric_norm = dist_metric/len(prob_gt)
    return dist_metric_norm


def gibbs_fuzzy(data, accuracy_data, g_data, truth_obj_list):
    dist_list = []
    iter_list = []
    for round in range(10):
        var_index, g_values, obj_values, counts, prob, pi_prob, accuracy_list,\
            alpha1, alpha2, gamma1, gamma2 = init_var(data=data)
        iter_number = 0
        dist_metric = 1.
        dist_delta = 0.3
        dist_temp = []
        while dist_delta > eps and iter_number < max_rounds:
            for o_ind in var_index[0]:
                 obj_values[o_ind], counts, prob[o_ind] = get_o(o_ind=o_ind, g_values=g_values,
                                                                obj_values=obj_values, counts=counts,
                                                                data=data, prob=prob, accuracy_list=accuracy_list)

            for g_ind in range(len(g_values)):
                g_prev = g_values[g_ind]
                g_values[g_ind] = get_g(g_ind=g_ind, g_prev=g_prev, pi_prob=pi_prob, obj_values=obj_values,
                                        accuracy_list=accuracy_list, counts=counts, data=data)

            for cl_ind in range(len(pi_prob)):
                pi_prob[cl_ind] = get_pi(cl_ind=cl_ind, g_values=g_values, data=data, gamma1=gamma1, gamma2=gamma2)

            for s_ind in range(len(accuracy_list)):
                s_counts = counts[s_ind]
                accuracy_list[s_ind] = get_a(s_counts=s_counts, alpha1=alpha1, alpha2=alpha2)
            iter_number += 1
            dist_metric_old = dist_metric
            dist_metric = get_dist_metric(data=data, truth_obj_list=truth_obj_list, prob=prob)
            dist_delta = abs(dist_metric-dist_metric_old)
            # print 'dist: {}'.format(dist_metric)
            dist_temp.append(dist_metric)
        print iter_number

        dist_metric = np.mean(dist_temp[-5:])
        dist_list.append(dist_metric)
        iter_list.append(iter_number)
        print 'dist: {}'.format(dist_metric)
        print '------'

    return [np.mean(dist_list), np.mean(iter_list)]






# import sys
# import time
# import pandas as pd
# # sys.path.append('/home/evgeny/fuzzy-fusion/src/')
# sys.path.append('/Users/Evgeny/wonderful_programming/fuzzy-fusion-venv/fuzzy-fusion/src/')
# from generator.generator import generator
# from algorithm.gibbs import gibbs_sampl
# from algorithm.em import em
# from algorithm.m_voting import m_voting
#
# print 'Python version ' + sys.version
# print 'Pandas version ' + pd.__version__
#
# s_number = 5
# obj_number = 20
# cl_size = 2
# cov_list = [0.7]*s_number
# p_list = [0.7]*s_number
# accuracy_list = [random.uniform(0.6, 0.95) for i in range(s_number)]
# accuracy_for_df = [[i, accuracy_list[i]] for i in range(s_number)]
# accuracy_data = pd.DataFrame(accuracy_for_df, columns=['S', 'A'])
#
# result_list = []
# em_t = []
# g_t = []
# gf_t = []
#
# for g_true in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]:
#
#     print g_true
#     print '*****'
#
#     for i in range(10):
#         print i
#         ground_truth = [random.randint(0, len(possible_values)-1) for i in range(obj_number)]
#         data, g_data = generator(cov_list, p_list, ground_truth, cl_size, g_true, possible_values)
#
#         # m_v = m_voting(data=data, truth_obj_list=ground_truth)
#
#
#         # t_em = time.time()
#         em_d, em_it = em(data=data, truth_obj_list=ground_truth, values=possible_values)
#         print 'em: {}'.format(em_d)
#
#         # ex_t_em = time.time() - t_em
#         # em_t.append(ex_t_em)
# #         print("--- %s seconds em ---" % (ex_t_em))
#
#         while True:
#             try:
#                 # t_g = time.time()
#                 # g_d, g_it = gibbs_sampl(data=data, truth_obj_list=ground_truth, values=possible_values)
#                 # print 'g: {}'.format(g_d)
#                 # ex_t_g = time.time() - t_g
#                 # g_t.append(ex_t_g)
#                 # print("--- %s seconds g ---" % (ex_t_g))
#
#                 # t_gf = time.time()
#                 gf_d, gf_it = gibbs_fuzzy(data=data, accuracy_data=accuracy_data, g_data=g_data,
#                                           truth_obj_list=ground_truth)
#                 print 'gf: {}'.format(gf_d)
#                 # print gf_it
#                 print '---'
#                 # ex_t_gf = time.time() - t_gf
#                 # gf_t.append(ex_t_gf)
# #                 print("--- %s seconds gf ---" % (ex_t_gf))
#                 break
#             except ZeroDivisionError:
#                 print 'zero {}'.format(i)
#         result_list.append([g_true, em_d, gf_d])
# df = pd.DataFrame(data=result_list, columns=['g_true', 'em', 'gf'])
# df.to_csv('2_true_param.csv')
