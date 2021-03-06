'''
The implementation of Sums algorithm from
Pasternack, J & Roth, D., (2010),
Knowing What to Believe. In COLING.

@author: Evgeny Krivosheev (e.krivoshe@gmail.com)
'''


from numpy.random import beta

max_rounds = 20
eps = 10e-3


def get_trustw(data, belief, sources):
    trustw_new = []
    for s in sources:
        claim_beliefs = 0.
        for obj_index in data.keys():
            obj_data = data[obj_index]
            if s not in obj_data[0]:
                continue
            obj_possible_values = sorted(set(obj_data[1]))
            observed_val = obj_data[1][obj_data[0].index(s)]
            val_ind = obj_possible_values.index(observed_val)
            claim_beliefs += belief[obj_index][val_ind]
        trustw_new.append(claim_beliefs)

    t_max = max(trustw_new)
    trustw_new = map(lambda t: t/t_max, trustw_new)

    return trustw_new


def get_belief(data, trustw_list):
    belief = {}
    for obj_index in data.keys():
        obj_data = data[obj_index]
        possible_values = sorted(set(obj_data[1]))
        l = len(possible_values)
        term_list = [0]*l
        for s_ind, v in zip(obj_data[0], obj_data[1]):
            s_trustw = trustw_list[s_ind]
            term_ind = possible_values.index(v)
            term_list[term_ind] += s_trustw

        b_max = max(term_list)
        if b_max == 0.:
            b_max = 1.
        term_list = map(lambda b: b/b_max, term_list)
        belief.update({obj_index: term_list})

    return belief


def sums(s_number, data, alpha=(4, 1)):
    sources = range(s_number)
    # trustw_list = [np.random.uniform(0.7, 1.0) for _ in range(s_number)]
    trustw_list = list(beta(alpha[0], alpha[1], s_number))
    trustw_delta = 0.3
    iter_number = 0
    while trustw_delta > eps and iter_number < max_rounds:
        belief = get_belief(data=data, trustw_list=trustw_list)
        trustw_prev = trustw_list[:]
        trustw_list = get_trustw(data=data, belief=belief, sources=sources)
        trustw_delta = max([abs(k-l) for k, l in zip(trustw_prev, trustw_list)])
        iter_number += 1

    return belief
