import numpy as np
import itertools
from typing import List, Tuple
from coopheur.batchbuilder.fake_data import fake_order_data_random
from coopheur.batchbuilder.indexer import id_provider, Indexer

def event_association_aggregator(eventset_ids: np.array, eventset_contents: np.array, max_rule_length: int = 2, rule_count_threshold:int = 0, combination_count_threshold:int=0):
    # store the combinations that are found on the eventsets
    combinations = {}

    # store the generated rules that are found along with a count of occurrence
    rules = {}

    # store the rules that exist on a given eventset
    eventset_profiles = {}

    # iterate on the eventset contents to find combinations, rules and eventset profiles
    for ii, x in enumerate(eventset_contents):
        order_combinations = combinations_of_list([item for item in x.tolist() if item], max_rule_length)
        eventset_profiles[eventset_ids[ii]] = order_combinations

        for comb in order_combinations:
            combinations.setdefault(comb, 0)
            combinations[comb] += 1
            for conseq in order_combinations:
                if not any(x in comb for x in conseq):
                    rules.setdefault((comb, conseq), 0)
                    rules[(comb, conseq)] += 1

    # deb = combinations[('g_0', 'q_0')]

    return eventset_profiles,\
          {k:v for k, v in sorted(combinations.items(), key=lambda item: item[1]) if v > combination_count_threshold}, \
          {k:v for k, v in  sorted(rules.items(), key=lambda item: item[1]) if v > rule_count_threshold}

def event_association_profiler(eventset_ids: np.array, eventset_contents: np.array, max_rule_length: int = 2, rule_count_threshold:int = 0, combination_count_threshold:int=0):
    eventset_profiles, combination_count, rule_count = event_association_aggregator(eventset_ids=eventset_ids,
                                                                eventset_contents=eventset_contents,
                                                                max_rule_length=max_rule_length,
                                                                rule_count_threshold=rule_count_threshold,
                                                                combination_count_threshold=combination_count_threshold)




    metrics = {}
    for rule, count in rule_count.items():
        count_ant = combination_count[rule[0]]
        count_con = combination_count[rule[1]]
        count_ant_intersec_conseq = count # len([x for x in order_profiles if rule[0] in x and rule[1] in x])
        count_ant_union_conseq = count_ant + count_con - count_ant_intersec_conseq
        p_ant = combination_count[rule[0]] / len(eventset_contents)
        p_con = combination_count[rule[1]] / len(eventset_contents)
        p_ant_intersect_con = count_ant_intersec_conseq / len(eventset_contents)
        p_conseq_given_ant = count_ant_intersec_conseq / count_ant
        p_ant_union_conseq = count_ant_union_conseq / len(eventset_contents)

        confidence = p_conseq_given_ant
        support = count_ant_intersec_conseq / len(eventset_contents)
        lift = p_ant_intersect_con / (p_ant * p_con)
        jaccard = p_ant_intersect_con / p_ant_union_conseq

        metrics[rule] = (confidence, support, lift, jaccard)

    return eventset_profiles, combination_count, rule_count, metrics




def eventset_association_profiler(eventset_ids: np.array, eventset_contents: np.array):

    association_matrix = np.ones((len(eventset_contents), len(eventset_contents)))

    for ii, first_eventset in enumerate(eventset_contents):
        for jj, second_eventset in enumerate(eventset_contents):
            if jj <= ii:
                pass
            else:
                association_ij, association_ji = similairity_between_eventsets(eventset_contents[ii], eventset_contents[jj])
                association_matrix[ii][jj] = association_ij
                association_matrix[jj][ii] = association_ji

    return association_matrix

def similairity_between_eventsets(a: np.array, b: np.array) -> (float, float):

    a_set = set([item for item in a.tolist() if item])
    b_set = set([item for item in b.tolist() if item])
    intersect_a_b = a_set.intersection(b_set)

    perc_a_in_b = len(intersect_a_b) / len(a_set)
    perc_b_in_a = len(intersect_a_b) / len(b_set)

    return perc_a_in_b, perc_b_in_a


def combinations_of_list(item_list: List, max_rule_length: int):
    rules = []
    # Remove duplicates and sort to ensure a constant output
    sorted_items = sorted(list(set(item_list)))

    # Get combinations of all lengths
    for ii in range(1, max_rule_length + 1):
        rules += itertools.combinations(sorted_items, ii)

    return rules

def str_of_tuple(tup: Tuple):
    ret = ""
    for x in tup:
        if type(x) in [float, int]:
            x = round(x, 2)

        if ret == "":
            ret += str(x)
        else:
            ret += f"|{x}"

    return ret

def str_of_rule(rule_tup: Tuple, values: Tuple):
    return str_of_tuple(rule_tup[0]) + " --> " + str_of_tuple(rule_tup[1]) + f": {str_of_tuple(values)}"

class AssociationRuleSet:
    def __init__(self, antecedent: Tuple, consequent: Tuple):
        self.antecedent = antecedent
        self.consequent = consequent


if __name__ == "__main__":
    indexer = Indexer()
    ids, orders = fake_order_data_random(10, items_list=[id_provider(ii) for ii in range(10)], max_items_on_order=4,
                                  order_id_provider=lambda: str(indexer.index()),
                                  seed=1)


    # ids, orders = fake_order_data_fixed_small()
    association_matrix = eventset_association_profiler(ids, orders)

    with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.2f}'.format}, linewidth=100):
        print(orders)
        print(association_matrix)

