import unittest
from coopheur.batchbuilder.association import event_association_aggregator, event_association_profiler, str_of_rule
from coopheur.batchbuilder.fake_data import fake_order_data_random
from coopheur.batchbuilder.indexer import id_provider, Indexer


class Test_EventAssociation(unittest.TestCase):


    def test__item_association_aggregator__main(self):
        indexer = Indexer()

        ids, orders = fake_order_data_random(100, items_list=[id_provider(ii) for ii in range(100)],
                                             max_items_on_order=10,
                                             order_id_provider=lambda: str(indexer.index()),
                                             seed=1)

        order_profiles, combination_count, rule_count = event_association_aggregator(eventset_ids=ids,
                                                                                    eventset_contents=orders,
                                                                                    max_rule_length=3,
                                                                                    rule_count_threshold=5,
                                                                                    combination_count_threshold=5)
        self.assertEqual(len(order_profiles), len(orders))

        # pprint.pprint(order_profiles)
        # [print(f"{str_of_tuple(key)}: {value}") for key, value in combination_count.items()]
        # [print(str_of_rule(key, (value, ))) for key, value in rule_count.items()]
        # [print(str_of_rule(key, values)) for key, values in sorted(metrics.items(), key=lambda x: x[0][0])]



    def test__eventset_association_profiler__main(self):
        # comb = combinations_of_list(lst, 4)

        indexer = Indexer()
        ids, orders = fake_order_data_random(100, items_list=[id_provider(ii) for ii in range(100)],
                                             max_items_on_order=10,
                                             order_id_provider=lambda: str(indexer.index()),
                                             seed=1)

        # elements = ['j_0', 'g_0', 'h_2', 'q_0']
        # vec_mask = [all(np.isin(elements, row)) for row in orders]
        # deb2 = orders[vec_mask]
        #
        threshold = 0
        order_profiles, combination_count, rule_count, metrics = event_association_profiler(eventset_ids=ids,
                                                                                            eventset_contents=orders,
                                                                                            max_rule_length=3,
                                                                                            rule_count_threshold=threshold,
                                                                                            combination_count_threshold=threshold)

        print(f"Number of qualifying rules: {len(metrics)}")
        n_interesting = 20
        print(f"Top {n_interesting} rules based on Lift")
        [print(str_of_rule(key, values)) for key, values in
         sorted(metrics.items(), key=lambda x: x[1][1], reverse=True)[:n_interesting]]

        lookup = (('g_0', 'j_0'), ('h_2', 'q_0'))
        print(metrics[lookup])
