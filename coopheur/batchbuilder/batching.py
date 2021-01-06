import numpy as np
from typing import Callable, List
from coopheur.batchbuilder.fake_data import fake_batching_data, fake_order_data_random
from coopbugger.buggers import timer
import random as rnd
from coopheur.batchbuilder.indexer import id_provider, Indexer
from coopheur.batchbuilder.association import eventset_association_profiler

def max_commonality_aggregator(old: np.array, new: np.array) -> np.array:
    if old is None:
        return new
    else:
        return np.maximum(old, new)

@timer
def build_a_batch(ids: np.array,
                  seed_priorities: np.array,
                  batch_size: int,
                  item_to_batch_provider: Callable[[np.array, np.array], str],
                  priority_tie_breaker: Callable[[np.array], int] = None) -> np.array:

    remaining_priorities = seed_priorities

    batched_items = []
    remaining_ids = ids
    while len(batched_items) < batch_size and len(remaining_ids) > 0:
        # Add item to batch
        if len(batched_items) == 0:
            # Choose a seed
            next_ind = choose_a_seed(remaining_ids, remaining_priorities, priority_tie_breaker)
        else:
            # Choose index with highest commonality to batch
            next_ind = get_next_item_for_batch(np.array(batched_items),
                                               np.array(list(set(remaining_ids) - set(batched_items))),
                                               item_to_batch_provider)

        batched_items.append(remaining_ids[next_ind])

        # update id matrix
        remaining_ids = np.delete(remaining_ids, next_ind)

        # update priority matrix
        remaining_priorities = np.delete(remaining_priorities, next_ind)

    return np.array(batched_items)

def choose_a_seed(ids: np.array,
                  seed_priorities: np.array,
                  priority_tie_breaker: Callable[[np.array], int] = None):
    # Choose Seed
    highest_prio_indicies = np.where(seed_priorities == np.amin(seed_priorities))[0]
    if len(highest_prio_indicies) == 1 or priority_tie_breaker is None:
        # choose first item of highest priority
        next_ind = highest_prio_indicies[0]
    else:
        # use the tie breaker to choose the next item
        next_ind = np.where(ids == priority_tie_breaker(ids[highest_prio_indicies]))[0][0]

    return next_ind

def get_next_item_for_batch(current_batch: np.array,
                            options: np.array,
                            item_to_batch_provider: Callable[[np.array, np.array], str] = None):

    # Run the item_to_batch_provider
    ret = item_to_batch_provider(current_batch, options)

    # Verify that provided value is in options
    if not np.isin(ret, options):
        raise ValueError(f"The returned value needs to be one of the options provided. {ret} was returned but should be in {options}")

    # Get the index of the value provided
    next_ind = np.where(options == ret)[0][0]

    #return index
    return next_ind



@timer
def request_batches_by_association_matrix(ids: np.array,
                                          association_matrix: np.array,
                                          n_batches: int,
                                          batch_size: int,
                                          seed_priorities: np.array,
                                          priority_tie_breaker: Callable[[np.array], int] = None):

    i_t_b_provider = lambda x, y: association_matrix_item_to_batch_provider(ids, association_matrix, x, y)

    batches = request_batches(ids=ids,
                              seed_priorities=seed_priorities,
                              n_batches=n_batches,
                              batch_size=batch_size,
                              item_to_batch_provider=i_t_b_provider,
                              priority_tie_breaker=priority_tie_breaker)

    return batches

def association_matrix_item_to_batch_provider(ids: np.array, association_matrix: np.array, batch: np.array, options: np.array)->str:
    batched_indexes = np.where(np.isin(ids, batch))[0]
    options_indexes = np.where(np.isin(ids, options))[0]

    if len(options) == 1:
        return options[0]

    evaluation_matrix = association_matrix[:, batched_indexes][options_indexes, :]

    if len(batch) > 1:
        batch_commonality = np.amax(evaluation_matrix, axis=1)
    else:
        batch_commonality = evaluation_matrix


    try:
        indexes = np.where(batch_commonality == np.amax(batch_commonality))[0]


        ret = options[indexes][0]
    except :
        deb = True

    return ret

@timer
def request_batches(ids: np.array,
                    seed_priorities: np.array,
                    n_batches: int,
                    batch_size: int,
                    item_to_batch_provider: Callable[[np.array, np.array], str],
                    priority_tie_breaker: Callable[[np.array], int] = None):
    remaining_priorities = seed_priorities
    batches = []
    while len(ids) > 0 and n_batches > 0:
        batch_item_ids = build_a_batch(ids,
                                       remaining_priorities,
                                       batch_size,
                                       item_to_batch_provider=item_to_batch_provider,
                                       priority_tie_breaker=priority_tie_breaker)
        batches.append(batch_item_ids)

        # find indexes of batched items
        indexes = np.where(np.isin(ids, batch_item_ids))

        # update id matrix
        ids = np.delete(ids, indexes)

        # update priority matrix
        remaining_priorities = np.delete(remaining_priorities, indexes)

        n_batches -= 1
    return batches

def id_provider(ii):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    alpha = ii%26
    numeric = ii // 26
    return f"{alphabet[alpha]}_{numeric}"


def uniques(list_of_list: List[List]):
    my_set = set()
    all = []

    [[(my_set.add(x), all.append(x)) for x in lst] for lst in list_of_list]
    return my_set, all


def batch_aggregates(ids: np.array, orders: np.array, batches_to_aggregate: List[List]):

    unique_lst = [
        uniques([list([x for x in item if x]) for item in [orders[np.where(ids == item)[0][0], :] for item in batch]])
        for ii, batch in enumerate(batches_to_aggregate)]

    return unique_lst


def evaluate_an_item_to_batch_provider(sample_size: int = 100,
                                       n_orders: int = 100,
                                       n_items: int = 100,
                                       n_batches: int = 25,
                                       batch_size: int = 4):
    indexer = Indexer()
    ret = []


    for ii in range(sample_size):
        ids, orders = fake_order_data_random(n_orders, items_list=[id_provider(ii) for ii in range(n_items)],
                                        max_items_on_order=4,
                                        order_id_provider=lambda: str(indexer.index()),
                                        seed=ii
                                        )
        association_matrix = eventset_association_profiler(ids, orders)
        i_t_b_provider = lambda x, y: association_matrix_item_to_batch_provider(ids, association_matrix, x, y)
        base = request_batches(ids=ids,
                           seed_priorities=np.ones_like(ids, dtype=float),
                           n_batches=n_batches,
                           batch_size=batch_size,
                           item_to_batch_provider=lambda x, y: y[rnd.randint(0, len(y) - 1)],
                           priority_tie_breaker=lambda x: x[-1])

        test = request_batches(ids=ids,
                           seed_priorities=np.ones_like(ids, dtype=float),
                           n_batches=n_batches,
                           batch_size=batch_size,
                           item_to_batch_provider=i_t_b_provider,
                           priority_tie_breaker=lambda x: x[-1])

        batchset_stats_1 = batchset_stats(ids, orders, base)
        n_unique_1 = sum(lst[3] for lst in batchset_stats_1)
        n_tot_1 = sum(lst[4] for lst in batchset_stats_1)


        batchset_stats_2 = batchset_stats(ids, orders, test)
        n_unique_2 = sum(lst[3] for lst in batchset_stats_2)
        n_tot_2 = sum(lst[4] for lst in batchset_stats_2)

        ret.append((n_unique_1,  n_unique_2,  n_tot_2))

    return ret

def batchset_stats(ids: np.array, orders: np.array, batchset: List[List]):
    stats = []

    aggregates = batch_aggregates(ids, orders, batchset)

    # Create list of index, batch, items in batch, num_unique items, num total items for each batch in the batchset
    [stats.append((ii,
                   batch,
                   [list([x for x in item if x]) for item in [orders[np.where(ids == item)[0][0], :] for item in batch]],
                   len(aggregates[ii][0]),
                   len(aggregates[ii][1]),
                   ))
     for ii, batch in enumerate(batchset)]

    return stats

if __name__ == "__main__":
    import logging

    loggingLvl = logging.DEBUG

    rootLogger = logging.getLogger('')
    rootLogger.handlers = []
    rootLogger.setLevel(loggingLvl)

    #formatters
    # fileFormatter = logging.Formatter('%(name)s -- %(asctime)s : [%(levelname)s] %(message)s (%(filename)s lineno: %(lineno)d)')
    consoleFormatter = logging.Formatter('%(name)s -- %(asctime)s : [%(levelname)s] %(message)s (%(filename)s lineno: %(lineno)d)')
    uiFormatter = logging.Formatter(' %(levelname)s: %(message)s')


    #Console Handler
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(consoleFormatter)
    rootLogger.addHandler(console)


    # Init Items
    NUM_ITEMS = 1000
    associations, priorities = fake_batching_data(NUM_ITEMS)

    ids = np.array([id_provider(ii) for ii in range(NUM_ITEMS)])

    with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=100):
        print(associations)
        # # print(priorities)
        # # print(ids)

    n_batches = 4
    batch_size = 250

    # ids = build_a_batch(ids=ids, associations=associations, seed_priorities=priorities, batch_size=batch_size, commonality_aggregator=max_commonality_aggregator)
    # print(ids)

    batches = request_batches(ids=ids, associations=associations, seed_priorities=priorities, n_batches=n_batches, batch_size=batch_size, commonality_aggregator=max_commonality_aggregator)
    [print(ii, batch) for ii, batch in enumerate(batches)]
