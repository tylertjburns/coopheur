import numpy as np
import random as rnd
from typing import List, Callable

def fake_batching_data(n_items: int, seed: int = None) ->  (np.array, np.array):
    associations = []

    if seed is not None:
        rnd.seed(seed)

    for ii in range(n_items):
        row = []
        for jj in range(n_items):
            if ii == jj:
                row.append(1)
            elif ii > jj:
                row.append(associations[jj][ii])
            else:
                row.append(rnd.random())
        associations.append(row)

    priorities = []

    for ii in range(n_items):
        priorities.append(rnd.randint(0, 10))

    return np.array(associations), np.array(priorities)


def fake_order_data_random(n_orders: int, items_list: List, max_items_on_order: int, order_id_provider: Callable[..., str], seed:int = None) -> (np.array, np.array):
    if seed is not None:
        rnd.seed(seed)

    ids = []
    items = []
    for ii in range(n_orders):
        # Get Order Name
        id = order_id_provider()

        # Generate items
        n_items = rnd.randint(1, max_items_on_order)
        order_items = rnd.choices(items_list, k=n_items)

        # Add
        ids.append(id)
        items.append(order_items)

    length = max(map(len, items))
    items_array = np.array([order_items+[None]*(length-len(order_items)) for order_items in items])

    return np.array(ids), items_array

def fake_order_data_fixed_small() -> (np.array, np.array):

    ids = [1, 2, 3]
    items = [   ['a', 'b', 'c'],
                ['b', 'c'],
                ['a', 'd']
            ]

    length = max(map(len, items))
    items_array = np.array([order_items+[None]*(length-len(order_items)) for order_items in items])

    return np.array(ids), items_array




if __name__ == "__main__":
    from batchbuilder.indexer import Indexer, id_provider
    associations, priorities = fake_batching_data(10)

    print(associations)
    print(priorities)

    indexer = Indexer()
    ids, orders = fake_order_data_random(100, items_list=[id_provider(ii) for ii in range(100)], max_items_on_order=10, order_id_provider = lambda: str(indexer.index()))

    print(ids)
    print(orders)