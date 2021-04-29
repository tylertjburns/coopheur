import random as rnd
import time
from pprint import pprint
from coopheur.tsp.constants import MEMO_ACCESS_COUNT, ITERATION_COUNT
from coopheur.tsp.primatives import Point
from coopheur.tsp.memoizedTSP import tsp


def generate_points(n, min_value: int = 0, max_value: int = 100):
    points = []
    for ii in range(0, n):
        new = Point(rnd.randint(min_value, max_value), rnd.randint(min_value, max_value))
        points.append(new)
    return points


def simulate(n, debug_print: bool = False):
    tic = time.perf_counter()

    fset = frozenset(generate_points(n))

    if debug_print:
        print(fset)

    start = Point(0, 0)
    stats = {}

    tsp_ret = tsp(start, fset, run_stats=stats)
    toc = time.perf_counter()
    return (tsp_ret, stats.get(MEMO_ACCESS_COUNT, 0), stats.get(ITERATION_COUNT, 0), toc - tic)


def scaled_simulation(n, debug_print: bool = False):
    performance = {}
    for jj in range(1, n):
        if debug_print:
            print(n)
        performance[jj] = simulate(jj)

        if performance[jj][3] > 30: break
    return performance


rnd.seed(0)
ret = simulate(15, debug_print=True)
pprint(ret)
