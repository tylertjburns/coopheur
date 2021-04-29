from typing import FrozenSet
from coopheur.tsp.primatives import Point
from coopheur.tsp.constants import MEMO_ACCESS_COUNT, ITERATION_COUNT

def tsp(start: Point, points: FrozenSet[Point], memo=None, run_stats=None, deb_print: bool = False):
    if memo is None:
        memo = {}

    if run_stats is None:
        run_stats = {}

    run_stats.setdefault(ITERATION_COUNT, 0)
    run_stats[ITERATION_COUNT] += 1

    if len(points) == 0:  # No points case
        ret = (0, [start])
    elif len(points) == 1:  # Base Case
        end = next(iter(points))
        path = [start, end]
        new = (end.dist(start), path)
        ret = new
    elif (start, points) in memo:  # Memo Case
        run_stats.setdefault(MEMO_ACCESS_COUNT, 0)
        run_stats[MEMO_ACCESS_COUNT] += 1
        if deb_print:
            print(f"reading MEMO: {(start, points)} -- {memo[(start, points)]}")
        ret = memo[(start, points)]
    else:  # Calculation Case
        options = []
        for point in points:
            sub_tsp = tsp(point, points - frozenset([point]), memo, run_stats)
            new_dist = sub_tsp[0] + point.dist(start)
            path = list(sub_tsp[1])
            path.insert(0, start)
            options.append((new_dist, path))

        best = min(options, key=lambda x: x[0])
        if deb_print:
            print(f"MEMO-ing: {(start, points)} -- {best}")
        if start != best[1][0]: raise Exception(f"start in best: {start} in {best[1]}")
        memo[(start, points)] = best

        if deb_print:
            print(f"options: {options}")
            print(f"memo: {memo}")
        ret = memo[(start, points)]

    if deb_print:
        print(ret)
    return ret