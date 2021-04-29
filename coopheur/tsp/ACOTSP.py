from typing import FrozenSet, Dict, List, Tuple, Any
import numpy as np
from coopbugger.buggers import timer
import math
import matplotlib.pyplot as plt

class AntPathObject:
    def __init__(self, path, distance):
        self.path = path
        self.distance = distance

class ACOReturnObject:
    def __init__(self, paths, index_of_best, pheromone_history, overall_improvement):
        self.paths = paths
        self.index_of_best = index_of_best
        self.pheromone_history = pheromone_history
        self.overall_improvement = overall_improvement

def distances_between_all_points(points: List[Tuple[float, float]]) -> Dict:

    distances = {}

    for ii, point in enumerate(points):
        remaining_points = [x for jj, x in enumerate(points) if ii != jj]
        for other_point in remaining_points:
            key = ((point, other_point))
            if key in distances.keys():
                continue

            point1 = np.array((point[0], point[1]))
            point2 = np.array((other_point[0], other_point[1]))

            # calculating Euclidean distance
            dist = np.linalg.norm(point1 - point2)
            distances[key] = dist
            distances[(key[1], key[0])] = dist

    return distances


def weighted_random_choice(choices_dict: Dict[Any, float], seed: int = None):
    if seed is not None:
        np.random.seed(seed)

    total = sum([v for k, v in choices_dict.items()])
    selection_probs = [v / total for k, v in choices_dict.items()]

    return list(choices_dict)[np.random.choice(len(choices_dict), p=selection_probs)]




def choose_a_next_step(visited: List[Tuple[float, float]],
                  options: List[Tuple[float, float]],
                  distances: Dict[Tuple[Tuple[float, float], Tuple[float, float]], float],
                  pheromone_trails: Dict[Tuple[float, float], float],
                  distance_power: float,
                  pheromone_power: float,
                  seed: int = None
                  ):
    # Get choices
    choices = {x: None for x in options if x not in visited}

    # Analyze each choice for desireability
    for option in choices.keys():
        s_e = (visited[-1], option)
        dist = distances.get(s_e, None)
        if dist is None:
            text = f"The distance value for {s_e} was not provided"
            raise ValueError(text)

        pheromone_strength = pheromone_trails.get((visited[-1], option), 1)

        choices[option] = math.pow(1 / dist, distance_power) * math.pow(pheromone_strength, pheromone_power)

    # Make a choice
    chosen = weighted_random_choice(choices, seed=seed)

    return chosen



def ant_takes_a_path(points: List[Tuple[float, float]],
                     distances: Dict[Tuple[Tuple[float, float], Tuple[float, float]], float],
                     pheromone_trails=None,
                     start: Tuple[float, float]=None,
                     seed: int = None,
                     pheromone_power: float = 1.0,
                     distance_power: float = 4.0,
                     pheromone_decay: float = 0.03):
    """
    :arg points is a list of the points that an ant will need to traverse
    :arg distances is a dictionary containing the distance between the two points in the key
    :arg start is the the point where the ant is assumed to start (if missing, will be random choice)
    """

    if pheromone_trails is None:
        pheromone_trails = {}

    if start is None:
        start = rnd.choice(points)
    visited = [start]
    distance = 0.0

    while True:
        options = [x for x in points if x not in visited]

        if len(options) == 0:
            distance += distances[(visited[-1], start)]
            visited.append(start)
            return AntPathObject(path=visited, distance=distance)

        chosen = choose_a_next_step(visited=visited,
                                    options=options,
                                    distances=distances,
                                    pheromone_trails=pheromone_trails,
                                    distance_power=distance_power,
                                    pheromone_power=pheromone_power,
                                    seed=seed
                                    )

        # update pheromones
        pheromone_trails.setdefault((visited[-1], chosen), 1)
        pheromone_trails[(visited[-1], chosen)] += 1

        pheromone_trails.setdefault((chosen, visited[-1]), 1)
        pheromone_trails[(chosen, visited[-1])] += 1

        for path, trail in pheromone_trails.items():
            pheromone_trails[path] *= (1 - pheromone_decay)

        # update metrics
        distance += distances[(visited[-1], chosen)]
        visited.append(chosen)

@timer
def aco_tsp(points: List[Tuple[float, float]],
            n_ants: int,
            pheromone_power: float = 1.0,
            distance_power: float = 4.0,
            pheromone_decay: float = 0.03,
            early_exit_iterations: int = None
            ):
    dists = distances_between_all_points([x for x in points])

    pheromone_history = [{}]
    paths = []

    best = None
    index_of_best = None

    for ii in range(0, n_ants):
        last_pheromone_trailset = pheromone_history[-1].copy()
        ant_path = ant_takes_a_path(points,
                                    distances=dists,
                                    pheromone_trails=last_pheromone_trailset,
                                    pheromone_power=pheromone_power,
                                    distance_power=distance_power,
                                    pheromone_decay=pheromone_decay)
        pheromone_history.append(last_pheromone_trailset)
        paths.append(ant_path)

        # Update bests
        if best is None or ant_path.distance < best:
            index_of_best = ii
            best = ant_path.distance

        # evaulate early exit criteria
        if ii - index_of_best > early_exit_iterations:
            break

    overall_improvement = (paths[0].distance - paths[index_of_best].distance) / paths[0].distance
    logging.debug(f"initial distance: {round(paths[0].distance, 3)}"
                  f"\nfinal distance: {round(paths[index_of_best].distance, 3)}"
                  f"\nimprovement: {round(overall_improvement * 100, 1)}%")


    return ACOReturnObject(paths, index_of_best, pheromone_history, overall_improvement)


def aco_tsp_display(aco_ret: ACOReturnObject, display_progression: bool = True):

    plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)


    node_color = "navy"
    best_color = "green"
    last_color = "blue"
    pheromone_color = "grey"
    best_hist_color = "red"
    known_best_color = "grey"
    bests = []

    best_value = None
    best_index = None

    for ii in range(1, len(aco_ret.paths)):
        if best_value is None or aco_ret.paths[ii].distance < best_value:
            best_value = aco_ret.paths[ii].distance
            best_index = ii
        current_best = aco_ret.paths[best_index]
        bests.append((ii, current_best))


    if display_progression:
        for ii in range(1, len(aco_ret.paths)):
            ax.clear()
            x_s_last = [x[0] for x in aco_ret.paths[ii].path]
            y_s_last = [x[1] for x in aco_ret.paths[ii].path]
            x_s_best = [x[0] for x in bests[ii][1].path]
            y_s_best = [x[1] for x in bests[ii][1].path]
            ax.scatter(x_s_last, y_s_last, color=node_color)

            for k, v in aco_ret.pheromone_history[ii].items():
                ax.plot([x[0] for x in k], [y[1] for y in k], linewidth=v/len(aco_ret.paths)* 2, color=pheromone_color)

            ax2.plot([x[0] for x in bests], [x[1].distance for x in bests], color=best_hist_color)
            ax2.plot([0, len(aco_ret.paths)], [aco_ret.paths[aco_ret.index_of_best].distance, aco_ret.paths[aco_ret.index_of_best].distance], color=known_best_color, linewidth=0.5)
            ax.plot(x_s_last, y_s_last, color=last_color)
            ax.plot(x_s_best, y_s_best, color=best_color, linewidth=.5)
            fig.canvas.draw()
            fig.canvas.flush_events()

    ax.clear()
    for k, v in aco_ret.pheromone_history[-1].items():
        ax.plot([x[0] for x in k], [y[1] for y in k], linewidth=v / len(aco_ret.paths) * 2, color=pheromone_color)

    ax2.plot([x[0] for x in bests], [x[1].distance for x in bests], color=best_hist_color)
    ax2.plot([0, len(aco_ret.paths)],
             [aco_ret.paths[aco_ret.index_of_best].distance, aco_ret.paths[aco_ret.index_of_best].distance],
             color=known_best_color, linewidth=0.5)
    x_s_best = [x[0] for x in aco_ret.paths[aco_ret.index_of_best].path]
    y_s_best = [x[1] for x in aco_ret.paths[aco_ret.index_of_best].path]
    ax.scatter(x_s_best, y_s_best, color=node_color)
    ax.plot(x_s_best, y_s_best, color=best_color, linewidth=3)
    plt.show(block=True)

if __name__ == "__main__":
    import random as rnd
    import logging
    rootLogger = logging.getLogger('')
    rootLogger.handlers = []
    rootLogger.setLevel(logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').disabled = True

    #SETUP
    points = [
        (0, 0),
        (1, 1),
        (10, 2),
        (-5, 3),
        (0, 0),
        (15, -10)
    ]


    #TEST 1
    # dists = distances(points)
    # for k, v in dists.items():
    #     print(k, v)

    #TEST 2
    # points = [Point(point[0], point[1]) for point in points]
    # aco_tsp(points[0], points)

    #TEST 3
    # points = [Point(rnd.randint(-50, 50), rnd.randint(-50, 50)) for ii in range(0, 500)]
    # dists = aco_tsp(points[0], points)

    # for k, v in dists.items():
    #     print(k, v)

    #TEST 4
    points = [(rnd.randint(-50, 50), rnd.randint(-50, 50)) for ii in range(0, 20)]
    pheromone_power = 1
    pheromone_decay = 0
    distance_power = 4
    aco_ret = aco_tsp(points,
                      n_ants=2500,
                      pheromone_power=pheromone_power,
                      pheromone_decay=pheromone_decay,
                      distance_power=distance_power,
                      early_exit_iterations=250)
    aco_tsp_display(aco_ret, display_progression=False)
