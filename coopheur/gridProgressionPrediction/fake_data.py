from coopstructs.vectors import Vector2
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import random as rnd
from noise import pnoise3
from coopheur.gridProgressionPrediction.dataClassObjs import PeriodObj
import numpy as np
from coopheur.gridProgressionPrediction.utils import normalize_ndarray

def fake_data_random(
        n: int,
        coord_x_bounds: Vector2,
        coord_y_bounds: Vector2,
        datetime_min: datetime,
        datetime_max: datetime,
        value_bounds: Vector2
) -> List[Tuple[datetime, Vector2, float]]:

    ret = []

    datetime_range_seconds = int((datetime_max - datetime_min).total_seconds())

    for ii in range(n):
        # rnd date stamp
        ii_datetime = datetime_min + timedelta(seconds=rnd.randint(0, datetime_range_seconds))

        # rnd coord
        ii_x = rnd.uniform(coord_x_bounds.x, coord_x_bounds.y)
        ii_y = rnd.uniform(coord_y_bounds.x, coord_y_bounds.y)
        ii_coord = Vector2(ii_x, ii_y)

        # rnd value
        ii_val = rnd.uniform(value_bounds.x, value_bounds.y)

        ret.append((ii_datetime, ii_coord, ii_val))


    return ret


def fake_data_perlin2d(n: int,
                       size: Vector2,
                       step_size: Vector2 = None,
                       origin: Vector2 = None,
                       noise_amplitude: float = 1) -> Dict[int, PeriodObj]:
    ret = {}

    if origin is None: origin = Vector2(0, 0)
    if step_size is None: step_size = Vector2(1, 1)

    octaves = 6
    persistence = .5
    lacunarity = 2

    for ii in range(n):
        noise_field = [[
            noise_amplitude * pnoise3(
                (i) / size.x + step_size.x * ii + origin.x,
                (j) / size.y + step_size.y * ii + origin.y,
                0,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=1024,
                repeaty=1024,
                base=0)
        for j in range(size.x)] for i in range(size.y)]


        ret[ii] = PeriodObj(normalize_ndarray(np.array(noise_field)))

    return ret

if __name__ == "__main__":
    from pprint import pprint
    import matplotlib.pyplot as plt
    import time
    # ret = fake_data_random(100,
    #                        coord_x_bounds=Vector2(0, 100),
    #                        coord_y_bounds=Vector2(0, 100),
    #                        datetime_min=datetime.today() - timedelta(days=10),
    #                        datetime_max=datetime.today(),
    #                        value_bounds=Vector2(1000, 2000))

    ret = fake_data_perlin2d(100, Vector2(100, 100), .1, .1, 1)

    plt.ion()
    plt.show()

    for ii, obj in ret.items():
        plt.cla()
        plt.imshow(obj.grid, cmap='gray')
        plt.draw()
        plt.pause(0.01)

