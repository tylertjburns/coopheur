from typing import List, Tuple, Dict

import numpy as np

from coopheur.gridProgressionPrediction.dataClassObjs import *
from datetime import datetime, timedelta
from coopstructs.vectors import Vector2
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math
from coopheur.gridProgressionPrediction.enums import *

def accumulate(values: List[float], accumulation_func: AccumulationFunction) -> float:
    if accumulation_func == AccumulationFunction.AVG:
        try:
            return sum(values) / len(values)
        except:
            return np.nan
    elif accumulation_func == AccumulationFunction.SUM:
        return sum(values)
    elif accumulation_func == AccumulationFunction.COUNT:
        return len(values)
    else:
        raise NotImplementedError(f"Accumulation type {accumulation_func} not implemented")


def translate_to_time_period_grid_data(
    occurence_data: List[Tuple[datetime, Vector2, float]],
    grid_def: GridDefinition,
    time_box_size: timedelta,
    n_periods: int,
    start_datetime: datetime,
    accum_func: AccumulationFunction
) -> Dict[int, PeriodObj]:

    ret = {}
    ii = 0
    while ii < n_periods:
        start = start_datetime + ii * time_box_size
        end = start_datetime + (ii + 1) * time_box_size

        relevant_occurances = [x for x in occurence_data if start <= x[0] < end]

        array = np.ndarray(shape=(grid_def.shape.x, grid_def.shape.y), dtype=float)

        for x in range(grid_def.shape.x):
            for y in range(grid_def.shape.y):
                records_in_grid = [
                    occur[2] for occur in relevant_occurances
                    if grid_def.origin.x + x * grid_def.size.x <=
                       occur[1].x <
                       grid_def.origin.x + (x + 1) * grid_def.size.x
                and
                       grid_def.origin.y + y * grid_def.size.y <=
                       occur[1].y <
                       grid_def.origin.y + (y + 1) * grid_def.size.y
                ]
                array[x, y] = accumulate(records_in_grid, accumulation_func=accum_func)

        ret[ii] = PeriodObj(start_datetime=start,
                            end_datetime=end,
                            grid=array)
        ii += 1


    return ret

def decision_tree_regression_grid_fill(grid: np.ndarray) -> np.ndarray:

    x_s = [list(index) for index, val in np.ndenumerate(grid) if not math.isnan(val)]
    y_s = [val for index, val in np.ndenumerate(grid) if not math.isnan(val)]

    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(x_s, y_s)

    ret = np.ndarray(grid.shape)
    for index, _ in np.ndenumerate(grid):
        prediction = regressor.predict([list(index)])
        ret[index[0], index[1]] = prediction[0]
    return ret

def plot_grid_progression(time_period_grid_data: Dict[int, np.ndarray], pt_scale: int = None):
    if pt_scale is None: pt_scale = 10

    plt.ion()
    plt.show()

    for ii, grid in time_period_grid_data.items():
        plt.cla()

        x_s = []
        y_s = []
        sizes = []
        for index, val in np.ndenumerate(grid):
            if not math.isnan(val):
                x_s.append(index[0])
                y_s.append(index[1])
                sizes.append(val * pt_scale)

        plt.scatter(x=x_s, y=y_s, s=sizes)
        plt.draw()
        plt.pause(1)

def grid_prediction_at_t(time_period_grid_data: Dict[int, np.ndarray], t: int) -> Optional[np.ndarray]:
    if len(time_period_grid_data) == 0:
        return None

    max_ii = max([ii for ii in time_period_grid_data.keys()])
    min_ii = min([ii for ii in time_period_grid_data.keys()])
    if t < min_ii:
        raise ValueError(f"Cannot compute a prediction for a t earlier than provided records")
    elif min_ii <= t <= max_ii:
        return time_period_grid_data[t]
    else:
        ret = np.ndarray(next(grid for ii, grid in time_period_grid_data.items()).shape)

        periods = list(time_period_grid_data.keys())
        for index, _ in np.ndenumerate(ret):
            y_s = [time_period_grid_data[period][index[0], index[1]] for period in periods]

            prediction = value_prediction_at_t([[period] for period in periods], y_s, t=t)
            ret[index[0], index[1]] = prediction[0]

        return ret

def value_prediction_at_t(x_s: List[List[float]], y_s: List[float], t: float, degree: int = None):
    if degree is None: degree = 2

    poly_reg = PolynomialFeatures(degree=degree)
    x_poly = poly_reg.fit_transform(x_s)
    lin_reg = LinearRegression()
    lin_reg.fit(x_poly, y_s)

    return lin_reg.predict(poly_reg.fit_transform([[t]]))

def normalize_ndarray(data: np.ndarray):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def predictions_per_period(time_period_grid_data: Dict[int, np.ndarray], idxs: List[int] = None) -> Dict[int, np.ndarray]:
    ret = {}

    if idxs is None: idxs = list(time_period_grid_data.keys())

    for ii in idxs:
        # make a prediction
        prediction = grid_prediction_at_t({k: v for k, v in time_period_grid_data.items() if k < ii}, ii)
        if prediction is None: continue
        mask_for_predict_to_increase = np.ndarray(prediction.shape)
        for index, val in np.ndenumerate(prediction):
            mask_for_predict_to_increase[index] = val - time_period_grid_data[ii][index]

            # if val > time_period_grid_data[ii][index]:
            #     mask_for_predict_to_increase[index] = val
            # else:
            #     mask_for_predict_to_increase[index] = 0
        ret[ii] = mask_for_predict_to_increase

    return ret
