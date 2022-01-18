from coopheur.gridProgressionPrediction.dataClassObjs import *
from datetime import datetime, timedelta
from coopstructs.vectors import Vector2
from coopheur.gridProgressionPrediction.utils import translate_to_time_period_grid_data, grid_prediction_at_t, \
    decision_tree_regression_grid_fill, predictions_per_period
from coopheur.gridProgressionPrediction.enums import *
from coopheur.gridProgressionPrediction.fake_data import fake_data_random, fake_data_perlin2d
from typing import Dict

# The goal of this utility is to take a series of time-stamped value/coordinate pairs and split them up into a grid in
# periods based on a time-box. Then the routine predicts values of the grid moving forward into future time periods

def run_with_fake_occurence_data():
    start = datetime.today() - timedelta(days=10)
    occurence_data = fake_data_random(1000,
                                      coord_x_bounds=Vector2(0, 100),
                                      coord_y_bounds=Vector2(0, 100),
                                      datetime_min=start,
                                      datetime_max=datetime.today(),
                                      value_bounds=Vector2(1000, 2000))

    tpgd = translate_to_time_period_grid_data(occurence_data,
                                              n_periods=10,
                                              grid_def=GridDefinition(shape=Vector2(10, 10),
                                                                      size=Vector2(10, 10),
                                                                      origin=Vector2(0, 0)),
                                              time_box_size=timedelta(days=1),
                                              start_datetime=start,
                                              accum_func=AccumulationFunction.AVG
                                              )
    return tpgd

def plot_continuous(hist_size: int, grid_size: Vector2):

    origin = Vector2(0, 0)
    step_size = Vector2(.1, .1)

    # init
    tpgd = fake_data_perlin2d(hist_size, size=grid_size, step_size=step_size, origin=origin)

    plt.ion()
    fix, axes = plt.subplots(1, 2)
    plt.show()

    cc = 0
    while True:
        idx = hist_size + cc - 1

        tpgd_lean = {ii: obj.grid for ii, obj in tpgd.items()}
        predictions = predictions_per_period(time_period_grid_data=tpgd_lean, idxs=[idx])

        plt.cla()
        axes[1].imshow(tpgd[cc].grid, cmap='gray')

        data_masked = predictions[idx]
        data_masked[data_masked == 0] = np.nan
        axes[0].imshow(data_masked, interpolation='none', vmin=0, alpha=0.75)

        plt.draw()
        plt.pause(0.01)

        #delete old records
        while len(tpgd) > hist_size:
            min_idx = min(tpgd.keys())
            del tpgd[min_idx]

        #add new record
        new = fake_data_perlin2d(1, size=grid_size, step_size=step_size, origin=origin + step_size * (idx + 1))
        tpgd[idx + 1] = new[0]

        # increment
        cc += 1


def plot_with_prediction(time_period_grid_data: Dict[int, np.ndarray],
                         predictions: Dict[int, np.ndarray] = None):
    plt.ion()
    plt.show()

    for ii, obj in time_period_grid_data.items():

        plt.cla()
        plt.imshow(obj, cmap='gray')
        if predictions and predictions.get(ii+1, None) is not None:
            # data_masked = np.ma.masked_where(predictions[ii + 1] != 0, predictions[ii + 1])
            data_masked = predictions[ii+1]
            data_masked[data_masked == 0] = np.nan
            plt.imshow(data_masked, interpolation='none', vmin=0, alpha=0.75)
        plt.draw()
        plt.pause(0.01)

    plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # tpgd = fake_data_perlin2d(50, Vector2(50, 50), step_size=Vector2(.1, .1), noise_amplitude=1)
    # # tpgd_lean = {ii: decision_tree_regression_grid_fill(obj.grid) for ii, obj in tpgd.items()}
    # tpgd_lean = {ii: obj.grid for ii, obj in tpgd.items()}
    # predictions = predictions_per_period(time_period_grid_data=tpgd_lean)
    # plot_with_prediction(time_period_grid_data=tpgd_lean,
    #                      predictions=predictions)

    plot_continuous(20, Vector2(100, 100))

    # plot_grid_progression({ii: obj.filled() for ii, obj in tpgd.items()}, 10)

    # print("FIRST PERIOD RAW AND FILLED")
    # print(tpgd[0].grid, decision_tree_regression_grid_fill(tpgd[0].grid))
    # print("\n")
    #
    # print("GRID PREDICTION")
    # print(grid_prediction_at_t({ii: decision_tree_regression_grid_fill(obj.grid) for ii, obj in tpgd.items()}, 10))
    # print("\n")