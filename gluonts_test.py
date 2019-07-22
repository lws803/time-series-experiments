import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.dataset.util import to_pandas
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.evaluation.backtest import make_evaluation_predictions
import os
import mxnet as mx

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--prediction', default=12, type=int, help='prediction length')
parser.add_argument('--train', action='store_true', help='enter training mode')
parser.add_argument('--epochs', type=int, help='num epochs')
parser.add_argument('--gpu', action='store_true', help='gpu mode')

args = parser.parse_args()


train_ratio = 0.6


def init_data():
    url = "https://raw.githubusercontent.com/numenta/NAB/master/data/artificialNoAnomaly/art_daily_perfect_square_wave.csv"
    df = pd.read_csv(url, header=0, index_col=0)
    # df[:].plot(linewidth=2)
    # plt.grid(which='both')
    # plt.show()
    list_values = []
    for item in df.value:
        list_values.append(item)
    return df, list_values

# Make path when path does not exist
if not os.path.exists("models"):
    os.makedirs("models")


df, list_values = init_data()

training_data = ListDataset(
    [{"start": df.index[0], "target": list_values[:int(len(list_values)*train_ratio)]}],
    freq = "5min"
)

def init_model():
    epochs = None
    context = 'cpu'
    if args.epochs is not None:
        epochs = args.epochs
    if args.gpu:
        context = 'gpu'

    predictor = None
    if args.train:
        my_trainer = Trainer(epochs=epochs, ctx=context) # TODO: Find a way to make it such that we do not set epoch when there is no need to
        estimator = DeepAREstimator(freq="5min", prediction_length=args.prediction, trainer=my_trainer)

        predictor = estimator.train(training_data=training_data)
        predictor.serialize(Path("models/"))
    else:
        # predictor = Predictor.deserialize(Path("models/"))
        predictor = RepresentableBlockPredictor.deserialize(Path("models/"))
        predictor.ctx = mx.Context('cpu')
    return predictor


predictor = init_model()

test_data = ListDataset(
    [{"start": df.index[0], "target": list_values[int(len(list_values)*train_ratio):]}],
    freq = "5min"
)

# TODO: Fix the issue here
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_data,  # test dataset
    predictor=predictor,  # predictor,
    num_eval_samples=100
)
forecasts = list(forecast_it)
tss = list(ts_it)

ts_entry = tss[0]
forecast_entry = forecasts[0]


def plot_prob_forecasts(ts_entry, forecast_entry):
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.show()

plot_prob_forecasts(ts_entry, forecast_entry)
