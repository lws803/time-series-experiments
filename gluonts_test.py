import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.dataset.util import to_pandas
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from gluonts.model.predictor import Predictor
import os

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--prediction', default=12, type=int, help='prediction length')
parser.add_argument('--train', action='store_true', help='enter training mode')
parser.add_argument('--epochs', type=int, help='num epochs')

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
    epochs = 10
    if args.epochs is not None:
        epochs = args.epochs

    my_trainer = Trainer(epochs=epochs, ctx='gpu')
    estimator = DeepAREstimator(freq="5min", prediction_length=args.prediction, trainer=my_trainer)
    
    predictor = None
    if args.train:
        predictor = estimator.train(training_data=training_data)
        predictor.serialize(Path("models/"))
    else:
        predictor = Predictor.deserialize(Path("models/"))
    return predictor


predictor = init_model()

test_data = ListDataset(
    [{"start": df.index[0], "target": list_values[int(len(list_values)*train_ratio):]}],
    freq = "5min"
)

for test_entry, forecast in zip(test_data, predictor.predict(test_data)):
    to_pandas(test_entry)[-200:].plot(linewidth=2)
    forecast.plot(color='g', 
        prediction_intervals=[50.0, 90.0]
        )
plt.grid(which='both')

plt.show()
