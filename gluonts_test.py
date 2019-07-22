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
args = parser.parse_args()


train_ratio = 0.6

url = "https://raw.githubusercontent.com/numenta/NAB/master/data/artificialNoAnomaly/art_daily_perfect_square_wave.csv"
df = pd.read_csv(url, header=0, index_col=0)
# df[:].plot(linewidth=2)
# plt.grid(which='both')
# plt.show()

list_values = []

for item in df.value:
    list_values.append(item)


if not os.path.exists("models"):
    os.makedirs("models")

training_data = ListDataset(
    [{"start": df.index[0], "target": list_values[:int(len(list_values)*train_ratio)]}],
    freq = "5min"
)


estimator = DeepAREstimator(freq="5min", prediction_length=args.prediction, trainer=Trainer(epochs=10))
predictor = None
if args.train:
    predictor = estimator.train(training_data=training_data)
    predictor.serialize(Path("models/"))
else:
    predictor = Predictor.deserialize(Path("models/"))


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
