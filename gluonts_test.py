import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.dataset.util import to_pandas
import matplotlib.pyplot as plt

train_ratio = 0.6

url = "https://raw.githubusercontent.com/numenta/NAB/master/data/artificialNoAnomaly/art_daily_perfect_square_wave.csv"
df = pd.read_csv(url, header=0, index_col=0)
# df[:].plot(linewidth=2)
# plt.grid(which='both')
# plt.show()

list_values = []

for item in df.value:
    list_values.append(item)

plt.plot(list_values)
plt.show()

training_data = ListDataset(
    [{"start": df.index[0], "target": list_values[:int(len(list_values)*train_ratio)]}],
    freq = "5min"
)


estimator = DeepAREstimator(freq="5min", prediction_length=200, trainer=Trainer(epochs=10))
predictor = estimator.train(training_data=training_data)


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
