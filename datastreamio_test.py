from dsio.main import restream_dataframe
from dsio.anomaly_detectors import Percentile1D

# pip install -e git+https://github.com/MentatInnovations/datastream.io#egg=dsio

import pandas as pd
dataframe = pd.read_csv('https://raw.githubusercontent.com/numenta/NAB/master/data/artificialWithAnomaly/art_daily_jumpsup.csv', header=0, index_col=0)
detector = Percentile1D
restream_dataframe(dataframe, detector, sensors=['value'], cols=2, speed=50)
