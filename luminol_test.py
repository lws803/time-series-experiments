from luminol.anomaly_detector import AnomalyDetector
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_FB.csv", header=0, index_col=0)
# df[:].plot(linewidth=2)
# plt.grid(which='both')
# plt.show()

ts = {}
i = 0
for item in df.value:
    ts[i] = item
    i += 1

my_detector = AnomalyDetector(ts)
anomalies_chart = []
score = my_detector.get_all_scores()
for timestamp, value in score.iteritems():
#     print(timestamp, value)
    anomalies_chart.append(value)

list_values = [ v for v in ts.values() ]
plt.plot(list_values)
plt.show()
plt.plot(anomalies_chart, color='r')
plt.show()
