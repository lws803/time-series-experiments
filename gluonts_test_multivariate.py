from gluonts.dataset.common import ListDataset
from gluonts.distribution import MultivariateGaussianOutput
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


N = 10  # number of time series
T = 100  # number of timesteps
prediction_length = 50
freq = '1H'

custom_datasetx = np.random.normal(size=(2, N, T))

print(custom_datasetx.shape)


plt.plot(custom_datasetx[0])
plt.show()

start = pd.Timestamp("01-01-2019", freq=freq) 

train_ds = [{'target': x, 'start': start} for x in custom_datasetx[:, :, :-prediction_length]]
test_ds = [{'target': x, 'start': start} for x in custom_datasetx[:,:,:]]

# Trainer parameters
epochs = 1
learning_rate = 1E-3
batch_size=1
num_batches_per_epoch=2

# create estimator
estimator = DeepAREstimator(    
    prediction_length=prediction_length,
    
    context_length=prediction_length, 
    
    freq=freq, 
    
#     trainer=Trainer(ctx="gpu", epochs=epochs, learning_rate=learning_rate, hybridize=True, 
#                     batch_size=batch_size, num_batches_per_epoch=num_batches_per_epoch,),

    distr_output=MultivariateGaussianOutput(dim=2),
)

predictor = estimator.train(train_ds)
