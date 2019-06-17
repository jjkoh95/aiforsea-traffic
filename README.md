# AI For SEA - Traffic Management

## Environment (main libraries used)
I suppose Python 3.6 and tensorflow 1.13 should work fine with the code.
- Python 3.7
- Tensorflow 1.14.1 (tf-nightly)
- Keras 2.2.4

## To evaluate test dataset
- install any necessary packages (requirements.txt is not optimised as it inherits my base conda environment)
- start notebooks/5-model-evaluation and look for Evaluation/Submission tab

## Feature engineering
I tried to be minimalistic and realistic. While I was given the choice to pick up to 2 weeks of historical data, I found that with only 8 weeks of data, if I did a walk-forward split of my training data, I would end up with about 4 weeks of training data (with very little validation data).
- normalized distance from hand-picked POI (qp09d8, qp03xx, qp03wf)
- datetime features day, hour, per-fifteen-mins - sin/cos equivalent
- last 2 hours demands - T, T-1, T-2, .. T-7
- last 7 days demands of the predicted time window +/- 1 hour - (day-1 T-3, T-2 ... T+9), (day-2 T-3 ... T+9) ... (day-7 T-3, ... T+9)

## Model summary
- Last 7 days features feed into a RNN
- Last 7 days features feed into a CNN
- Last 2 hours features feed into a RNN
- Distance/datetime features feed into a DNN
- Last 7th day features feed into a DNN (act as attention)
- Outputs of all above serve as metadata and feed into another DNN to predict next 5 demands

## Hyperparameter tuning
This is rather tricky and time-consuming. I tried a rather complex model with more nodes and layers, it turned out the complex model converges to bad local minima very easily so I decided to give that up after an overnight hyperparameter tuning job and still couldn't yield any satisfying result. While relying on a lucky seed and learning rate, I was able to hit a relatively decent result using a less complex model.

## Reference
- https://github.com/Arturus/kaggle-web-traffic
- https://gist.github.com/thousandvoices/7d01f366a388516359915a4b090e29d4
- https://github.com/jfpuget/Kaggle/tree/master/WebTrafficPrediction
- http://blog.davidkaleko.com/feature-engineering-cyclical-features.html

