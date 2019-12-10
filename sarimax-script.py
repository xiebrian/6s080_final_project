import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX

# grid search sarima hyperparameters
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from sklearn.metrics import mean_squared_error

df_hospital = pd.read_csv('data/hospital_visits.csv')
df_hospital['Week'] = df_hospital.Week.map(str) + "-" + df_hospital.Year.map(str)
df_hospital = df_hospital[['Week', 'Total ILI']]
df_hospital.columns = ['Week', 'ILI']
df_hospital = df_hospital[28:256]
df_hospital = df_hospital.reset_index(drop=True)

df_twitter = pd.read_csv('data/twitter-flu-data.csv', header=None)
df_twitter.columns = ['Year', 'Week', '1', '2', '3', '4', '5', '6', '7']
df_twitter['Week'] = df_twitter.Week.map(str) + "-" + df_twitter.Year.map(str)
df_twitter['Tweets'] = df_twitter[['1', '2', '3', '4', '5', '6', '7']].sum(axis=1)
df_twitter = df_twitter[['Week', 'Tweets']][27:-1]
df_twitter = df_twitter.reset_index(drop=True)

df_us = pd.read_csv('data/USA_flu_virus_counts.csv')
df_us['Week'] = df_us.Week.map(str) + "-" + df_us.Year.map(str)
df_us = df_us[['Week', 'ALL_INF']]
df_us = df_us[786+27:1042]
df_us = df_us.drop(851)
df_us = df_us.reset_index(drop=True)

df_aus = pd.read_csv('data/AUS_flu_virus_counts.csv')
df_aus['Week'] = df_aus.Week.map(str) + "-" + df_aus.Year.map(str)
df_aus = df_aus[['Week', 'ALL_INF']]
df_aus = df_aus[912:1168-27]
df_aus = df_aus.drop(977)
df_aus = df_aus.reset_index(drop=True)

df_google = pd.read_csv('data/flu_trends_past5years.csv')
df_google['Week'] = df_google.Week.map(str) + "-" + df_google.Year.map(str)
df_google = df_google[['Week', 'Searches']]
df_google = df_google[21:250]
df_google = df_google.drop(86)
df_google.reset_index(drop=True)

list_hospital = list(df_hospital['ILI'])
list_tweets = list(df_twitter['Tweets'])
list_us = list(df_us['ALL_INF'])
list_aus = list(df_aus['ALL_INF'])
list_google = list(df_google['Searches'])


def sarima_forecast(history, config, train_features, test_features):
    order, sorder, trend = config
    # define model
    #     print(len(history))
    #     print(len(train_features))
    model = SARIMAX(history, exog=train_features, order=order, seasonal_order=sorder, trend=trend,
                    enforce_stationarity=False, enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history), exog=test_features)
    return yhat[0]


def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]


def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# walk-forward validation for univariate data
def walk_forward_validation(data, cfg, features):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, 52)
    train_features, test_features = train_test_split(features, 52)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set

    #     print((len(train_features), len(train_features[0])))
    #     print(len(history))

    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = sarima_forecast(history, cfg, train_features, [test_features[i]])
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        train_features.append(test_features[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
    # print(error)
    return error


# score a model, return None on failure
def score_model(data, cfg, features, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data, cfg, features)
    else:
        # # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(data, cfg, features)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return (key, result)

def grid_search(data, cfg_list, features, parallel=True):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, cfg, features) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, cfg, features) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores

# create a set of sarima configs to try
def sarima_configs():
    models = list()
    # define config lists
    p_params = [0, 1, 2]
    d_params = [0, 1]
    q_params = [0, 1, 2]
    P_params = [0, 1, 2]
    D_params = [0, 1]
    Q_params = [0, 1, 2]
    m = 52
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for P in P_params:
                    for D in D_params:
                        for Q in Q_params:
                            cfg = [(p,d,q), (P,D,Q,m), 'c']
                            models.append(cfg)
    return models

# define dataset
data = list_hospital
# data split
n_test = 52
# model configs
cfg_list = sarima_configs()
print("configs done")
# grid search

features = [list_tweets, list_aus, list_google]
features = list(map(list, zip(*features)))

# print((len(features), len(features[0])))
# print(len(data))

scores = grid_search(data, cfg_list, features)
print('done')
# list top 3 configs
for cfg, error in scores[:5]:
    print(cfg, error)
