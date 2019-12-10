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
list_hospital = list(df_hospital['ILI'])


def sarima_forecast(history, config):
    order, sorder, trend = config
    # define model
    #     print(len(history))
    #     print(len(train_features))
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend,
                    enforce_stationarity=False, enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]


def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]


def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# walk-forward validation for univariate data
def walk_forward_validation(data, cfg):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, 52)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set

    #     print((len(train_features), len(train_features[0])))
    #     print(len(history))

    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = sarima_forecast(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
    # print(error)
    return error


# score a model, return None on failure
def score_model(data, cfg, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data, cfg)
    else:
        # # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(data, cfg)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return (key, result)

def grid_search(data, cfg_list, parallel=True):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, cfg) for cfg in cfg_list]
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

scores = grid_search(data, cfg_list)
print('done')
# list top 3 configs
for cfg, error in scores[:5]:
    print(cfg, error)
