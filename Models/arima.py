import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

from datetime import date, timedelta, datetime
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def pre_process(region: int) -> (list, list):
    """
    Returns the train and test data as a tuple of two lists.
    """
    df = pd.read_csv('260_weeks_data.csv')
    df.replace(r'^\s*$', np.nan, regex=True)
    df = df.fillna(0)
    #df.loc[((1909 >= df['row']) & (df['row'] >= 1380)), 'Week of'] = np.nan
    df = df.drop([i for i in range(1380, 1910)])

    df = df[df['Region'] == region]
    week_of = df.loc[:, 'Week of']
    illnesses = df.loc[:, 'Regional Illnesses']

    train_pct = 0.8
    illnesses = list(illnesses)[::-1]
    week_of = list(week_of)[::-1]
    N = len(illnesses)

    print(N)

    train = illnesses[:int(train_pct * N)] 
    train_week_of = week_of[:int(train_pct * N)]
    test = illnesses[int(train_pct * N):]
    test_week_of = week_of[int(train_pct * N):]

    return train, test, train_week_of, test_week_of

def fit_model(train: list, test: list, train_week_of: list, test_week_of: list, 
              region: int, order=(0, 0, 0), save_plot=False, verbose=False) -> float:
    """
    Returns the RMSE of the trained model, optionally saves the prediction plot and prints out the RMSE value.
    """
    N = len(train) + len(test)

    history = [x for x in train]
    predictions = []

    for t in range(len(test)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)

    rmse = mean_squared_error(test, predictions, squared=False)

    if verbose:
        print(f'Test RMSE: {rmse}')

    if save_plot:
        times = [i for i in range(len(train), N)]
        test_week_of = [datetime.strptime(d,'%m/%d/%Y').date() for d in test_week_of]
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.plot(test_week_of, test, label=f'Actual')
        plt.plot(test_week_of, predictions, label=f'Predicted')
        plt.gcf().autofmt_xdate()
        plt.xlabel('Time')
        plt.ylabel('Regional Illness Count')
        plt.title(f'ARIMA Model w/ Order: {order}, Actual vs. Predicted')
        plt.legend()
        plt.savefig(f'ARIMA_Plot_Region{region}_{order}')
    
    return rmse


def grid_search(region: int):
    """
    Performs a grid search on the three parameters of the ARIMA model: (p, d, q)
    """
    p_values = [0, 1, 2, 4, 6, 8, 10]
    d_values = [0, 1, 2]
    q_values = [0, 1, 2]

    orders = [(p, d, q) for p in p_values for d in d_values for q in q_values]

    train, test, d1, d2 = pre_process(region)

    min_rmse = np.inf
    best_order = -1

    for order in tqdm(orders):
        try:
            rmse = fit_model(train, test, d1, d2, region, order=order)

            if rmse < min_rmse:
                best_order = order
                min_rmse = rmse

                print(f'UPDATE: best order = {best_order} with RMSE {min_rmse}')
        except:
            print(f'Failed to create model for order: {order}')
    
    print(f'Best order = {best_order} with RMSE {min_rmse}')

    
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # best order for wrong region 4: (10, 0, 2)

    # best order for region 10: (10, 0, 1)

    region_to_analyze = 10
    pre_process(region_to_analyze)

    # Uncomment the two lines below to run the model
    #train, test, train_week_of, test_week_of = pre_process(region_to_analyze)
    #fit_model(train, test, train_week_of, test_week_of, region_to_analyze, 
    #          order=(10, 0, 1), save_plot=True, verbose=True)
    
    # Uncomment the line below to run a grid search
    # grid_search(region_to_analyze)

    
