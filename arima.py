import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

from datetime import date, timedelta, datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def pre_process(region: int) -> (list, list):
    """
    Returns the train and test data as a tuple of two lists.
    """
    df = pd.read_csv('260_weeks_data.csv')
    df.replace(r'^\s*$', np.nan, regex=True)
    df.loc[((1909 >= df['row']) & (df['row'] >= 1380)), 'Week of'] = np.nan

    df = df[df['Region'] == region]
    illnesses = df.loc[:, 'Regional Illnesses']

    train_pct = 0.8
    illnesses = list(illnesses)
    N = len(illnesses)

    train = illnesses[:int(train_pct * N)] 
    test = illnesses[int(train_pct * N):]

    return train, test

def fit_model(train: list, test: list, region: int, order=(10, 0, 2), save_plot=False, verbose=False) -> float:
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
        plt.plot(times, test, label=f'Ground Truth Region {region} Illness Count')
        plt.plot(times, predictions, label=f'Predicted')
        plt.xlabel('Week Number')
        plt.ylabel('Regional Illness Count')
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

    train, test = pre_process(region)

    min_rmse = np.inf
    best_order = -1

    for order in tqdm(orders):
        try:
            rmse = fit_model(train, test, region, order=order)

            if rmse < min_rmse:
                best_order = order
                min_rmse = rmse

                print(f'UPDATE: best order = {best_order} with RMSE {min_rmse}')
        except:
            print(f'Failed to create model for order: {order}')
    
    print(f'Best order = {best_order} with RMSE {min_rmse}')

    
if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    region_to_analyze = 4

    # Uncomment the two lines below to run the model
    train, test = pre_process(region_to_analyze)
    fit_model(train, test, region_to_analyze, save_plot=True, verbose=True)
    
    # Uncomment the line below to run a grid search
    #grid_search(region_to_analyze)

    
