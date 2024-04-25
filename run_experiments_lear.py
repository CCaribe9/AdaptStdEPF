print("Toca pandas")
import pandas as pd
print("Numpy")
import numpy as np
print("Datetime")
import datetime as dt
# from tqdm import tqdm
print("Pickle")
import pickle
print("Copy")
from copy import copy
# print("Sys")
# import sys
import os

print("adaptive_standardisation")
from adaptive_standardisation import adaptive_standardisation

print("epftoolbox.models")
from epftoolbox.models import LEAR
print("epftoolbox.models._lear ")
from epftoolbox.models._lear import LEAR_adaptive_normalization as LEAR_as
print("epftoolbox.evaluation")
from epftoolbox.evaluation import MAE, sMAPE

import concurrent.futures

# print("warnings")
# import warnings
# warnings.filterwarnings(action='ignore')



def process_combination(combination):
    apply_adaptive_standardisation = combination[0]
    dataset = combination[1]
    calibration_window = combination[2]
    df = pd.read_csv(f"Data//{dataset}")
    df['Date'] = pd.to_datetime(df.Date)

    if dataset in ["BE.csv", "FR.csv"]:
        date_test = dt.datetime(2015, 1, 4)

    elif dataset in ["DE_2023.csv", "SP_2023.csv"] :
        date_test = dt.datetime(2022, 1, 1)
    
    elif dataset in ["NP.csv"] :
        date_test = dt.datetime(2016, 12, 27)

    if apply_adaptive_standardisation:
        original_df = copy(df)
        df['Simple Date'] = df.Date.dt.strftime("%Y-%m-%d")
        df['Hour'] = df.Date.dt.hour
        df.columns = ['Date', 'Price', 'Exogenous 1', 'Exogenous 2', 'Simple Date', 'Hour']
        try:
            with open(f'dicts_as_py//dataset_{dataset.replace(".csv", "")}.pkl', 'rb') as f:
                dict_new_df = pickle.load(f)
        except:
            dict_new_df = adaptive_standardisation(df, window_size=7)
            with open(f'dicts_as_py//dataset_{dataset.replace(".csv", "")}.pkl', 'wb') as f:
                pickle.dump(dict_new_df, f)
        df = pd.DataFrame(dict_new_df)[['Date', 'Price', 'Exogenous 1', 'Exogenous 2']]
        df['Date'] = pd.to_datetime(df.Date)
        df_scalers = pd.DataFrame({'Date':dict_new_df['Date'], 'scaler':dict_new_df['scaler']})
    df = df.set_index('Date')
    df.columns = ['Price', 'Exogenous 1', 'Exogenous 2']
    if apply_adaptive_standardisation:
        original_df = original_df.set_index('Date')
        original_df.columns = ['Price', 'Exogenous 1', 'Exogenous 2']

    df_train = df[df.index < date_test]
    df_test= df[df.index >= date_test]

    forecast = pd.DataFrame(index=df_test.index[::24], columns=['h' + str(k) for k in range(24)])
    if apply_adaptive_standardisation:
        real_values = original_df[original_df.index >= date_test].loc[:, ['Price']].values.reshape(-1, 24)
    else:
        real_values = df_test.loc[:, ['Price']].values.reshape(-1, 24)
    real_values = pd.DataFrame(real_values, index=forecast.index, columns=forecast.columns)
    forecast_dates = forecast.index
    if False:
        if apply_adaptive_standardisation:
            model = LEAR_as(calibration_window=calibration_window)
        else:
            model = LEAR(calibration_window=calibration_window)

        # For loop over the recalibration dates
        for date in forecast_dates: #tqdm(forecast_dates[:10], file=sys.stdout):

            # For simulation purposes, we assume that the available data is
            # the data up to current date where the prices of current date are not known
            data_available = pd.concat([df_train, df_test.loc[:date + pd.Timedelta(hours=23), :]], axis=0)

            # We set the real prices for current date to NaN in the dataframe of available data
            data_available.loc[date:date + pd.Timedelta(hours=23), 'Price'] = np.NaN

            # Recalibrating the model with the most up-to-date available data and making a prediction
            # for the next day
            if apply_adaptive_standardisation:
                scalers = df_scalers[(df_scalers.Date >= date) & (df_scalers.Date <= date + pd.Timedelta(hours=23))].scaler.to_numpy()
                Yp = model.recalibrate_and_forecast_next_day(df=data_available, next_day_date=date,
                                                        calibration_window=calibration_window, scalers=scalers)
            else:
                Yp = model.recalibrate_and_forecast_next_day(df=data_available, next_day_date=date,
                                                            calibration_window=calibration_window)
            # Saving the current prediction
            forecast.loc[date, :] = Yp

            # Computing metrics up-to-current-date
            mae = np.mean(MAE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values))
            smape = np.mean(sMAPE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values)) * 100

            # Pringint information
            # print('\r\033[2K\033[1G', end='', flush=True) # TQDM compatibility
            print('{} - sMAPE: {:.2f}%  |  MAE: {:.3f}'.format(str(date)[:10], smape, mae))
        if apply_adaptive_standardisation:
            forecast.to_csv(f"Results_py//dataset_{dataset.replace('.csv', '')}model_LEAR_as_calibration_window{calibration_window}.csv")
        else:
            forecast.to_csv(f"Results_py//dataset_{dataset.replace('.csv', '')}model_LEAR_calibration_window{calibration_window}.csv")
    forecast = pd.DataFrame(index=df_test.index[::24], columns=['h' + str(k) for k in range(24)])

    if apply_adaptive_standardisation and not os.path.isfile(f"Results_py//dataset_{dataset.replace('.csv', '')}_model_LEAR_as_calibration_window_None.csv"):
        model = LEAR_as(calibration_window=None)
        # For loop over the recalibration dates
        for date in forecast_dates: #tqdm(forecast_dates, desc='Calibration Window None'):

            # For simulation purposes, we assume that the available data is
            # the data up to current date where the prices of current date are not known
            data_available = pd.concat([df_train, df_test.loc[:date + pd.Timedelta(hours=23), :]], axis=0)

            # We set the real prices for current date to NaN in the dataframe of available data
            data_available.loc[date:date + pd.Timedelta(hours=23), 'Price'] = np.NaN

            # Recalibrating the model with the most up-to-date available data and making a prediction
            # for the next day
            scalers = df_scalers[(df_scalers.Date >= date) & (df_scalers.Date <= date + pd.Timedelta(hours=23))].scaler.to_numpy()
            Yp = model.recalibrate_and_forecast_next_day(df=data_available, next_day_date=date,
                                                    calibration_window=calibration_window, scalers=scalers)

            # Saving the current prediction
            forecast.loc[date, :] = Yp

            # Computing metrics up-to-current-date
            mae = np.mean(MAE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values))
            smape = np.mean(sMAPE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values)) * 100

            # Pringint information
            # print('\r\033[2K\033[1G', end='', flush=True) # TQDM compatibility
            print('{} - sMAPE: {:.2f}%  |  MAE: {:.3f}'.format(str(date)[:10], smape, mae))

        forecast.to_csv(f"Results_py//dataset_{dataset.replace('.csv', '')}_model_LEAR_as_calibration_window_None.csv")
        forecast = pd.DataFrame(index=df_test.index[::24], columns=['h' + str(k) for k in range(24)])
    else:
        print("Combination already processed: ", apply_adaptive_standardisation, dataset, None)
    
    if not apply_adaptive_standardisation and not os.path.isfile(f"Results_py//dataset_{dataset.replace('.csv', '')}model_LEAR_calibration_window_None.csv"):
        model = LEAR(calibration_window=None)
        # For loop over the recalibration dates
        for date in forecast_dates: #tqdm(forecast_dates, desc='Calibration Window None'):

            # For simulation purposes, we assume that the available data is
            # the data up to current date where the prices of current date are not known
            data_available = pd.concat([df_train, df_test.loc[:date + pd.Timedelta(hours=23), :]], axis=0)

            # We set the real prices for current date to NaN in the dataframe of available data
            data_available.loc[date:date + pd.Timedelta(hours=23), 'Price'] = np.NaN

            # Recalibrating the model with the most up-to-date available data and making a prediction
            # for the next day
            Yp = model.recalibrate_and_forecast_next_day(df=data_available, next_day_date=date,
                                                        calibration_window=calibration_window)

            # Saving the current prediction
            forecast.loc[date, :] = Yp

            # Computing metrics up-to-current-date
            mae = np.mean(MAE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values))
            smape = np.mean(sMAPE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values)) * 100

            # Pringint information
            print('{} - sMAPE: {:.2f}%  |  MAE: {:.3f}'.format(str(date)[:10], smape, mae))

        forecast.to_csv(f"Results_py//dataset_{dataset.replace('.csv', '')}model_LEAR_calibration_window_None.csv")
        forecast = pd.DataFrame(index=df_test.index[::24], columns=['h' + str(k) for k in range(24)])
    else:
        print("Combination already processed: ", apply_adaptive_standardisation, dataset, None)


print("Starting")
datasets = ["SP_2023.csv", "DE_2023.csv", "BE.csv", "FR.csv", "NP.csv"]
apply_adaptive_standardisation_list = [True, False]
calibration_windows_1 = [56, 84, 1092, 1456]
calibration_windows_2 = [56, 84, 364, 728]

combinations = []

for apply_adaptive_standardisation in apply_adaptive_standardisation_list:
    for dataset in datasets:
        if dataset in ["BE.csv", "FR.csv", "NP.csv"]:
            calibration_windows = calibration_windows_1
        elif dataset in ["DE_2023.csv", "SP_2023.csv"]:
            calibration_windows = calibration_windows_2
        for calibration_window in calibration_windows:
            if apply_adaptive_standardisation:
                file_name = f"dataset_{dataset.replace('.csv', '')}model_LEAR_as_calibration_window{calibration_window}.csv"
            else:
                file_name = f"dataset_{dataset.replace('.csv', '')}model_LEAR_calibration_window{calibration_window}.csv"
            if not os.path.isfile("Results_py//" + file_name):
                combinations.append([apply_adaptive_standardisation, dataset, calibration_window])
            else:
                print("Combination already processed: ", apply_adaptive_standardisation, dataset, calibration_window)

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    executor.map(process_combination, combinations)

# for combination in combinations:
#     process_combination(combination)