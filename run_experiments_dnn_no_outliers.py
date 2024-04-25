import pandas as pd
import numpy as np
import pickle
from copy import copy
import os
import random

from adaptive_standardisation import adaptive_standardisation_no_outliers

from epftoolbox.data import read_data
from epftoolbox.evaluation import MAE, sMAPE
from epftoolbox.models import DNN

import concurrent.futures
        

datasets = ["DE_2023_no_outliers.csv", "SP_2023_no_outliers.csv", "FR_no_outliers.csv", "BE_no_outliers.csv", "NP_no_outliers.csv"]
apply_adaptive_standardisation_list = [True, False]

path_datasets_folder = "Data"
path_hyperparameters_folder = "experimental_files"
path_recalibration_folder = "Results_py"

nlayers = 2
shuffle_train = 1
data_augmentation = 0


def process_combination(combination):
    print(combination)
    apply_adaptive_standardisation, dataset, calibration_window, experiment_id = combination[0], combination[1], combination[2], combination[3]
    df = pd.read_csv(f"Data//{dataset}")
    df['Date'] = pd.to_datetime(df.Date)

    if dataset in ["BE_no_outliers.csv", "FR_no_outliers.csv"]:
        begin_test_date = "04/01/2015 00:00"
        end_test_date = "31/12/2016 23:00"
        
    elif dataset in ["DE_2023_no_outliers.csv", "SP_2023_no_outliers.csv"] :
        begin_test_date = "01/01/2022 00:00"
        end_test_date = "31/05/2023 23:00"
    elif dataset in ["NP_no_outliers.csv"] :
        begin_test_date = "27/12/2016 00:00"
        end_test_date = "24/12/2018 23:00"

    if apply_adaptive_standardisation:
        # original_df = copy(df)
        df['Simple Date'] = df.Date.dt.strftime("%Y-%m-%d")
        df['Hour'] = df.Date.dt.hour
        df.columns = ['Date', 'Price', 'Exogenous 1', 'Exogenous 2', 'Price_no_outliers', 'Simple Date', 'Hour']
        try:
            with open(f'dicts_as_py//dataset_{dataset.replace(".csv", "")}.pkl', 'rb') as f:
                dict_new_df = pickle.load(f)
        except:
            dict_new_df = adaptive_standardisation_no_outliers(df, window_size=7)
            with open(f'dicts_as_py//dataset_{dataset.replace(".csv", "")}.pkl', 'wb') as f:
                pickle.dump(dict_new_df, f)
        df = pd.DataFrame(dict_new_df)[['Date', 'Price', 'Exogenous 1', 'Exogenous 2', 'Price_no_outliers']]
        df.to_csv(f"Data//{dataset.replace('.csv', '_as.csv')}", index=False)
        df['Date'] = pd.to_datetime(df.Date)
        df_scalers = pd.DataFrame({'Date':dict_new_df['Date'], 'scaler':dict_new_df['scaler']})
    df = df.set_index('Date')
    df.drop('Price', axis=1, inplace=True)
    df.columns = ['Exogenous 1', 'Exogenous 2', 'Price']
    if apply_adaptive_standardisation:
        # original_df = original_df.set_index('Date')
        # original_df.columns = ['Price', 'Exogenous 1', 'Exogenous 2']
        dataset_name = dataset.replace('.csv', '_as')
    else:
        dataset_name = dataset.replace('.csv', '')

    df_train, df_test = read_data(dataset=dataset_name, path=path_datasets_folder,
                                    begin_test_date=begin_test_date, end_test_date=end_test_date)
    
    # df_train.drop('Exogenous 3', axis=1, inplace=True)
    # df_test.drop('Exogenous 3', axis=1, inplace=True)

    forecast = pd.DataFrame(index=df_test.index[::24], columns=['h' + str(k) for k in range(24)])
    real_values = df_test.loc[:, ['Price']].values.reshape(-1, 24)
    real_values = pd.DataFrame(real_values, index=forecast.index, columns=forecast.columns)
        
    forecast_dates = forecast.index

    model = DNN(
            experiment_id=experiment_id, path_hyperparameter_folder=path_hyperparameters_folder, nlayers=nlayers, 
            dataset=dataset_name, shuffle_train=shuffle_train, data_augmentation=data_augmentation, calibration_window=calibration_window)

    # For loop over the recalibration dates
    for date in forecast_dates:

        # For simulation purposes, we assume that the available data is
        # the data up to current date where the prices of current date are not known
        data_available = pd.concat([df_train, df_test.loc[:date + pd.Timedelta(hours=23), :]], axis=0)

        # We set the real prices for current date to NaN in the dataframe of available data
        data_available.loc[date:date + pd.Timedelta(hours=23), 'Price'] = np.NaN

        # Recalibrating the model with the most up-to-date available data and making a prediction
        # for the next day
        Yp = model.recalibrate_and_forecast_next_day(df=data_available, next_day_date=date)

        # Saving the current prediction
        forecast.loc[date, :] = Yp

        # Computing metrics up-to-current-date
        mae = np.mean(MAE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values)) 
        smape = np.mean(sMAPE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values)) * 100

        # Pringint information
        print('{} - sMAPE: {:.2f}%  |  MAE: {:.3f}'.format(str(date)[:10], smape, mae))
    if apply_adaptive_standardisation:
        forecast.to_csv(f"Results_py//dataset_{dataset.replace('.csv', '')}model_DNN_as_calibration_window_{calibration_window}_experiment_id_{experiment_id}.csv")
        df_test_original = pd.DataFrame(forecast.values.reshape((-1,)), columns=['Price'])
        df_test_original['Date'] = df_scalers.tail(len(df_test_original)).Date.to_numpy()

        predictions = np.array([])
        for cont in range(0, len(df_test_original)):
            scaler_obj = df_scalers[df_scalers.Date == df_test_original.iloc[cont].Date].scaler.values[0]
            predictions = np.append(predictions, scaler_obj.inverse_transform(np.array(df_test_original.iloc[cont].Price).reshape(1, -1))[0][0])
                
        df_test_original['Price'] = predictions
        df_test_original = df_test_original.set_index(df_test_original.Date).drop('Date', axis = 1)

        forecast_final = df_test_original.loc[:, ['Price']].values.reshape(-1, 24)
        forecast_final = pd.DataFrame(forecast_final, index=df_test_original.index[::24], columns=['h' + str(k) for k in range(24)])

        forecast_final.to_csv(f"Results_py//dataset_{dataset.replace('.csv', '')}model_DNN_as_calibration_window_{calibration_window}_experiment_id_{experiment_id}.csv")
    else:
        forecast.to_csv(f"Results_py//dataset_{dataset.replace('.csv', '')}model_DNN_calibration_window_{calibration_window}_experiment_id_{experiment_id}.csv")


combinations = []

for apply_adaptive_standardisation in apply_adaptive_standardisation_list:
    for dataset in datasets:
        if apply_adaptive_standardisation:
            dataset_name = dataset.replace('.csv', '_as')
        else:
            dataset_name = dataset.replace('.csv', '')
        for experiment_id in range(1, 5):
            if dataset in ["BE_no_outliers.csv", "FR_no_outliers.csv", "NP._no_outliers.csv"]:
                if apply_adaptive_standardisation:
                    calibration_window=None
                else:
                    calibration_window=4
            elif dataset in ["DE_2023_no_outliers.csv", "SP_2023_no_outliers.csv"]:
                if apply_adaptive_standardisation:
                    calibration_window=None
                else:
                    calibration_window=3

            if apply_adaptive_standardisation:
                file_name_forecast = f"dataset_{dataset.replace('.csv', '')}model_DNN_as_calibration_window_{calibration_window}_experiment_id_{experiment_id}.csv"
            else:
                file_name_forecast = f"dataset_{dataset.replace('.csv', '')}model_DNN_calibration_window_{calibration_window}_experiment_id_{experiment_id}.csv"

            if not os.path.isfile("Results_py//" + file_name_forecast):
                combinations.append([apply_adaptive_standardisation, dataset, calibration_window, experiment_id])
            else:
                print("Combination already processed: ", apply_adaptive_standardisation, dataset, calibration_window, experiment_id)

# with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
#     executor.map(process_combination, combinations)


random.shuffle(combinations)

for combination in combinations:
    # print(combination)
    process_combination(combination)


