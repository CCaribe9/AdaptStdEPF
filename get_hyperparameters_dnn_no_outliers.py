import pandas as pd
import pickle
from copy import copy
import os
import gc

from adaptive_standardisation import adaptive_standardisation_no_outliers

from epftoolbox.models import hyperparameter_optimizer

import warnings
warnings.filterwarnings(action='ignore')

import random

datasets = ["DE_2023_no_outliers.csv", "SP_2023_no_outliers.csv", "FR_no_outliers.csv", "BE_no_outliers.csv", "NP_no_outliers.csv"]
apply_adaptive_standardisation_list = [False, True]

path_datasets_folder = "Data"
path_hyperparameters_folder = "experimental_files"
path_recalibration_folder = "Results_py"

nlayers = 2
shuffle_train = 1
data_augmentation = 0
new_hyperopt = 1
max_evals = 100

combinations = []


def process_combination(combination):
    dataset_name, calibration_window, experiment_id, begin_test_date, end_test_date = combination[0], combination[1], combination[2], combination[3], combination[4]

    hyperparameter_optimizer(path_datasets_folder=path_datasets_folder, 
                            path_hyperparameters_folder=path_hyperparameters_folder, 
                            new_hyperopt=new_hyperopt, max_evals=max_evals, nlayers=nlayers, dataset=dataset_name, calibration_window=calibration_window, 
                            shuffle_train=shuffle_train, data_augmentation=0, experiment_id=experiment_id,
                            begin_test_date=begin_test_date, end_test_date=end_test_date)


for experiment_id in range(1, 5):
    for apply_adaptive_standardisation in apply_adaptive_standardisation_list:
        for dataset in datasets:
            df = pd.read_csv(f"Data//{dataset}")
            df['Date'] = pd.to_datetime(df.Date)

            if dataset in ["BE_no_outliers.csv", "FR_no_outliers.csv"]:
                calibration_window = 4
                if apply_adaptive_standardisation:
                    calibration_window = None
                begin_test_date = "04/01/2015 00:00"
                end_test_date = "31/12/2016 23:00"
                
            elif dataset in ["DE_2023_no_outliers.csv", "SP_2023_no_outliers.csv"] :
                calibration_window = 3
                if apply_adaptive_standardisation:
                    calibration_window = None
                begin_test_date = "01/01/2022 00:00"
                end_test_date = "31/05/2023 23:00"
            elif dataset in ["NP_no_outliers.csv"] :
                calibration_window = 4
                if apply_adaptive_standardisation:
                    calibration_window = None
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
            
            file_name = 'DNN_hyperparameters_nl' + str(nlayers) + '_dat' + str(dataset_name) + \
            '_YT' + str(2) + '_SF' * (shuffle_train) + \
            '_DA' * (data_augmentation) + '_CW' + str(calibration_window) + \
            '_' + str(experiment_id)

            if os.path.isfile("experimental_files//" + file_name):
                print(file_name + " already computed")
            else:
                combinations.append([dataset_name, calibration_window, experiment_id, begin_test_date, end_test_date])

random.shuffle(combinations)

for combination in combinations:
    print(combination)
    process_combination(combination)



