import pandas as pd
import numpy as np
import datetime as dt
# from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings(action='ignore')

def adaptive_standardisation(df, window_size=7):
    print('Adaptive standardisation...')
    columns = list(df.columns)
    dict_new_df = {col: np.array([]) for col in columns}
    columns.remove("Date")
    columns.remove("Simple Date")
    columns.remove("Hour")
    dict_new_df['scaler'] = np.array([])

    for idx, row in df.iterrows():#tqdm(df.iterrows(), total=len(df), desc='Performing adaptive standardisation'):
        if idx >= window_size*24:
            dict_new_df['Date'] = np.append(dict_new_df['Date'], row['Date'])
            dict_new_df['Simple Date'] = np.append(dict_new_df['Simple Date'], row['Simple Date'])
            dict_new_df['Hour'] = np.append(dict_new_df['Hour'], row['Hour'])

            df_aux = df[(pd.to_datetime(df['Simple Date']) < pd.to_datetime(row['Simple Date'])) & (pd.to_datetime(df['Simple Date']) >= pd.to_datetime(row['Simple Date']) - dt.timedelta(days=window_size))]

            for col in columns:
                df_aux_col = df_aux[col].to_numpy().reshape(-1, 1)
                scaler = StandardScaler()
                scaler.fit(df_aux_col)
                if col == 'Price':
                    dict_new_df['scaler'] = np.append(dict_new_df['scaler'], scaler)
                dict_new_df[col] = np.append(dict_new_df[col], scaler.transform(np.array(row[col]).reshape(1, -1))[0][0])
    return dict_new_df

def adaptive_standardisation_no_outliers(df, window_size=7):
    print('Adaptive standardisation...')
    columns = list(df.columns)
    dict_new_df = {col: np.array([]) for col in columns}
    columns.remove("Date")
    columns.remove("Simple Date")
    columns.remove("Hour")
    dict_new_df['scaler'] = np.array([])

    for idx, row in df.iterrows():#tqdm(df.iterrows(), total=len(df), desc='Performing adaptive standardisation'):
        if idx >= window_size*24:
            dict_new_df['Date'] = np.append(dict_new_df['Date'], row['Date'])
            dict_new_df['Simple Date'] = np.append(dict_new_df['Simple Date'], row['Simple Date'])
            dict_new_df['Hour'] = np.append(dict_new_df['Hour'], row['Hour'])

            df_aux = df[(pd.to_datetime(df['Simple Date']) < pd.to_datetime(row['Simple Date'])) & (pd.to_datetime(df['Simple Date']) >= pd.to_datetime(row['Simple Date']) - dt.timedelta(days=window_size))]

            for col in columns:
                df_aux_col = df_aux[col].to_numpy().reshape(-1, 1)
                scaler = StandardScaler()
                scaler.fit(df_aux_col)
                if col == 'Price_no_outliers':
                    dict_new_df['scaler'] = np.append(dict_new_df['scaler'], scaler)
                dict_new_df[col] = np.append(dict_new_df[col], scaler.transform(np.array(row[col]).reshape(1, -1))[0][0])
    return dict_new_df
 