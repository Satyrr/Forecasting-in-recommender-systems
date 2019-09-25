import os, zipfile
import urllib.request
import pandas as pd
import numpy as np
import surprise
from sklearn.model_selection import KFold


def get_movielens(dataset_name):
    
    filenames = {
    '1m': 'ml-1m.zip',
    '20m': 'ml-20m.zip',
    'latest': 'ml-latest.zip' ,
    'latest-small': 'ml-latest-small.zip',
    }

    filename = filenames[dataset_name]
    file_path = 'datasets/{}'.format(filename)
    folder_path = file_path.split('.')[0]
    link = 'http://files.grouplens.org/datasets/movielens/{}'.format(filename)

    if not os.path.exists('datasets'):
        os.makedirs('datasets')

    if not os.path.isfile(file_path):
        urllib.request.urlretrieve(link, file_path)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall('datasets')

    return DataSet(folder_path, dataset_name)

class CachedData(object):

    def __init__(self, forecaster, test_df):
        self.f_ratings = forecaster.ratings
        self.movie_forecast_models = forecaster.f_models
        self.movie_forecast_cached_predictions = forecaster.f_cached_predictions
        self.forecasted_movie_ids = forecaster.f_movies
        self.means_dict = forecaster.means_dict
        self.test_df = test_df

class DataSet(object):
    
    def __init__(self, folder_path, dataset_name):
        self.dataset_name = dataset_name
        self.fill_from_file(folder_path)

    def fill_from_file(self, folder_path):

        if self.dataset_name == '1m':
            self.movies = pd.read_csv('{}/movies.dat'.format(folder_path), sep='::', engine='python', header=None, names=['movieId', 'title', 'category'])
            self.ratings = rename_to_prophet(pd.read_csv('{}/ratings.dat'.format(folder_path), sep='::', engine='python', 
                header=None, names=['userId', 'movieId', 'rating', 'timestamp']))
        else:
            self.movies = pd.read_csv('{}/movies.csv'.format(folder_path))
            self.ratings = rename_to_prophet(pd.read_csv('{}/ratings.csv'.format(folder_path)))
            self.tags = pd.read_csv('{}/tags.csv'.format(folder_path)) 

        self.ratings = self.ratings.sort_values(by='ds').reset_index(drop=True)

def to_surprise_trainset(data, rating_scale=(0.5, 5.0)):
    data = data.iloc[:, :3]
    reader = surprise.Reader(rating_scale=rating_scale)
    return surprise.Dataset.load_from_df(data, reader=reader).build_full_trainset()

def to_surprise_testset(data, rating_scale=(0.5, 5.0)):
    data = data.iloc[:, :3]
    reader = surprise.Reader(rating_scale=rating_scale)
    return surprise.Dataset.load_from_df(data, reader=reader).build_full_trainset().build_testset()

def rename_to_prophet(rates):
    if 'y' in rates.columns and 'ds' in rates.columns:
        return rates
    
    rates['timestamp'] = pd.to_datetime(rates['timestamp'], unit='s')

    return rates.rename(index=str, columns={"rating": "y", "timestamp": "ds"})

def split_dataset(dataset, offset_ratio, train_ratio=0.5, test_years=None, test_ratio=0.3, ratings_order='random',
    random_seed=1):
    if ratings_order == 'random':
        dataset = dataset.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    offset = int(dataset.shape[0] * offset_ratio)
    train_n = int(dataset.shape[0] * train_ratio)
    test_n = int(dataset.shape[0] * test_ratio)
    
    train_data_raw = dataset.iloc[offset:offset+train_n, :].copy()
    
    if not test_years is None:
        last_train_date = dataset.iloc[offset:offset+train_n, :]['ds'].max()
        last_test_date = last_train_date + pd.DateOffset(years=test_years)
        test_data_raw = dataset[dataset['ds'].between(last_train_date, last_test_date)].copy()
    else:
        test_data_raw = dataset.iloc[offset+train_n:offset+train_n+test_n, :].copy()
    
    return train_data_raw, test_data_raw

def split_dataset_for_forecasting(dataset, train_ratio=0.75, timebin=3, drop_ratio=0.8):
    filtered_ratings = filter_dataset_for_forecasting(dataset.copy(), timebin=timebin, drop_ratio=drop_ratio )
    idx_permutation = np.random.permutation(filtered_ratings.ds.count())
    train_count = int(idx_permutation.shape[0] * train_ratio)

    train_idx, test_idx = idx_permutation[:train_count], idx_permutation[train_count:]

    return filtered_ratings.iloc[train_idx], filtered_ratings.iloc[test_idx]

def filter_dataset_for_forecasting(dataset, timebin=2, drop_ratio=0.8):
    dataset = add_timebins_by_item(dataset, 10)
    ratings_to_filter_count = len(dataset[dataset.timebin > timebin].index)
    ratings_to_leave_indexes = np.random.permutation(ratings_to_filter_count)[:int((1.0-drop_ratio)*ratings_to_filter_count)]
    ratings_to_leave = dataset[dataset.timebin > timebin].iloc[ratings_to_leave_indexes]
    
    return pd.concat([dataset[dataset.timebin <= 2], ratings_to_leave])

def add_timebins_by_item(data, time_bins_num=10):
  total_days_for_movie = data.groupby('movieId').ds.max() - data.groupby('movieId').ds.min()
  total_days_for_movie = total_days_for_movie.dt.days.reset_index()
  total_days_for_movie['min_date'] = data.groupby('movieId').ds.min().reset_index(drop=True)
  
  total_days_for_ratings = total_days_for_movie.merge(data, on='movieId')
  total_days_for_ratings['timebin'] = ((total_days_for_ratings.ds_y - total_days_for_ratings.min_date).dt.days / (total_days_for_ratings.ds_x+1)) * time_bins_num
  total_days_for_ratings['timebin'] = total_days_for_ratings['timebin'].astype('long')

  return data.merge(total_days_for_ratings.loc[:,['movieId', 'userId', 'timebin']], on=['movieId', 'userId'])