import source.datasets as ds
import source.recommender as rcm
import source.evaluation as evl
import source.forecast as ft
import pickle
import gc
import surprise
import numpy as np
from sklearn.model_selection import KFold

def cross_split(df, k=5, shuffle=True, seed=10):
  np.random.seed(seed)
  torch.manual_seed(seed)
  count = len(df.index)
  idxs = np.arange(count)
    
  kf = KFold(n_splits=k, shuffle=shuffle)  
  
  for train_index, test_index in kf.split(idxs):
    yield df.iloc[train_index], df.iloc[test_index]

def cross_validate(model_constructor, data, k=5, seed=10, splitter=cross_split):
  np.random.seed(seed)
  torch.manual_seed(seed)

  total_rmse = 0.0
  total_ndpm = 0.0
  total_mae = 0.0
  total_fcp = 0.0

  for train_df, test_df in splitter(data, k=k):
    gc.collect()
    
    model = model_constructor(train_df, test_df)
    
    train_dataset = ds.to_surprise_trainset(train_df)
    test_dataset = ds.to_surprise_testset(test_df)

    model.fit(train_dataset)
      
    predictions = model.test(test_dataset)

    total_ndpm += evl.NDPM(predictions)
    total_rmse += surprise.accuracy.rmse(predictions)
    total_mae += surprise.accuracy.mae(predictions)
    total_fcp += surprise.accuracy.fcp(predictions)

  print('Average RMSE = {}'.format(total_rmse/k))
  print('Average MAE = {}'.format(total_mae/k))
  print('Average NDPM = {}'.format(total_ndpm/k))
  print('Average FCP = {}'.format(total_fcp/k))
  

def create_cached_cross_validation_data(movielens_data, filename, k, popular_movies_num=4000, seed=10):
    np.random.seed(seed)
    idx = 0
    for train_df, test_df in ds.cross_split(movielens_data.ratings, k, True):
        most_popular_movies = ds.get_popular_movies_ids(train_df, movielens_data, popular_movies_num)

        forecaster = rcm.RatingsForecaster(ft.LastSample)
        forecaster.compute_devs(train_df, most_popular_movies.movieId.tolist())
        forecaster.precompute_means(test_df)

        with open(filename + str(idx), 'wb') as output:
            pickle.dump(ds.CachedData(forecaster, test_df), output, pickle.HIGHEST_PROTOCOL)
        idx += 1

def cross_validate_cached(model_constructor, data_filenames):
  total_rmse = 0.0
  total_ndpm = 0.0
  total_mae = 0.0
  total_fcp = 0.0

  for filename in data_filenames:
    with open(filename, 'rb') as file:
      gc.collect()
      data = pickle.load(file)
      
      train_df = data.f_ratings
      test_df = data.test_df
      
      forecaster = rcm.RatingsForecaster(ft.LastSample)
      forecaster.load_from_cached_data(data)
      
      model = model_constructor(train_df, test_df, forecaster)
      
      surprise_trainset = ds.to_surprise_trainset(train_df)
      model.fit(surprise_trainset)

      predictions = model.test(ds.to_surprise_testset(test_df))
      
      # collect garbage before evaluating predictions to avoid out of memory error
      del data
      del train_df
      del test_df
      del forecaster
      del model
      del surprise_trainset
      gc.collect()
      
      total_ndpm += evl.NDPM(predictions)
      total_rmse += surprise.accuracy.rmse(predictions)
      total_mae += surprise.accuracy.mae(predictions)
      total_fcp += surprise.accuracy.fcp(predictions)

  k = len(data_filenames)
  print('Average RMSE = {}'.format(total_rmse/k))
  print('Average MAE = {}'.format(total_mae/k))
  print('Average NDPM = {}'.format(total_ndpm/k))
  print('Average FCP = {}'.format(total_fcp/k))

def add_timebins(data, time_bins_num=50):
  total_days = (data.ds.max() - data.ds.min()).days 
  data['timebin'] = ((data.ds - data.ds.min()).dt.days / total_days * time_bins_num).astype('long')

def get_timestamps_dict(test_timestamps_df):
    return test_timestamps_df.set_index(['userId', 'movieId']).to_dict()['ds']

def get_popular_movies_ids(data, movielens20m, count):
    rating_counts = data.groupby('movieId')['y'].agg(['count', 'mean']).sort_values('count', ascending=False)
    most_popular_movies = rating_counts.reset_index().merge(movielens20m.movies, on='movieId')[:count].reset_index()
    return most_popular_movies.loc[:, ['movieId', 'title', 'genres', 'count', 'mean']]

def batchify_numpy(data, batch_size=10000, shuffle=True):
    count = data.shape[0]
    idxs = np.random.permutation(count) if shuffle else np.arange(count)
    start_idx = 0
    
    while start_idx < count:
        yield data[idxs[start_idx:start_idx+batch_size]]
        start_idx += batch_size

