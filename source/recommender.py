import numpy as np
import pandas as pd
import surprise
import source.forecast as ft
import source.datasets as ds

class MeanPredictorRecommedner(surprise.AlgoBase):
    # Gets predicted item mean for given timestamp
    # If the item is unknown, then it returns global mean of trainset
    def item_mean(self, id, timestamp=None):
        if self.trainset.knows_item(id):
            raw_id = self.trainset.to_raw_iid(id)
            mean = self.forecasting_model.item_mean(raw_id, timestamp)
        else:
            mean = self.forecasting_model.global_mean
        return mean

    def raw_iid(self, id):
        if self.trainset.knows_item(id):
            return self.trainset.to_raw_iid(id)
        else:
            return int(id[5:])
    
    def raw_uid(self, id):
        if self.trainset.knows_user(id):
            return self.trainset.to_raw_uid(id)
        else:
            return int(id[5:])

class MeanDeviationsRecommender(MeanPredictorRecommedner):

    def __init__(self, base_model, forecasting_model, timestamps_dict):

        self.base_model = base_model
        self.forecasting_model = forecasting_model
        self.ratings_to_timestamps_dict = timestamps_dict

    def fit(self, trainset):
        
        surprise.AlgoBase.fit(self, trainset)
        deviations_trainset = ds.to_surprise_trainset(self.forecasting_model.ratings[['userId', 'movieId', 'y_dev']], rating_scale=(-5.0, 5.0))
        self.base_model.fit(deviations_trainset)

        return self

    def estimate(self, u, i):
        raw_i = self.raw_iid(i)
        raw_u = self.raw_uid(u)

        timestamp = None
        if (raw_u, raw_i) in self.ratings_to_timestamps_dict:
            timestamp = self.ratings_to_timestamps_dict[(raw_u, raw_i)]

        deviation_est = self.base_model.predict(raw_u, raw_i).est

        mean_est = self.item_mean(i, timestamp)
        return mean_est + deviation_est 

class ForecastAdjusterRecommender(MeanPredictorRecommedner):

    def __init__(self, base_model, forecasting_model, timestamps_dict):

        self.base_model = base_model
        self.forecasting_model = forecasting_model
        self.ratings_to_timestamps_dict = timestamps_dict

    def fit(self, trainset):
        
        surprise.AlgoBase.fit(self, trainset)
        deviations_trainset = ds.to_surprise_trainset(self.forecasting_model.ratings[['userId', 'movieId', 'y']], rating_scale=(1.0, 5.0))
        self.base_model.fit(deviations_trainset)

        return self

    def estimate(self, u, i):
        raw_i = self.raw_iid(i)
        raw_u = self.raw_uid(u)

        timestamp = None
        if (raw_u, raw_i) in self.ratings_to_timestamps_dict:
            timestamp = self.ratings_to_timestamps_dict[(raw_u, raw_i)]

        base_est = self.base_model.predict(raw_u, raw_i).est

        mean_est = self.item_mean(i, timestamp)
        if raw_i in self.forecasting_model.means_dict:
            item_mean = self.forecasting_model.means_dict[raw_i]
        else:
            item_mean = self.forecasting_model.global_mean

        return base_est + mean_est - item_mean

class RatingsForecaster(object):

    def __init__(self, forecast_model_class):
        self.forecast_model_class = forecast_model_class
        self.f_models = {}
        self.f_cached_predictions = {}

    # compute estimations of mean of popular movies for given timestamps for efficiency
    def precompute_means(self, ts):
        for m_id in self.f_movies:
            predictions = self.f_models[m_id].predict((ts[ts['movieId'] == m_id][['ds']]).copy())

            self.f_cached_predictions[m_id] = predictions.set_index('ds').to_dict()['yhat']

    # compute deviations from predicted mean for each movie
    def compute_devs(self, ratings, forecasted_movies=[]):
        self.f_movies = forecasted_movies
        self.global_mean = ratings['y'].mean()

        movie_means = ratings.groupby('movieId')['y'].mean()
        self.means_dict = movie_means.to_dict()

        # Movie rating deviations from their mean
        ratings_with_means = ratings.merge(pd.DataFrame(movie_means), right_index=True, left_on='movieId').sort_index()['y_y']
        ratings['y_dev'] = ratings['y'] - ratings_with_means

        # Popular movies deviations
        for idx, movie_id in enumerate(self.f_movies):
            movie_ratings_smoothed = ft.get_smooth_by_id(ratings,
                movie_id,
                timespan=10, 
                halflife=50, 
                min_count=5, 
                min_cum_count=0)
            self.f_models[movie_id] = self.forecast_model_class()
            self.f_models[movie_id].fit(movie_ratings_smoothed.loc[:,['ds', 'y']])

            movie_mask = ratings['movieId']==movie_id
            dates_to_predict = ratings.loc[movie_mask, ['ds']]
            forecasted_means = self.f_models[movie_id].predict(dates_to_predict)['yhat']
            ratings.loc[movie_mask ,'y_dev'] = ratings.loc[movie_mask ,'y'].values - forecasted_means#self.means_dict[movie_id]
            
            print('forecasting movie {} ({}/{})'.format(movie_id, idx, len(self.f_movies)))
        
        self.ratings = ratings

    def item_mean(self, movie_id, timestamp=None):
        # forecasted movie mean
        if movie_id in self.f_movies and timestamp:
            if movie_id in self.f_cached_predictions:
                mean = self.f_cached_predictions[movie_id][timestamp]
            else:
                ts = pd.DataFrame({'ds':[timestamp]})
                mean = self.f_models[movie_id].predict(ts)['yhat'].iloc[0]
        # global movie mean
        elif movie_id in self.means_dict:
            mean = self.means_dict[movie_id]
        else:
            mean = self.global_mean
        
        
        return mean

    def load_from_cached_data(self, forecasting_data):
        self.f_models = forecasting_data.movie_forecast_models
        self.f_cached_predictions = forecasting_data.movie_forecast_cached_predictions
        self.f_movies = forecasting_data.forecasted_movie_ids
        self.means_dict = forecasting_data.means_dict
        self.ratings = forecasting_data.f_ratings
        self.global_mean = self.ratings['y'].mean()
