import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt


######### Models ###########

# Each model has two methods : fit(data) and predict(X).
# fit methods take a data parameter which is pandas dataframe with two columns: ds and y.
# ds is datetime of a sample and y is a value a of sample. 

# Baseline forecast methods
class LastSample(object):

    def __init__(self):
        pass

    def fit(self, data):

        self.data = data.sort_values(by='ds')
        self.data['date'] = self.data.ds.dt.date

        self.min_date, self.max_date = self.data['date'].iloc[0], self.data['date'].iloc[-1]

        #fill date gaps 
        self.data = pd.DataFrame({'date':pd.date_range(self.min_date, self.max_date).date}).merge(self.data, on='date', how='left')
        #fill rating gaps
        self.data.y = self.data.y.fillna(method='ffill').fillna(method='bfill')

        # first and last values in dataset for dates beyond the daterange
        self.first_val, self.last_val = self.data.y.iloc[0], self.data.y.iloc[-1], 

    def predict(self, X):
        X.loc[:,'date'] = X.ds.dt.date

        merged = X.merge(self.data, on='date', how='left')
        X['yhat'] = merged['y'].fillna(method='ffill').fillna(method='bfill').values
        
        if np.isnan(X['yhat'].iloc[0]):
            if X['date'].iloc[0] < self.min_date:
                X['yhat'] = self.first_val
            else:
                X['yhat'] = self.last_val

        return X[['yhat', 'ds']]

class SimpleMean(object):

    def __init__(self):
        pass

    def fit(self, data):

        self.data = data.sort_values(by='ds')

    def predict(self, X):

        X['yhat'] = 0.0
        for index, x in X.iterrows():
            X.loc[index, 'yhat'] = self.data[self.data.ds <= x['ds']].loc[:,'y'].mean()

        return X

class ExponentialSmoothing(object):

    def __init__(self):
        pass

    def fit(self, data):

        self.data = data.sort_values(by='ds')

    def predict(self, X):
        fit = sm.tsa.ExponentialSmoothing(self.data['y'].values ,seasonal_periods=10 ,trend='add', seasonal='add').fit()
        X['yhat'] = fit.forecast(len(X))
        return X

class Evaluator(object):

    def __init__(self, model, model_creator=None):
        self.model = model
        self.model_creator = model_creator

    def eval(self, test_data=None, verbose=True):
        self.test_data = test_data = test_data.copy()

        self.y_truth = y_truth = test_data.y.copy()

        test_data['y'] = np.NaN
        self.y_predicted = y_predicted = self.model.predict(test_data).yhat
        
        self.rmse = sqrt(mean_squared_error(y_truth, y_predicted))
        self.mape = (np.abs(100.0*(y_predicted - y_truth))/y_truth).mean()

        if verbose:
            print('Model evaluation:\nRMSE: {0}\nMAPE: {1}%\n\n'.format(self.rmse, self.mape))

    def eval_many(self, ratings_list, train_factor=0.8, plot=False):

        total_rmse = 0.0

        for ratings in ratings_list:
            train_data, test_data = split_by_date(ratings)

            if not self.model_creator is None:
                self.model = self.model_creator()

            self.model.fit(train_data)
            self.eval(test_data, verbose=False)
            if plot:
                self.plot(train_data)
                
            total_rmse += self.rmse

        self.rmse = total_rmse
        print('Total rmse = {}'.format(total_rmse))

    def plot(self, X_train=None, train_prediction=False):
        plt.figure(figsize=(10,10))

        if not X_train is None:
            plt.plot(X_train['ds'], X_train['y'], label='train datapoints', color='b')
        if train_prediction:
            self.train_y_predicted = self.model.predict(X_train).yhat
            plt.plot(X_train['ds'], self.model.predict(X_train).yhat, label='train datapoints model prediction')


        plt.plot(self.test_data['ds'], self.y_predicted, label='test prediction', color='y')
        plt.plot(self.test_data['ds'], self.y_truth, label='test truth', color='g')
        plt.legend()
        plt.ylim(1.0, 5.0)
        plt.title('Model evaluation')

class RatingsAnalyzer(object):

    def __init__(self, ratings):
        self.ratings = ratings.copy()
        self.ratings['month'] = pd.to_datetime(self.ratings['ds'], unit='s').dt.month
        self.ratings['day'] = pd.to_datetime(self.ratings['ds'], unit='s').dt.dayofweek

    def box_plot(self, period='month'):

        self.ratings[['y', period]].boxplot(by=period)
        plt.show()

    def seasonal_plot(self, period='month'):
        df = self.ratings.groupby(period)['y'].agg(['mean', 'count'])
        df['mean'].plot()

def split(ts, train_factor=0.8):
    train_size = int(ts.y.count() * train_factor)
    return ts.loc[:train_size,:].reset_index(drop=True).copy(), ts.loc[train_size:,:].reset_index(drop=True).copy()

def split_by_date(ts, test_date='01/06/2012'):
    test_votes = pd.to_datetime(ts['ds'], unit='s') > pd.to_datetime(test_date)
    return ts[test_votes == False].reset_index(drop=True).copy(), ts[test_votes].reset_index(drop=True).copy()

#### Ratings aggregating

def aggregate_by_days(data, days_num):
    data = data.loc[:, ['ds', 'y']].copy()
    data.loc[:,'ds'] = pd.to_datetime(data.loc[:,'ds'], unit='s')
    
    return data.set_index('ds').resample('{}D'.format(days_num)).y \
        .agg(['mean', 'count']).reset_index().rename(columns={ 'mean':'y'})

def plot_aggregated_ratings(aggregated, timespan_days, count_bars=True, title=""):
    handles = plt.plot(aggregated.loc[:, 'ds'], aggregated.loc[:, 'y'], color='green', label='{} day time bin average'.format(timespan_days))
    plt.xlabel('time')
    plt.ylabel('mean rating')
    plt.ylim(1.0, 5.0)

    if count_bars:
        ax2 = plt.gca().twinx()
        bars = ax2.bar(aggregated.loc[:, 'ds'].values, aggregated.loc[:, 'count'].values, alpha=0.3,
                       width=timespan_days, label='#ratings in time bin', color='b')
        ax2.set_ylabel('count', color='b')
        handles.append(bars)
    else:
        ax2 = plt.gca()
    # legend of plot and histogram
    labels = [l.get_label() for l in handles]
    ax2.legend(handles, labels)
    
    plt.title(title)
        
def smooth_aggregated(aggregated, halflife, min_count=5, min_cum_count=1500):
    aggregated = aggregated.loc[aggregated['count'] >= min_count].copy()
    
    aggregated.y = aggregated.y.ewm(halflife=halflife).mean()

    aggregated['cum_count'] = aggregated['count'].cumsum()
    aggregated = aggregated[aggregated['cum_count'] > min_cum_count]
    
    return aggregated

def get_by_id(ts, item_id):
    return ts[ts['movieId'] == item_id].copy()
    
def get_smooth_by_id(ts, item_id, timespan=1, halflife=40, min_count=1, min_cum_count=0):
    aggregated_ts = aggregate_by_days(get_by_id(ts, item_id), timespan)
    smooth_ts = smooth_aggregated(aggregated_ts, halflife, min_count=min_count, min_cum_count=min_cum_count)
    smooth_ts['ds'] = pd.to_datetime(smooth_ts['ds'])
    return smooth_ts

def get_smooth(ts, timespan=1, halflife=40, min_count=1, min_cum_count=0):
    smooth_ts = aggregate_by_days(ts, timespan)
    smooth_ts = smooth_aggregated(smooth_ts, halflife, min_count=min_count, min_cum_count=min_cum_count)
    
    return smooth_ts

def to_mean_deviations(ratings):
    movies_means = ratings.groupby('movieId')['y'].mean().to_frame().rename(columns={'y':'mean_rating'})
    ratings_with_mean = ratings.merge(movies_means.reset_index(), on='movieId')
    ratings_with_mean['y'] = ratings_with_mean['y'] - ratings_with_mean['mean_rating']

    return ratings_with_mean