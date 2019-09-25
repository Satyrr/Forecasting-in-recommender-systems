import os, time, gc
import numpy as np
import pandas as pd
import surprise
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import source.datasets as ds

class Net(nn.Module):
    def __init__(self, trainset, n_features=100, n_hidden=[100, 200], dropout=0.5, embedding_dropout=0.1, time_bins_num=50, time_biased=False):
        super(Net, self).__init__()

        self.u_embeddings = nn.Embedding(trainset.trainset.n_users, n_features)
        self.u_biases = nn.Embedding(trainset.trainset.n_users, 1)
        self.i_embeddings = nn.Embedding(trainset.trainset.n_items, n_features)
        self.i_biases = nn.Embedding(trainset.trainset.n_items, 1)

        self.time_biased = time_biased
        self.i_time_biases = nn.Embedding(trainset.trainset.n_items, time_bins_num+1)

        self.hidden_dropout = nn.Dropout(p=dropout)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.linears = nn.ModuleList()

        in_features = n_features
        out_features = 0

        for i in range(len(n_hidden)):
            out_features = n_hidden[i]
            self.linears.append(nn.Linear(in_features, out_features))
            self.linears.append(self.hidden_dropout)
            self.linears.append(nn.ReLU())

            in_features = out_features

        self.last_layer = nn.Linear(out_features, 1)

        if torch.cuda.is_available():
            self.to(torch.device('cuda:0'))

    def forward(self, x):
        if x.shape[1] == 3:
            user_id, item_id, timebin = x[:, 0].long(), x[:, 1].long(), x[:, 2]
        else:
            user_id, item_id = x[:, 0].long(), x[:, 1].long()

        user_embeddings = self.u_embeddings(user_id)
        item_embeddings = self.i_embeddings(item_id)
        u_bias = self.u_biases(user_id)
        i_bias = self.i_biases(item_id)

        user_embeddings = self.embedding_dropout(user_embeddings)
        item_embeddings = self.embedding_dropout(item_embeddings)

        x = (user_embeddings + item_embeddings)/2.0

        for layer in self.linears:
            x = layer(x)

        x = self.last_layer(x) + u_bias + i_bias

        if self.time_biased:
            i_time_bias = self.i_time_biases(item_id)[np.arange(timebin.shape[0]), timebin].view(-1, 1)
            x = x + i_time_bias

        return x

class NeuralNetworkSR(surprise.AlgoBase):

    def __init__(self, train_df, test_df=None, num_epochs=4, n_features=100, hiddens=[100, 100], dropout=0.5, embedding_dropout=0.1, lr=0.001, wd=1e-1, max_time=0.0,
        batch_size=10000, log_freq=10, biased=False, time_bins=50, time_biased=False):

        self.train_df = train_df
        self.test_df = test_df
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.hiddens = hiddens
        self.n_features = n_features
        self.lr = lr
        self.wd = wd
        self.max_time = max_time
        self.batch_size = batch_size
        self.log_freq = log_freq
        self.biased = biased
        self.time_biased = time_biased
        self.time_bins = time_bins

    def fit(self, trainset):
       
        surprise.AlgoBase.fit(self, trainset)
        
        pytorch_trainset = SurpriseTorchTrainset(trainset, self.train_df)
        self.train_net(pytorch_trainset)

        pytorch_testset = SurpriseTorchTestset(trainset, ds.to_surprise_testset(self.test_df))
        if self.test_df is not None:
            self.estimate_all(pytorch_testset)

        print("Estimated all")

        return self

    def train_net(self, trainset):
        self.net = Net(trainset, self.n_features, self.hiddens, self.dropout, embedding_dropout=self.embedding_dropout, time_biased=self.time_biased,
        time_bins_num=self.time_bins)

        criterion = nn.MSELoss(reduction='sum')
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        start_time = time.time()

        global_mean = self.trainset.global_mean
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            
            # collect garbage sometimes...
            if epoch % 10 == 0:
                gc.collect()

            if self.max_time < time.time() - start_time:
                break

            running_loss = 0.0
            for i, data in enumerate(batchify_numpy(trainset.data, batch_size=self.batch_size)):
                data = torch.from_numpy(data)
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[:, :-1].long(), data[:, -1].float()

                if torch.cuda.is_available():
                    inputs = inputs.to(torch.device('cuda:0'), dtype=torch.long)
                    labels = labels.to(torch.device('cuda:0'), dtype=torch.float)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = global_mean + self.net(inputs)

                loss = criterion(outputs.view(-1), labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item() / float(len(outputs))
                if i % self.log_freq == self.log_freq-1:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / self.log_freq))
                    running_loss = 0.0

        self.u_biases = self.net.u_biases.weight.view(-1).tolist()
        self.i_biases = self.net.i_biases.weight.view(-1).tolist()

        if self.time_biased:
            self.i_time_biases = self.net.i_time_biases.weight.tolist()
            self.timebin_dict = self.test_df.set_index(['userId', 'movieId']).to_dict()['timebin']

        optimizer.zero_grad()
        print('Finished Training')

    def estimate_all(self, testset):
        self.est_dicts = {}
        self.net.eval()
        with torch.no_grad():
            if self.time_biased:
                data = np.stack([testset.user_ids, testset.item_ids, self.test_df.timebin.to_numpy()[np.array(testset.ids)]], axis=1)
            else:
                data = np.stack([testset.user_ids, testset.item_ids, ], axis=1)
            labels = []

            for x in batchify_numpy(data, batch_size=self.batch_size, shuffle=False):
                x = to_device(torch.tensor(x, requires_grad=False))
                labels.append(self.net(x))

            labels = torch.cat(labels).view(-1).tolist()

            for idx in range(len(labels)):
                u = testset.user_ids[idx]
                i = testset.item_ids[idx]
                self.est_dicts[(u, i)] = labels[idx]

    def estimate(self, u, i):
        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            return self.trainset.global_mean + self.est_dicts[(u, i)]

        if self.trainset.knows_user(u):
            return self.trainset.global_mean + self.u_biases[u]
        if self.trainset.knows_item(i):
            result = self.trainset.global_mean + self.i_biases[i]
            
            if self.time_biased:
                raw_u = int(u[5:])
                raw_i = self.trainset.to_raw_iid(i)
                return result + self.i_time_biases[i][self.timebin_dict[(raw_u,raw_i)]]
            else:
                return result

        return self.trainset.global_mean

class SurpriseTorchTrainset(object):

    def __init__(self, trainset, trainset_df):
        self.n_ratings = trainset.n_ratings
        self.trainset = trainset

        inner_uids = trainset_df['userId'].apply(lambda x : trainset.to_inner_uid(x)).to_numpy()
        inner_iids = trainset_df['movieId'].apply(lambda x : trainset.to_inner_iid(x)).to_numpy()
        ratings = trainset_df.iloc[:,2].to_numpy()

        if 'timebin' in trainset_df:
            time_bins = trainset_df['timebin'].to_numpy()
            self.data = np.hstack([inner_uids[:,None], inner_iids[:,None], time_bins[:, None], ratings[:,None]]).astype(np.float64)
        else:
            self.data = np.hstack([inner_uids[:,None], inner_iids[:,None], ratings[:,None]]).astype(np.float64)


class SurpriseTorchTestset(object):

    def __init__(self, trainset, testset):
        self.data = np.array(testset)
        self.n_ratings = len(self.data)
        self.user_ids, self.item_ids, self.ids = [], [], []
        id_num = 0
        for x in testset:
            try:
                u = trainset.to_inner_uid(x[0])
                i = trainset.to_inner_iid(x[1])
                self.user_ids.append(u)
                self.item_ids.append(i)
                self.ids.append(id_num)
            except:
                pass
            id_num += 1

def to_device(tensor):
    if torch.cuda.is_available():
        return tensor.to(torch.device('cuda:0'))
    else:
        return tensor

def batchify_numpy(data, batch_size=10000, shuffle=True):
    count = data.shape[0]
    idxs = np.random.permutation(count) if shuffle else np.arange(count)
    start_idx = 0

    while start_idx < count:
        yield data[idxs[start_idx:start_idx+batch_size]]
        start_idx += batch_size