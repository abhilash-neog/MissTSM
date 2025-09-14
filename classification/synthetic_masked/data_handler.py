import sys
sys.path.insert(1, './utils/')

import warnings
import argparse
import torch
import random
import numpy as np
import os
import json
import pandas as pd
import math
import datetime
import timefeatures

from utils import Utils
from functools import partial
from sklearn.model_selection import train_test_split

class ETTHour():
    
    def __init__(self, args, target='OT'):
        
        self.split_ratios = {'train':0.6, 
                             'val':0.2, 
                             'test':0.2}
        self.args = args

    def read_data(self):
        
        filepath = os.path.join(self.args.root_path, self.args.dataset, self.args.source_filename)
        
        df = pd.read_csv(filepath+'.csv')

        self.features_col = df.columns[1:]
        self.date_col = df.columns[0]
        
        return df

    def add_time_feats(self, df):
        
        df_date = df[[self.date_col]]
        df_date[self.date_col] = pd.to_datetime(df_date[self.date_col])

        if self.args.timeenc==0:
            df_date['month'] = df_date.date.apply(lambda row: row.month, 1)
            df_date['day'] = df_date.date.apply(lambda row: row.day, 1)
            df_date['weekday'] = df_date.date.apply(lambda row: row.weekday(), 1)
            df_date['hour'] = df_date.date.apply(lambda row: row.hour, 1)
            df_date = df_date.drop(['date'], 1)
        elif self.args.timeenc==1:
            df_date = time_features(pd.to_datetime(df_date['date'].values), freq=self.args.freq)
            df_date = df_date.transpose(1, 0)
        else:
            # No time features
            return df[self.features_col]
        
        df = df[self.features_col]
        df = pd.concat([df, df_date], axis=1)
        return df

class ETTMin():
    
    def __init__(self, args, target='OT'):
        
        self.split_ratios = {'train':0.6, 
                             'val':0.2, 
                             'test':0.2}
        self.args = args

    def read_data(self):
        
        filepath = os.path.join(self.args.root_path, self.args.dataset, self.args.source_filename)
        
        df = pd.read_csv(filepath+'.csv')

        self.features_col = df.columns[1:]
        self.date_col = df.columns[0]
        
        return df

    def add_time_feats(self, df):
        
        df_date = df[[self.date_col]]
        df_date[self.date_col] = pd.to_datetime(df_date[self.date_col])

        if self.args.timeenc==0:
            df_date['month'] = df_date.date.apply(lambda row: row.month, 1)
            df_date['day'] = df_date.date.apply(lambda row: row.day, 1)
            df_date['weekday'] = df_date.date.apply(lambda row: row.weekday(), 1)
            df_date['hour'] = df_date.date.apply(lambda row: row.hour, 1)
            df_date['minute'] = df_date.date.apply(lambda row: row.minute, 1)
            df_date['minute'] = df_date.minute.map(lambda x: x // 15)
            df_date = df_date.drop(['date'], 1)
        elif self.args.timeenc==1:
            df_date = time_features(pd.to_datetime(df_date['date'].values), freq=self.args.freq)
            df_date = df_date.transpose(1, 0)
        else:
            # No time features
            return df[self.features_col]
        
        df = df[self.features_col]
        df = pd.concat([df, df_date], axis=1)
        return df

class Custom():
    
    def __init__(self, args, target='OT'):
        
        self.split_ratios = {'train':0.7, 
                             'val':0.1, 
                             'test':0.2}
        self.args = args
        self.target = target
        
    def read_data(self):
        
        filepath = os.path.join(self.args.root_path, self.args.dataset, self.args.source_filename)
        
        df = pd.read_csv(filepath+'.csv')

        self.features_col = df.columns[1:]
        self.date_col = df.columns[0]
        
        cols = list(df.columns)
        cols.remove(self.target)
        cols.remove('date')
        df = df[['date'] + cols + [self.target]]
        
        return df
    
    def add_time_feats(self, df):
        '''
        not using it correctly
        '''
        
        df_date = df[[self.date_col]]
        df_date[self.date_col] = pd.to_datetime(df_date[self.date_col])
        
        if self.args.timeenc==0:
            df_date['month'] = df_date.date.apply(lambda row: row.month, 1)
            df_date['day'] = df_date.date.apply(lambda row: row.day, 1)
            df_date['weekday'] = df_date.date.apply(lambda row: row.weekday(), 1)
            df_date['hour'] = df_date.date.apply(lambda row: row.hour, 1)
            df_date['minute'] = df_date.date.apply(lambda row: row.minute, 1)
            df_date['minute'] = df_date.minute.map(lambda x: x // 15)
            df_date = df_date.drop(['date'], 1)
        elif self.args.timeenc==1:
            df_date = time_features(pd.to_datetime(df_date['date'].values), freq=self.args.freq)
            df_date = df_date.transpose(1, 0)
        else:
            # No time features
            return df[self.features_col]
        
        df = df[self.features_col]
        df = pd.concat([df, df_date], axis=1)
        return df
        
    
class DataHandler():
    
    def __init__(self, args):
        
        data_map = {
            'ETTh1': ETTHour,
            'ETTh2': ETTHour,
            'ETTm1': ETTMin,
            'ETTm2': ETTMin,
            'weather': Custom,
            'traffic': Custom,
            'electricity': Custom
        }
        
        self.args = args
        if args.dataset=='ETT':
            self.dataClass = data_map[args.source_filename]
        else:
            self.dataClass = data_map[args.dataset]
    
    def read_data(self):
        # path = "./RawData/Physio2012_mega/"
        path = '.'
        
        X_train = np.load(os.path.join(path, 'train_X.npy'))

        y_train = np.load(os.path.join(path, 'train_y.npy'))
        
        X_test = np.load(os.path.join(path, 'test_X.npy'))

        y_test = np.load(os.path.join(path, 'test_y.npy'))
        
        X_val = np.load(os.path.join(path, 'val_X.npy'))

        y_val = np.load(os.path.join(path, 'val_y.npy'))
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def handle(self):
        
        '''
        read the file
        '''
        train_X, val_X, test_X, train_Y, val_Y, test_Y = self.read_data()
        
        '''
        initialize utils object
        '''
        utils = Utils(inp_cols=["0"]*37, 
                      date_col=None, 
                      args=self.args,
                      stride=1)
        
        '''
        standardize the data
        '''
        train_X = torch.from_numpy(train_X).type(torch.Tensor)
        train_Y = torch.from_numpy(train_Y).type(torch.Tensor)
        
        test_X = torch.from_numpy(test_X).type(torch.Tensor)
        test_Y = torch.from_numpy(test_Y).type(torch.Tensor)
        
        val_X = torch.from_numpy(val_X).type(torch.Tensor)
        val_Y = torch.from_numpy(val_Y).type(torch.Tensor)

#         train_X = utils.normalize_tensor(train_X, use_stat=False)
#         test_X = utils.normalize_tensor(test_X, use_stat=True)
        
#         val_ratio = 0.2

#         # Calculate the number of samples for validation
#         num_val_samples = int(val_ratio * len(train_X))

#         # Generate random indices for validation samples
#         val_indices = torch.randperm(len(train_X))[:num_val_samples]

#         # Generate the complement of val_indices for training samples
#         train_indices = torch.tensor(list(set(range(len(train_X))) - set(val_indices.tolist())))

#         # Split train_X and train_Y into training and validation sets
#         # train_X, val_X = train_X[train_indices], train_X[val_indices]
#         val_X = train_X[val_indices]
#         val_Y = train_Y[val_indices]
        
        return {'train_X':train_X, 'train_Y':train_Y, 'val_X':val_X, 'val_Y':val_Y, 'test_X':test_X, 'test_Y':test_Y}, utils