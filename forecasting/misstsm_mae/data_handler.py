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

from utils.util import Utils
from functools import partial


class ETTHour():
    
    def __init__(self, args, target='OT'):
        
        self.split_ratios = {'train':0.6, 
                             'val':0.2, 
                             'test':0.2}
        self.args = args

    def read_data(self, gt=None):
        
        if gt is not None:
            root_path=self.args.gt_root_path
            data_path=self.args.gt_source_filename
        else:
            root_path=self.args.root_path
            data_path=self.args.source_filename
            
        filepath = os.path.join(root_path, data_path)
        
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

    def read_data(self, gt=None):
        
        if gt is not None:
            root_path=self.args.gt_root_path
            data_path=self.args.gt_source_filename
        else:
            root_path=self.args.root_path
            data_path=self.args.source_filename
            
        filepath = os.path.join(root_path, data_path)
        
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
        
    def read_data(self, gt=None):
        
        if gt is not None:
            root_path=self.args.gt_root_path
            data_path=self.args.gt_source_filename
        else:
            root_path=self.args.root_path
            data_path=self.args.source_filename
        
        filepath = os.path.join(root_path, data_path)
        
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
            'ETTh2': ETTHour,
            'ETTm2': ETTMin,
            'weather': Custom,
        }
        
        self.args = args
        if args.dataset=='ETT':
            self.dataClass = data_map[args.source_filename]
        else:
            self.dataClass = data_map[args.dataset]
            
    def handle(self, gt=None):
        
        self.handler = self.dataClass(self.args)
        
        '''
        read the file
        '''
        df = self.handler.read_data()

        '''
        add time features - if timeenc is set
        '''
        df_X = self.handler.add_time_feats(df)
        
        '''
        initialize utils object
        '''
        utils = Utils(inp_cols=self.handler.features_col, 
                      date_col=self.handler.date_col, 
                      args=self.args,
                      stride=1)
        
        '''
        create train and val set
        '''
        train_df, val_df, test_df = utils.split_data(df_X, self.handler.split_ratios)
        
        '''
        create windowed dataset or load one 
        '''
        data_path = os.path.join(self.args.root_path, self.args.dataset)
        
        train_X = utils.perform_windowing(train_df, data_path, name=self.args.source_filename, split='train')
        val_X = utils.perform_windowing(val_df, data_path, name=self.args.source_filename, split='val')
        test_X = utils.perform_windowing(test_df, data_path, name=self.args.source_filename, split='test')
        
        '''
        standardize the data
        '''
        train_X = torch.from_numpy(train_X).type(torch.Tensor)
        val_X = torch.from_numpy(val_X).type(torch.Tensor)
        test_X = torch.from_numpy(test_X).type(torch.Tensor)

        train_X = utils.normalize_tensor(train_X, use_stat=False)
        val_X = utils.normalize_tensor(val_X, use_stat=True)
        test_X = utils.normalize_tensor(test_X, use_stat=True)
        
        return train_X, val_X, test_X, utils