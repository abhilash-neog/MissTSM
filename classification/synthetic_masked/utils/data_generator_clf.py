from functools import partial

import os
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import math
import argparse
import random
import datetime

from tqdm import tqdm

eps = 1e-6

parser = argparse.ArgumentParser(description='datagenerator')

parser.add_argument('--dataset', type=str, required=True, default='weather', help='dataset type')

args = parser.parse_args()
dataset = args.dataset

path="./data/"

def apply_mcar_with_probability(dataframe, p=0.001):
    """
    Apply MCAR to a pandas DataFrame with a specific probability for missing data.
    
    Parameters:
    - dataframe: pandas DataFrame, your dataset.
    - p: float, probability of data being marked as missing.
    
    Returns:
    - modified_df: pandas DataFrame, dataset with MCAR applied.
    """
    # Ensure the probability is between 0 and 1
    assert 0 <= p <= 1, "Probability must be between 0 and 1."
    
    # Create a mask with the same shape as the dataframe with random values
    mask = np.random.rand(*dataframe.shape) < p
    print(f"percentage = {mask.sum()/(mask.shape[0]*mask.shape[1])}")
    # Create a copy of the dataframe to apply the mask
    modified_df = dataframe.copy()
    
    # Apply the mask and set the selected values to NaN
    modified_df[mask] = np.nan
    
    return modified_df

def mask_data_randomly_torch(X, probability):
    # Generate a random tensor of the same shape as X
    random_mask = torch.rand(X.shape)
    
    # Create a mask where true values represent the data points to be masked
    mask = random_mask < probability
    
    # Clone the original tensor to avoid modifying it directly
    X_masked = X.clone()
    
    # Apply NaN where the mask is True
    X_masked[mask] = torch.tensor(float('nan'))
    
    return X_masked

def perform_spline_imputation(X_masked):
    # Convert each sample's masked tensor to a DataFrame, impute, and collect them back
    imputed_samples = []
    for i in range(X_masked.shape[0]):
        # Extract the sample, convert to numpy, and then to DataFrame
        sample_df = pd.DataFrame(X_masked[i].numpy())

        # Impute missing values using mean imputation for each sample
        imputed_sample_df = sample_df.interpolate(method='spline', order=2, limit_direction='both')

        # Collect the imputed DataFrame
        imputed_samples.append(imputed_sample_df)
    
    df_imputed = pd.concat(imputed_samples, ignore_index=True)
    imputed_array = df_imputed.to_numpy()
    original_shape_array = imputed_array.reshape(X_masked.shape[0], X_masked.shape[1], X_masked.shape[2])
    X_imputed = torch.tensor(original_shape_array)
    return X_imputed

pvalues = [0.2, 0.4, 0.6, 0.8]
for pvalue in tqdm(pvalues):
    
    print(f"For p value = {pvalue} \n")
    
    for trial in range(0, 5):
        fix_seed = random.randint(0, 1000)#seeds[trial]

        random.seed(fix_seed)
        
        np.random.seed(fix_seed)
        print(f"\nTRIAL = {trial} :: Perform masking \n")
        # print(f"Dataset = {dataset}")

        train=torch.load(os.path.join(path,dataset,'train.pt'))
        trainX = train['samples'].transpose(1,2)

        val=torch.load(os.path.join(path,dataset,'val.pt'))
        valX = val['samples'].transpose(1,2)

        test=torch.load(os.path.join(path,dataset,'test.pt'))
        testX = test['samples'].transpose(1,2)

        '''
        train
        '''
        train_masked = mask_data_randomly_torch(trainX, pvalue)
        torch.save({'samples':train_masked.transpose(2,1), 'labels':train['labels']}, os.path.join(path,dataset,'train_p'+str(pvalue)[-1] + '_v' + str(trial) + '.pt'))
        print(f"Train dataset masked")

        train_imputed = perform_spline_imputation(train_masked)
        torch.save({'samples':train_imputed.transpose(2,1), 'labels':train['labels']}, os.path.join(path,dataset,'train_p'+str(pvalue)[-1]+'_spline_imputed_' + 'v' + str(trial) + '.pt'))

        '''
        val
        '''
        val_masked = mask_data_randomly_torch(valX, pvalue)
        torch.save({'samples':val_masked.transpose(2,1), 'labels':val['labels']}, os.path.join(path,dataset,'val_p'+str(pvalue)[-1]+ '_v' + str(trial) + '.pt'))
        print(f"val dataset masked")

        val_imputed = perform_spline_imputation(val_masked)
        torch.save({'samples':val_imputed.transpose(2,1), 'labels':val['labels']}, os.path.join(path,dataset,'val_p'+str(pvalue)[-1]+'_spline_imputed_' + 'v' + str(trial) + '.pt'))

        '''
        test
        '''
        test_masked = mask_data_randomly_torch(testX, pvalue)
        torch.save({'samples':test_masked.transpose(2,1), 'labels':test['labels']}, os.path.join(path,dataset,'test_p'+str(pvalue)[-1] + '_v' + str(trial) + '.pt'))
        print(f"test dataset masked")

        test_imputed = perform_spline_imputation(test_masked)
        torch.save({'samples':test_imputed.transpose(2,1), 'labels':test['labels']}, os.path.join(path,dataset,'test_p'+str(pvalue)[-1]+'_spline_imputed_' + 'v' + str(trial) + '.pt'))