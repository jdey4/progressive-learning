#%%
import random
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
from itertools import product
import pandas as pd

import numpy as np
import pickle
import matplotlib
from sklearn.model_selection import StratifiedKFold
from math import log2, ceil 

import sys
sys.path.append("../src/")
#sys.path.append("../src_mapping_1/")
from lifelong_dnn import LifeLongDNN
from joblib import Parallel, delayed
from multiprocessing import Pool

import tensorflow as tf
# %%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_colors(colors, inds):
    c = [colors[i] for i in inds]
    return c

def generate_2d_rotation(theta=0, acorn=None):
    if acorn is not None:
        np.random.seed(acorn)
    
    R = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    
    return R


def generate_gaussian_parity(n, mean=np.array([-1, -1]), cov_scale=1, angle_params=None, k=1, acorn=None):
    if acorn is not None:
        np.random.seed(acorn)
        
    d = len(mean)
    
    if mean[0] == -1 and mean[1] == -1:
        mean = mean + 1 / 2**k
    
    mnt = np.random.multinomial(n, 1/(4**k) * np.ones(4**k))
    cumsum = np.cumsum(mnt)
    cumsum = np.concatenate(([0], cumsum))
    
    Y = np.zeros(n)
    X = np.zeros((n, d))
    
    for i in range(2**k):
        for j in range(2**k):
            temp = np.random.multivariate_normal(mean, cov_scale * np.eye(d), 
                                                 size=mnt[i*(2**k) + j])
            temp[:, 0] += i*(1/2**(k-1))
            temp[:, 1] += j*(1/2**(k-1))
            
            X[cumsum[i*(2**k) + j]:cumsum[i*(2**k) + j + 1]] = temp
            
            if i % 2 == j % 2:
                Y[cumsum[i*(2**k) + j]:cumsum[i*(2**k) + j + 1]] = 0
            else:
                Y[cumsum[i*(2**k) + j]:cumsum[i*(2**k) + j + 1]] = 1
                
    if d == 2:
        if angle_params is None:
            angle_params = np.random.uniform(0, 2*np.pi)
            
        R = generate_2d_rotation(angle_params)
        X = X @ R
        
    else:
        raise ValueError('d=%i not implemented!'%(d))
       
    return X, Y.astype(int)

# %%
def experiment(n_xor, n_nxor, n_test, n_trees, acorn=None):
    
    if acorn != None:
        np.random.seed(acorn)
    
    l2f = LifeLongDNN(parallel=False)

    xor, label_xor = generate_gaussian_parity(n_xor,cov_scale=0.1,angle_params=0)
    test_xor, test_label_xor = generate_gaussian_parity(n_test,cov_scale=0.1,angle_params=0)
    
    nxor, label_nxor = generate_gaussian_parity(n_nxor,cov_scale=0.1,angle_params=np.pi/2)

    l2f.new_forest(xor, label_xor, n_estimators=n_trees,max_depth=None)
    l2f.new_forest(nxor, label_nxor, n_estimators=n_trees,max_depth=None)

    l2f_task1=l2f.predict(test_xor, representation='all', decider=0)
    
    return 1 - np.mean(l2f_task1 == test_label_xor)

# %%
####main hyperparameters#####
n_trees = 1
reps = 1000
n_xor = (100*np.arange(0.5, 30, step=0.25)).astype(int)
n_nxor = 750
n_test = 1000

#%%
error = np.zeros(len(n_xor),dtype=float)

for i, n1 in enumerate(n_xor):
    err = np.array(
        Parallel(n_jobs=-1,verbose=1)(
        delayed(experiment)(n1,n_nxor,n_test,n_trees=n_trees) for _ in range(reps)
    )
    )
    error[i] = err

with open('result/true_data_res.pickle','wb') as f:
    pickle.dump(error,f)