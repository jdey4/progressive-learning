#%%
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import seaborn as sns 

import numpy as np
import pickle

from sklearn.model_selection import StratifiedKFold
from math import log2, ceil 

import sys
from lifelong_dnn import LifeLongDNN

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
xor, label_xor = generate_gaussian_parity(100,cov_scale=0.1,angle_params=0)
test_xor, test_label_xor = generate_gaussian_parity(1000,cov_scale=0.1,angle_params=0)

#nxor = xor
#label_nxor = (label_xor==0)*1
nxor, label_nxor = generate_gaussian_parity(10,cov_scale=0.1,angle_params=np.pi/2)
test_nxor, test_label_nxor = generate_gaussian_parity(1000,cov_scale=0.1,angle_params=np.pi/2)

min_xor = np.min(xor)
xor = (xor - min_xor)
max_xor = np.max(xor)
xor = xor/max_xor

min_nxor = np.min(nxor)
nxor = (nxor - min_nxor)
max_nxor = np.max(nxor)
nxor = nxor/max_nxor

test_xor = (test_xor-min_xor)/max_xor
#test_nxor = (test_nxor-min_nxor)/max_nxor

l2f = LifeLongDNN(parallel=False)
#np.random.seed(12345)
l2f.new_forest(xor, label_xor, n_estimators=10,max_depth=100)
#np.random.seed(12345)
l2f.new_forest(nxor, label_nxor, n_estimators=10,max_depth=100)

l2f_task1 = l2f.predict(test_xor, representation='all', decider=0)
uf_task1 = l2f.predict(test_xor, representation=0, decider=0)

print(np.mean(uf_task1 == test_label_xor))
print(np.mean(l2f_task1 == test_label_xor))
 # %%
