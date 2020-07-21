#%%
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import seaborn as sns 
import timeit
import numpy as np
import pickle

from sklearn.model_selection import StratifiedKFold
from math import log2, ceil 

import sys
sys.path.append("../src/")
from lifelong_dnn import LifeLongDNN
from joblib import Parallel, delayed

# %%
def generate_parity(low, high, n, d, type='xor',acorn=None):
    r'''
    A function that generates d dimensional parity data 
    with n samples with each dimension sampled as iid , i.e.,
    X1,......,X_p ~ U(low,high)
    '''
    if acorn != None:
        np.random.seed(acorn)

    #loop through each dimension to make them iid
    x = np.random.uniform(
        low=low,high=high,size=(n,1)
        )
    
    for d_ in range(d-1):
        x = np.concatenate(
            (
                x, np.random.uniform(
                low=low,high=high,size=(n,1)
                )
            ),
            axis=1
        )
    
    positive_value_per_sample = np.sum((x>0),axis=1)
    y = positive_value_per_sample%2

    if type =='nxor':
        y = y - (y==1) + (y==0)
    return x, y

# %%
def experiment(n, d, n_test, n_trees, reps, acorn=None):

    if acorn != None:
        np.random.seed(acorn)

    depth = ceil(log2(n))
    xor_err = np.zeros((reps,2),dtype=float)
    nxor_err = np.zeros((reps,2),dtype=float)
    time_elapsed = np.zeros(reps,dtype=float)

    for rep in range(reps):
        #train data
        xor, label_xor = generate_parity(-1,1,n,d)
        nxor, label_nxor = generate_parity(-1,1,n,d,type='nxor')

        #test data
        xor_test, label_xor_test = generate_parity(
            -1,1,n_test,d
            )
        nxor_test, label_nxor_test = generate_parity(
            -1,1,n_test,d,type='nxor'
            )

        start = timeit.timeit()
        l2f = LifeLongDNN(parallel=False)
        l2f.new_forest(
            xor, label_xor, n_estimators=ntrees, max_depth=depth
        )
        end = timeit.timeit()
        time_train_first_task = end-start

        predict_xor = l2f.predict(
            xor_test, representation=0, decider=0
            )
        xor_err[rep,0] = np.mean(predict_xor!=label_xor_test)
        ################################
        start = timeit.timeit()
        l2f.new_forest(
            nxor, label_nxor, n_estimators=ntrees, max_depth=depth
        )
        end = timeit.timeit()
        time_elapsed[rep] = time_train_first_task + (end-start)

        predict_xor = l2f.predict(
            xor_test, representation='all', decider=0
            )
        nxor_err[rep,1] = np.mean(predict_xor!=label_xor_test)
        ################################
        predict_nxor = l2f.predict(
            nxor_test, representation=1, decider=1
            )
        nxor_err[rep,0] = np.mean(predict_xor!=label_nxor_test)
        ################################
        predict_nxor = l2f.predict(
            nxor_test, representation='all', decider=1
            )
        nxor_err[rep,1] = np.mean(predict_xor!=label_nxor_test)
    return np.mean(xor_err,axis=0), np.mean(nxor_err,axis=0), np.std(xor_err,ddof=1,axis=0), np.std(nxor_err,ddof=1,axis=0), np.mean(time_elapsed)

# %%
#main hyperparameters#
#######################
n = 1000
n_test = 1000
n_trees = 10
reps = 100
max_dim = 1000
# %%
result = np.array(
        Parallel(n_jobs=-1,verbose=1)(
        delayed(experiment)(n, d, n_test, n_trees, reps, acorn=d) for d in range(2,max_dim)
        )
    )

with open('result/parity_without replay.pickle', 'wb') as f:
    pickle.dump(result,f)