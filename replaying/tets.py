#%%
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from itertools import product
import pandas as pd

import numpy as np
import pickle

from sklearn.model_selection import StratifiedKFold
from math import log2, ceil 

import sys
sys.path.append("../src/")
from lifelong_dnn import LifeLongDNN
from joblib import Parallel, delayed
from multiprocessing import Pool

import tensorflow as tf

#%%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#%%
def cross_val_data(data_x, data_y, num_points_per_task, total_task=10, shift=1):
    x = data_x.copy()
    y = data_y.copy()
    idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]
    
    batch_per_task=5000//num_points_per_task
    sample_per_class = num_points_per_task//total_task
    test_data_slot=100//batch_per_task

    for task in range(total_task):
        for batch in range(batch_per_task):
            for class_no in range(task*10,(task+1)*10,1):
                indx = np.roll(idx[class_no],(shift-1)*100)
                
                if batch==0 and class_no==0 and task==0:
                    train_x = x[indx[batch*sample_per_class:(batch+1)*sample_per_class],:]
                    train_y = y[indx[batch*sample_per_class:(batch+1)*sample_per_class]]
                    test_x = x[indx[batch*test_data_slot+500:(batch+1)*test_data_slot+500],:]
                    test_y = y[indx[batch*test_data_slot+500:(batch+1)*test_data_slot+500]]
                else:
                    train_x = np.concatenate((train_x, x[indx[batch*sample_per_class:(batch+1)*sample_per_class],:]), axis=0)
                    train_y = np.concatenate((train_y, y[indx[batch*sample_per_class:(batch+1)*sample_per_class]]), axis=0)
                    test_x = np.concatenate((test_x, x[indx[batch*test_data_slot+500:(batch+1)*test_data_slot+500],:]), axis=0)
                    test_y = np.concatenate((test_y, y[indx[batch*test_data_slot+500:(batch+1)*test_data_slot+500]]), axis=0)
            
    return train_x, train_y, test_x, test_y

#%%
### MAIN HYPERPARAMS ###
model = "uf"
num_points_per_task = 500
noise_size = 5000000
#delta = 0.001
########################
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
data_x = np.concatenate([X_train, X_test])
if model == "uf":
    data_x = data_x.reshape((data_x.shape[0], data_x.shape[1] * data_x.shape[2] * data_x.shape[3]))

data_x = data_x/np.max(data_x)

data_y = np.concatenate([y_train, y_test])
data_y = data_y[:, 0]


#%%
slot_fold = range(5000//num_points_per_task)
shift_fold = range(1,7,1)
n_trees=[10]
iterable = product(n_trees,shift_fold,slot_fold)


shift=1
slot=0
train_x, train_y, test_x, test_y = cross_val_data(data_x, data_y, num_points_per_task, shift=1)

df = pd.DataFrame()
single_task_accuracies = np.zeros(10,dtype=float)
shifts = []
tasks = []
base_tasks = []
accuracies_across_tasks = []
ntrees = 10

'''for task_ii in range(2):
    if model == "uf":
        single_task_learner = LifeLongDNN(model = "uf", parallel = False)

        np.random.seed(12345)


        single_task_learner.new_forest(
                train_x[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task,:],
                 train_y[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task], 
                max_depth=ceil(log2(num_points_per_task)), n_estimators=ntrees
                )

        llf_task=single_task_learner.predict(
                test_x[task_ii*1000:(task_ii+1)*1000,:], representation=0, decider=0
                )
        single_task_accuracies[task_ii] = np.mean(
                    llf_task == test_y[task_ii*1000:(task_ii+1)*1000]
                    )'''

lifelong_forest = LifeLongDNN(model = model, parallel = False)
for task_ii in range(10):
    print("Starting Task {} For Fold {}".format(task_ii, shift))
    np.random.seed(12345)

    lifelong_forest.new_forest(
        train_x[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task,:], 
        train_y[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task], 
        max_depth=ceil(log2(num_points_per_task)), n_estimators=ntrees
        )

    #excite with noise
    #np.random.seed(12345*task_ii)
    '''dims = train_x.shape[1]
    mu = np.mean(lifelong_forest.X_across_tasks[task_ii][:,0])
    sigma = np.var(lifelong_forest.X_across_tasks[task_ii][:,0])
    noise = np.random.normal(mu, sigma, noise_size).reshape(noise_size,1)

    for ii in range(dims-1):
        mu = np.mean(lifelong_forest.X_across_tasks[task_ii][:,ii+1])
        sigma = np.var(lifelong_forest.X_across_tasks[task_ii][:,ii+1])
        noise = np.concatenate(
            (noise, np.random.normal(mu, sigma, noise_size).reshape(noise_size,1)),
            axis = 1
        )'''
    '''noise = np.random.uniform(0,1,(noise_size,train_x.shape[1]))

    print(noise.shape)
    noise_label = lifelong_forest.predict(
         noise, representation=task_ii, decider=task_ii
    )
    print(np.unique(noise_label))

    lifelong_forest.X_across_tasks[task_ii] = noise
    lifelong_forest.y_across_tasks[task_ii] = noise_label'''

        
    for task_jj in range(task_ii+1):
        llf_task=lifelong_forest.predict(
            test_x[task_jj*1000:(task_jj+1)*1000,:], representation='all', decider=task_jj
            )
            
        print(task_ii, task_jj, np.mean(
            llf_task == test_y[task_jj*1000:(task_jj+1)*1000]
            ))
        shifts.append(shift)
        tasks.append(task_jj+1)
        base_tasks.append(task_ii+1)
        accuracies_across_tasks.append(np.mean(
            llf_task == test_y[task_jj*1000:(task_jj+1)*1000]
            ))
            
df['data_fold'] = shifts
df['task'] = tasks
df['base_task'] = base_tasks
df['accuracy'] = accuracies_across_tasks


 # %%
