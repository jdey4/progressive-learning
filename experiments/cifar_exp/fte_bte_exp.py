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
sys.path.append("../../src_mapping_2/")
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
def LF_experiment(train_x, train_y, test_x, test_y, ntrees, shift, slot, model, num_points_per_task, acorn=None):
       
    df = pd.DataFrame()
    single_task_accuracies = np.zeros(10,dtype=float)
    shifts = []
    tasks = []
    base_tasks = []
    accuracies_across_tasks = []

    for task_ii in range(10):
        if model == "uf":
            single_task_learner = LifeLongDNN(model = "uf", parallel = True)

            if acorn is not None:
                np.random.seed(acorn)

            single_task_learner.new_forest(
                train_x[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task,:],
                 train_y[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task], 
                max_depth=ceil(log2(num_points_per_task)), n_estimators=ntrees*(task_ii+1)
                )
            llf_task=single_task_learner.predict(
                test_x[task_ii*1000:(task_ii+1)*1000,:], representation=0, decider=0
                )
            single_task_accuracies[task_ii] = np.mean(
                    llf_task == test_y[task_ii*1000:(task_ii+1)*1000]
                    )

    lifelong_forest = LifeLongDNN(model = model, parallel = True if model == "uf" else False)
    for task_ii in range(10):
        print("Starting Task {} For Fold {}".format(task_ii, shift))
        if acorn is not None:
            np.random.seed(acorn)

        lifelong_forest.new_forest(
            train_x[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task,:], 
            train_y[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task], 
            max_depth=ceil(log2(num_points_per_task)), n_estimators=ntrees
            )
        if model == "dnn":
            llf_task=lifelong_forest.predict(
                test_x[task_ii*1000:(task_ii+1)*1000,:], representation=task_ii, decider=task_ii
                )
            single_task_accuracies[task_ii] = np.mean(
                    llf_task == test_y[task_ii*1000:(task_ii+1)*1000]
                    )
        
        for task_jj in range(task_ii+1):
            llf_task=lifelong_forest.predict(
                test_x[task_jj*1000:(task_jj+1)*1000,:], representation='all', decider=task_jj
                )
            
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

    df_single_task = pd.DataFrame()
    df_single_task['task'] = range(1, 11)
    df_single_task['data_fold'] = shift
    df_single_task['accuracy'] = single_task_accuracies

    summary = (df,df_single_task)
    file_to_save = 'result/result/'+model+str(num_points_per_task)+'_'+str(ntrees)+'_'+str(shift)+'_'+str(slot)+'.pickle'
    with open(file_to_save, 'wb') as f:
        pickle.dump(summary, f)

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
def run_parallel_exp(data_x, data_y, n_trees, model, num_points_per_task, slot=0, shift=1):
    train_x, train_y, test_x, test_y = cross_val_data(data_x, data_y, num_points_per_task, shift=shift)
    
    if model == "dnn":
        #with tf.device('/gpu:'+str(shift % 4)):
        #    LF_experiment(train_x, train_y, test_x, test_y, n_trees, shift, model, num_points_per_task, acorn=12345)
        LF_experiment(train_x, train_y, test_x, test_y, n_trees, shift, slot, model, num_points_per_task, acorn=12345)
    else:
        LF_experiment(train_x, train_y, test_x, test_y, n_trees, shift, slot, model, num_points_per_task, acorn=12345)

#%%
### MAIN HYPERPARAMS ###
model = "uf"
num_points_per_task = 5000
########################

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
data_x = np.concatenate([X_train, X_test])
if model == "uf":
    data_x = data_x.reshape((data_x.shape[0], data_x.shape[1] * data_x.shape[2] * data_x.shape[3]))
data_y = np.concatenate([y_train, y_test])
data_y = data_y[:, 0]


#%%
if model == "uf":
    slot_fold = range(5000//num_points_per_task)
    shift_fold = range(1,7,1)
    n_trees=[40]
    iterable = product(n_trees,shift_fold,slot_fold)
    Parallel(n_jobs=-2,verbose=1)(
        delayed(run_parallel_exp)(
                data_x, data_y, ntree, model, num_points_per_task, slot=slot, shift=shift
                ) for ntree,shift,slot in iterable
                )
elif model == "dnn":
    slot_fold = range(5000//num_points_per_task)
    shift_fold = range(1,7,1)
    iterable = product(shift_fold,slot_fold)

    for shift,slot in iterable:
        run_parallel_exp(data_x, data_y, 0, model, num_points_per_task, slot=slot, shift=shift)

'''elif model == "dnn":
    
    def perform_shift(shift):
        return run_parallel_exp(data_x, data_y, 0, model, num_points_per_task, slot=slot, shift=shift)
    
    print("Performing Stage 1 Shifts")
    stage_1_shifts = range(1, 5)
    with Pool(4) as p:
        p.map(perform_shift, stage_1_shifts) 
    
    print("Performing Stage 2 Shifts")
    stage_2_shifts = range(5, 7)
    with Pool(4) as p:
        p.map(perform_shift, stage_2_shifts) '''
    
        

# %%
