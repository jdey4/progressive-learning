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

#%%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_colors(colors, inds):
    c = [colors[i] for i in inds]
    return c

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

#%%
def experiment(n_xor, n_nxor, n_test, d, reps, n_trees, max_depth, acorn=None):
    #print(1)
    if n_xor==0 and n_nxor==0:
        raise ValueError('Wake up and provide samples to train!!!')
    
    if acorn != None:
        np.random.seed(acorn)
    
    errors = np.zeros((reps,5),dtype=float)
    
    for i in range(reps):
        l2f = LifeLongDNN()
        uf = LifeLongDNN()
        #source data
        xor, label_xor = generate_parity(-1, 1, n_xor, d, type='xor')
        test_xor, test_label_xor = generate_parity(-1, 1, n_test, d, type='xor')
    
        #target data
        nxor, label_nxor = generate_parity(-1, 1, n_xor, d, type='nxor')
        test_nxor, test_label_nxor = generate_parity(-1, 1, n_test, d, type='nxor')
    
        if n_xor == 0:
            start = timeit.timeit()
            l2f.new_forest(nxor, label_nxor, n_estimators=n_trees,max_depth=max_depth)
            end = timeit.timeit()

            errors[i,0] = 0.5
            errors[i,1] = 0.5
            
            uf_task2=l2f.predict(test_nxor, representation=0, decider=0)
            l2f_task2=l2f.predict(test_nxor, representation='all', decider=0)
            
            errors[i,2] = 1 - np.sum(uf_task2 == test_label_nxor)/n_test
            errors[i,3] = 1 - np.sum(l2f_task2 == test_label_nxor)/n_test
            errors[i,4] = end-start
        elif n_nxor == 0:
            start = timeit.timeit()
            l2f.new_forest(xor, label_xor, n_estimators=n_trees,max_depth=max_depth)
            end = timeit.timeit()
            
            uf_task1=l2f.predict(test_xor, representation=0, decider=0)
            l2f_task1=l2f.predict(test_xor, representation='all', decider=0)
            
            errors[i,0] = 1 - np.sum(uf_task1 == test_label_xor)/n_test
            errors[i,1] = 1 - np.sum(l2f_task1 == test_label_xor)/n_test
            errors[i,2] = 0.5
            errors[i,3] = 0.5
            errors[i,4] = end-start
        else:
            start = timeit.timeit()
            l2f.new_forest(xor, label_xor, n_estimators=n_trees,max_depth=max_depth)
            l2f.new_forest(nxor, label_nxor, n_estimators=n_trees,max_depth=max_depth)
            end = timeit.timeit()

            uf.new_forest(xor, label_xor, n_estimators=2*n_trees,max_depth=max_depth)
            uf.new_forest(nxor, label_nxor, n_estimators=2*n_trees,max_depth=max_depth)

            uf_task1=uf.predict(test_xor, representation=0, decider=0)
            l2f_task1=l2f.predict(test_xor, representation='all', decider=0)
            uf_task2=uf.predict(test_nxor, representation=1, decider=1)
            l2f_task2=l2f.predict(test_nxor, representation='all', decider=1)
            
            errors[i,0] = 1 - np.sum(uf_task1 == test_label_xor)/n_test
            errors[i,1] = 1 - np.sum(l2f_task1 == test_label_xor)/n_test
            errors[i,2] = 1 - np.sum(uf_task2 == test_label_nxor)/n_test
            errors[i,3] = 1 - np.sum(l2f_task2 == test_label_nxor)/n_test
            errors[i,4] = end-start

    return np.mean(errors,axis=0)

#%%
mc_rep = 1000
n_test = 1000
n_trees = 10
n_xor = (100*np.arange(0.5, 7.25, step=0.25)).astype(int)
n_nxor = (100*np.arange(0.5, 7.50, step=0.25)).astype(int)

mean_error = np.zeros((5, len(n_xor)+len(n_nxor)))
std_error = np.zeros((5, len(n_xor)+len(n_nxor)))

mean_te = np.zeros((2, len(n_xor)+len(n_nxor)))
std_te = np.zeros((2, len(n_xor)+len(n_nxor)))

dims = [2,3,4,6,8,10]

for d in dims:
    for i,n1 in enumerate(n_xor):
        print('starting to compute %s xor\n'%n1)
        error = np.array(
            Parallel(n_jobs=40,verbose=1)(
            delayed(experiment)(n1,0,n_test,d,1,n_trees=n_trees,max_depth=ceil(log2(750))) for _ in range(mc_rep)
        )
        )
        mean_error[:,i] = np.mean(error,axis=0)
        std_error[:,i] = np.std(error,ddof=1,axis=0)
        mean_te[0,i] = np.mean(error[:,0]/error[:,1])
        mean_te[1,i] = np.mean(error[:,2]/error[:,3])
        std_te[0,i] = np.std(error[:,0]/error[:,1],ddof=1)
        std_te[1,i] = np.std(error[:,2]/error[:,3],ddof=1)
        
        if n1==n_xor[-1]:
            for j,n2 in enumerate(n_nxor):
                print('starting to compute %s nxor\n'%n2)
                
                error = np.array(
                    Parallel(n_jobs=40,verbose=1)(
                    delayed(experiment)(n1,n2,n_test,d,1,n_trees=n_trees,max_depth=ceil(log2(750))) for _ in range(mc_rep)
                )
                )
                mean_error[:,i+j+1] = np.mean(error,axis=0)
                std_error[:,i+j+1] = np.std(error,ddof=1,axis=0)
                mean_te[0,i+j+1] = np.mean(error[:,0]/error[:,1])
                mean_te[1,i+j+1] = np.mean(error[:,2]/error[:,3])
                std_te[0,i+j+1] = np.std(error[:,0]/error[:,1],ddof=1)
                std_te[1,i+j+1] = np.std(error[:,2]/error[:,3],ddof=1)
                
    with open('./result/mean_xor_nxor'+str(d)+'.pickle','wb') as f:
        pickle.dump(mean_error,f)
        
    with open('./result/std_xor_nxor'+str(d)+'.pickle','wb') as f:
        pickle.dump(std_error,f)
        
    with open('./result/mean_te_xor_nxor'+str(d)+'.pickle','wb') as f:
        pickle.dump(mean_te,f)
        
    with open('./result/std_te_xor_nxor'+str(d)+'.pickle','wb') as f:
        pickle.dump(std_te,f)

