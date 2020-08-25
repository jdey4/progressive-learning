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
sys.path.append("../src_sampling/")
#sys.path.append("../src/")
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

#%%
def produce_heatmap_data(leaf_profile, posterior, delta=0.001):
    x = np.arange(leaf_profile[0][0],leaf_profile[0][1],step=delta)
    y = np.arange(leaf_profile[1][0],leaf_profile[1][1],step=delta)
    #print(leaf_profile[0][0],leaf_profile[0][1],leaf_profile[1][0],leaf_profile[1][1])
    x,y = np.meshgrid(x,y)

    '''points = np.concatenate(
                (
                    x.reshape(-1,1),
                    y.reshape(-1,1)
                ),
                axis=1
            )'''

    if x.shape[0] == 1:
        x = np.concatenate(
            (x,x),
            axis=0
        )
    
    if x.shape[1] == 1:
        x = np.concatenate(
            (x,x),
            axis=1
        )

    if y.shape[0] == 1:
        y = np.concatenate(
            (y,y),
            axis=0
        )
    if y.shape[1] == 1:
        y = np.concatenate(
            (y,y),
            axis=1
        )

    prob = posterior*np.ones(
        x.shape,
        dtype=float
    )
    #print(x.shape,prob.shape)
    return x, y, prob

# %%
reps = 1
max_depth = 200
sample_no = 1000
err = np.zeros(reps,dtype=float)
fte = np.zeros(reps,dtype=float)
bte = np.zeros(reps,dtype=float)

np.random.seed(1)
xor, label_xor = generate_gaussian_parity(sample_no,cov_scale=0.1,angle_params=0)
test_xor, test_label_xor = generate_gaussian_parity(1000,cov_scale=0.1,angle_params=0)

''' min_xor = np.min(xor)
xor = (xor - min_xor)
max_xor = np.max(xor)
xor = xor/max_xor
test_xor = (test_xor-min_xor)/max_xor'''

nxor, label_nxor = generate_gaussian_parity(sample_no,cov_scale=0.1,angle_params=np.pi/2)
test_nxor, test_label_nxor = generate_gaussian_parity(1000,cov_scale=0.1,angle_params=np.pi/2)

'''min_nxor = np.min(nxor)
nxor = (nxor - min_nxor)
max_nxor = np.max(nxor)
nxor = nxor/max_nxor
test_nxor = (test_nxor-min_nxor)/max_nxor'''

l2f = LifeLongDNN(parallel=False)
np.random.seed(2)
l2f.new_forest(xor, label_xor, n_estimators=1, max_depth=max_depth)

delta = .001
#sample the grid
x = np.arange(-1,1,step=delta)
y = np.arange(-1,1,step=delta)
x,y = np.meshgrid(x,y)
sample = np.concatenate(
        (
            x.reshape(-1,1),
            y.reshape(-1,1)
        ),
        axis=1
    )

#%%
sample_label = l2f._estimate_posteriors(sample, representation='all', decider=0)
l2f.X_across_tasks[0] = sample
l2f.y_across_tasks[0] = sample_label

np.random.seed(3)
l2f.new_forest(nxor, label_nxor, n_estimators=1, max_depth=max_depth)

l2f_task1 = l2f.predict(test_xor, representation='all', decider=0)
uf_task1 = l2f.predict(test_xor, representation=0, decider=0)

l2f_task2 = l2f.predict(test_nxor, representation='all', decider=1)
uf_task2 = l2f.predict(test_nxor, representation=1, decider=1)

fte = (1-np.mean(uf_task2 == test_label_nxor))/(1-np.mean(l2f_task2 == test_label_nxor))
bte = (1-np.mean(uf_task1 == test_label_xor))/(1-np.mean(l2f_task1 == test_label_xor))

print(np.mean(fte), np.mean(bte))


# %%
#make the heatmap data matrix
task_no = len(l2f.voters_across_tasks_matrix)
sns.set_context("talk")
fig, ax = plt.subplots(2,2, figsize=(16,16))#, sharex=True, sharey=True)

for task_id in range(task_no):
    for voter_id in range(task_no):
        #print(task_id, voter_id)
        current_voter = l2f.voters_across_tasks_matrix[task_id][voter_id]
        posterior_map = current_voter.tree_idx_to_node_ids_to_posterior_map
        leaf_map = current_voter.tree_id_to_leaf_profile

        for tree_id in list(leaf_map.keys()):
            tree_leaf_map = leaf_map[tree_id]

            for no, leaf_id in enumerate(list(tree_leaf_map.keys())):
                x, y, prb = produce_heatmap_data(
                    tree_leaf_map[leaf_id],
                    posterior_map[tree_id][leaf_id][0]
                )
                '''if no == 0:
                    x = points
                    y = prb
                else:
                    x = np.concatenate((x,points),axis=0)
                    y = np.concatenate((y,prb),axis=0)'''

                axs = ax[task_id][voter_id].contourf(x,y,prb,cmap='gray')#,alpha=prb[0][0])
        ax[task_id][voter_id].set_xticks([0,.2,.4,.6,.8,1])
        ax[task_id][voter_id].set_yticks([0,.2,.4,.6,.8,1])
        #data = pd.DataFrame(data={'x':x[:,0], 'y':x[:,1], 'z':y})
        #data = data.pivot(index='x', columns='y', values='z')
        #ax = sns.heatmap(data,ax=axes[task_id][voter_id], vmin=0, vmax=1,)
        #ax.set_xticklabels(['0','' , '', '', '', '', '','','','.5','','' , '', '', '', '', '','','1'])
        #ax.set_yticklabels(['0','' , '', '', '', '', '','','','','','.5','','' , '', '', '', '', '','','','','1'])
        #ax.set_xlabel('transformer task '+str(voter_id+1)+' decider task '+str(task_id+1),fontsize=20)
        #ax.set_ylabel('')
        #ax.set_xticks([0,.5,1])
fig.colorbar(matplotlib.cm.ScalarMappable(cmap='gray'),ax=ax[0][1]).set_ticklabels([0,.2,.4,.6,.8,1])
fig.colorbar(matplotlib.cm.ScalarMappable(cmap='gray'),ax=ax[1][1]).set_ticklabels([0,.2,.4,.6,.8,1])
#plt.savefig('result/figs/heatmap_mapping'+str(max_depth)+'_'+str(sample_no)+'.pdf')
# %%
