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
sys.path.append("../src_mapping_2/")
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
def experiment(n_xor, n_nxor, n_test, reps, n_trees, max_depth, acorn=None):
    #print(1)
    if n_xor==0 and n_nxor==0:
        raise ValueError('Wake up and provide samples to train!!!')
    
    if acorn != None:
        np.random.seed(acorn)
    
    errors = np.zeros((reps,4),dtype=float)
    
    for i in range(reps):
        l2f = LifeLongDNN(parallel=False)
        uf = LifeLongDNN(parallel=False)
        #source data
        xor, label_xor = generate_gaussian_parity(n_xor,cov_scale=0.1,angle_params=0)
        test_xor, test_label_xor = generate_gaussian_parity(n_test,cov_scale=0.1,angle_params=0)

        min_xor = np.min(xor)
        xor = (xor - min_xor)
        max_xor = np.max(xor)
        xor = xor/max_xor
        test_xor = (test_xor-min_xor)/max_xor
        #target data
        if n_nxor!=0:
            nxor, label_nxor = generate_gaussian_parity(n_nxor,cov_scale=0.1,angle_params=np.pi/2)
            test_nxor, test_label_nxor = generate_gaussian_parity(n_test,cov_scale=0.1,angle_params=np.pi/2)

            min_nxor = np.min(nxor)
            nxor = (nxor - min_nxor)
            max_nxor = np.max(nxor)
            nxor = nxor/max_nxor
            test_nxor = (test_nxor-min_nxor)/max_nxor

        if n_xor == 0:
            l2f.new_forest(nxor, label_nxor, n_estimators=n_trees,max_depth=max_depth)
            
            errors[i,0] = 0.5
            errors[i,1] = 0.5
            
            uf_task2=l2f.predict(test_nxor, representation=0, decider=0)
            l2f_task2=l2f.predict(test_nxor, representation='all', decider=0)
            
            errors[i,2] = 1 - np.sum(uf_task2 == test_label_nxor)/n_test
            errors[i,3] = 1 - np.sum(l2f_task2 == test_label_nxor)/n_test
        elif n_nxor == 0:
            l2f.new_forest(xor, label_xor, n_estimators=n_trees,max_depth=max_depth)
            
            uf_task1=l2f.predict(test_xor, representation=0, decider=0)
            l2f_task1=l2f.predict(test_xor, representation='all', decider=0)
            
            errors[i,0] = 1 - np.sum(uf_task1 == test_label_xor)/n_test
            errors[i,1] = 1 - np.sum(l2f_task1 == test_label_xor)/n_test
            errors[i,2] = 0.5
            errors[i,3] = 0.5
        else:
            l2f.new_forest(xor, label_xor, n_estimators=n_trees,max_depth=max_depth)

            '''delta = .01
            #sample the grid
            x = np.arange(0,1,step=delta)
            y = np.arange(0,1,step=delta)
            x,y = np.meshgrid(x,y)
            sample = np.concatenate(
                (
                    x.reshape(-1,1),
                    y.reshape(-1,1)
                ),
                axis=1
            )
            sample_label = l2f.predict(sample, representation=0,decider=0)
            l2f.X_across_tasks[0] = sample
            l2f.y_across_tasks[0] = sample_label'''
            ############################

            l2f.new_forest(nxor, label_nxor, n_estimators=n_trees,max_depth=max_depth)
            
            uf.new_forest(xor, label_xor, n_estimators=n_trees,max_depth=max_depth)
            uf.new_forest(nxor, label_nxor, n_estimators=n_trees,max_depth=max_depth)

            uf_task1=uf.predict(test_xor, representation=0, decider=0)
            l2f_task1=l2f.predict(test_xor, representation='all', decider=0)
            uf_task2=uf.predict(test_nxor, representation=1, decider=1)
            l2f_task2=l2f.predict(test_nxor, representation='all', decider=1)
            
            errors[i,0] = 1 - np.sum(uf_task1 == test_label_xor)/n_test
            errors[i,1] = 1 - np.sum(l2f_task1 == test_label_xor)/n_test
            errors[i,2] = 1 - np.sum(uf_task2 == test_label_nxor)/n_test
            errors[i,3] = 1 - np.sum(l2f_task2 == test_label_nxor)/n_test

    return np.mean(errors,axis=0)

#%%
mc_rep = 1000
n_test = 1000
n_trees = 10
n_xor = (100*np.arange(0.5, 7.25, step=0.25)).astype(int)
n_nxor = (100*np.arange(0.5, 7.5, step=0.25)).astype(int)

mean_error = np.zeros((4, len(n_xor)+len(n_nxor)))
std_error = np.zeros((4, len(n_xor)+len(n_nxor)))

mean_te = np.zeros((2, len(n_xor)+len(n_nxor)))
std_te = np.zeros((2, len(n_xor)+len(n_nxor)))

for i,n1 in enumerate(n_xor):
    print('starting to compute %s xor\n'%n1)
    error = np.array(
        Parallel(n_jobs=-1,verbose=1)(
        delayed(experiment)(n1,0,n_test,1,n_trees=n_trees,max_depth=200) for _ in range(mc_rep)
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
                Parallel(n_jobs=-1,verbose=1)(
                delayed(experiment)(n1,n2,n_test,1,n_trees=n_trees,max_depth=200) for _ in range(mc_rep)
            )
            )
            mean_error[:,i+j+1] = np.mean(error,axis=0)
            std_error[:,i+j+1] = np.std(error,ddof=1,axis=0)
            mean_te[0,i+j+1] = np.mean(error[:,0]/error[:,1])
            mean_te[1,i+j+1] = np.mean(error[:,2]/error[:,3])
            std_te[0,i+j+1] = np.std(error[:,0]/error[:,1],ddof=1)
            std_te[1,i+j+1] = np.std(error[:,2]/error[:,3],ddof=1)
            
with open('./result/mean_xor_nxor.pickle','wb') as f:
    pickle.dump(mean_error,f)
    
with open('./result/std_xor_nxor.pickle','wb') as f:
    pickle.dump(std_error,f)
    
with open('./result/mean_te_xor_nxor.pickle','wb') as f:
    pickle.dump(mean_te,f)
    
with open('./result/std_te_xor_nxor.pickle','wb') as f:
    pickle.dump(std_te,f)

#%% Plotting the result
#mc_rep = 50
mean_error = unpickle('result/mean_xor_nxor.pickle')
std_error = unpickle('result/std_xor_nxor.pickle')

n_xor = (100*np.arange(0.5, 7.25, step=0.25)).astype(int)
n_nxor = (100*np.arange(0.5, 7.5, step=0.25)).astype(int)

n1s = n_xor
n2s = n_nxor

ns = np.concatenate((n1s, n2s + n1s[-1]))
ls=['-', '--']
algorithms = ['Uncertainty Forest', 'Lifelong Forest']


TASK1='XOR'
TASK2='N-XOR'

fontsize=30
labelsize=28

colors = sns.color_palette("Set1", n_colors = 2)

fig = plt.figure(constrained_layout=True,figsize=(21,14))
gs = fig.add_gridspec(14, 21)
ax1 = fig.add_subplot(gs[7:,:6])
# for i, algo in enumerate(algorithms):
ax1.plot(ns, mean_error[0], label=algorithms[0], c=colors[1], ls=ls[np.sum(0 > 1).astype(int)], lw=3)
#ax1.fill_between(ns, 
#        mean_error[0] + 1.96*std_error[0], 
#        mean_error[0] - 1.96*std_error[0], 
#        where=mean_error[0] + 1.96*std_error[0] >= mean_error[0] - 1.96*std_error[0], 
#        facecolor=colors[1], 
#        alpha=0.15,
#        interpolate=True)

ax1.plot(ns, mean_error[1], label=algorithms[1], c=colors[0], ls=ls[np.sum(1 > 1).astype(int)], lw=3)
#ax1.fill_between(ns, 
#        mean_error[1] + 1.96*std_error[1, ], 
#        mean_error[1] - 1.96*std_error[1, ], 
#        where=mean_error[1] + 1.96*std_error[1] >= mean_error[1] - 1.96*std_error[1], 
#        facecolor=colors[0], 
#        alpha=0.15,
#        interpolate=True)

ax1.set_ylabel('Generalization Error (%s)'%(TASK1), fontsize=fontsize)
ax1.legend(loc='upper right', fontsize=20, frameon=False)
ax1.set_ylim(0.1, 0.21)
ax1.set_xlabel('Total Sample Size', fontsize=fontsize)
ax1.tick_params(labelsize=labelsize)
ax1.set_yticks([0.15, 0.2])
ax1.set_xticks([250,750,1500])
ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")
ax1.set_title('XOR', fontsize=30)

right_side = ax1.spines["right"]
right_side.set_visible(False)
top_side = ax1.spines["top"]
top_side.set_visible(False)

ax1.text(400, np.mean(ax1.get_ylim()), "%s"%(TASK1), fontsize=26)
ax1.text(900, np.mean(ax1.get_ylim()), "%s"%(TASK2), fontsize=26)

#plt.tight_layout()

#plt.savefig('./result/figs/generalization_error_xor.pdf',dpi=500)

#####
mean_error = unpickle('result/mean_xor_nxor.pickle')
std_error = unpickle('result/std_xor_nxor.pickle')

algorithms = ['Uncertainty Forest', 'Lifelong Forest']

TASK1='XOR'
TASK2='N-XOR'

ax1 = fig.add_subplot(gs[7:,7:13])
# for i, algo in enumerate(algorithms):
ax1.plot(ns[len(n1s):], mean_error[2, len(n1s):], label=algorithms[0], c=colors[1], ls=ls[1], lw=3)
#ax1.fill_between(ns[len(n1s):], 
#        mean_error[2, len(n1s):] + 1.96*std_error[2, len(n1s):], 
#        mean_error[2, len(n1s):] - 1.96*std_error[2, len(n1s):], 
#        where=mean_error[2, len(n1s):] + 1.96*std_error[2, len(n1s):] >= mean_error[2, len(n1s):] - 1.96*std_error[2, len(n1s):], 
#        facecolor=colors[1], 
#        alpha=0.15,
#        interpolate=True)

ax1.plot(ns[len(n1s):], mean_error[3, len(n1s):], label=algorithms[1], c=colors[0], ls=ls[1], lw=3)
#ax1.fill_between(ns[len(n1s):], 
#        mean_error[3, len(n1s):] + 1.96*std_error[3, len(n1s):], 
#        mean_error[3, len(n1s):] - 1.96*std_error[3, len(n1s):], 
#        where=mean_error[3, len(n1s):] + 1.96*std_error[3, len(n1s):] >= mean_error[3, len(n1s):] - 1.96*std_error[3, len(n1s):], 
#        facecolor=colors[0], 
#        alpha=0.15,
#        interpolate=True)

ax1.set_ylabel('Generalization Error (%s)'%(TASK2), fontsize=fontsize)
ax1.legend(loc='upper right', fontsize=20, frameon=False)
#         ax1.set_ylim(-0.01, 0.22)
ax1.set_xlabel('Total Sample Size', fontsize=fontsize)
ax1.tick_params(labelsize=labelsize)
# ax1.set_yticks([0.15, 0.25, 0.35])
ax1.set_yticks([0.15, 0.2])
ax1.set_xticks([250,750,1500])
ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")

ax1.set_ylim(0.11, 0.21)

ax1.set_xlim(-10)
right_side = ax1.spines["right"]
right_side.set_visible(False)
top_side = ax1.spines["top"]
top_side.set_visible(False)

# ax1.set_ylim(0.14, 0.36)
ax1.text(400, np.mean(ax1.get_ylim()), "%s"%(TASK1), fontsize=26)
ax1.text(900, np.mean(ax1.get_ylim()), "%s"%(TASK2), fontsize=26)

ax1.set_title('N-XOR', fontsize=30)
#plt.tight_layout()

#plt.savefig('./result/figs/generalization_error_nxor.pdf',dpi=500)

#####
mean_error = unpickle('result/mean_te_xor_nxor.pickle')
std_error = unpickle('result/std_te_xor_nxor.pickle')

algorithms = ['Backward Transfer', 'Forward Transfer']

TASK1='XOR'
TASK2='N-XOR'

ax1 = fig.add_subplot(gs[7:,14:])

ax1.plot(ns, mean_error[0], label=algorithms[0], c=colors[0], ls=ls[0], lw=3)
#ax1.fill_between(ns, 
#        mean_error[0] + 1.96*std_error[0], 
#        mean_error[0] - 1.96*std_error[0], 
#        where=mean_error[1] + 1.96*std_error[0] >= mean_error[0] - 1.96*std_error[0], 
#        facecolor=colors[0], 
#        alpha=0.15,
#        interpolate=True)

ax1.plot(ns[len(n1s):], mean_error[1, len(n1s):], label=algorithms[1], c=colors[0], ls=ls[1], lw=3)
#ax1.fill_between(ns[len(n1s):], 
#        mean_error[1, len(n1s):] + 1.96*std_error[1, len(n1s):], 
#        mean_error[1, len(n1s):] - 1.96*std_error[1, len(n1s):], 
#        where=mean_error[1, len(n1s):] + 1.96*std_error[1, len(n1s):] >= mean_error[1, len(n1s):] - 1.96*std_error[1, len(n1s):], 
#        facecolor=colors[0], 
#        alpha=0.15,
#        interpolate=True)

ax1.set_ylabel('Transfer Efficiency', fontsize=fontsize)
ax1.legend(loc='upper right', fontsize=20, frameon=False)
ax1.set_ylim(.99, 1.4)
ax1.set_xlabel('Total Sample Size', fontsize=fontsize)
ax1.tick_params(labelsize=labelsize)
ax1.set_yticks([1,1.2,1.4])
ax1.set_xticks([250,750,1500])
ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")
right_side = ax1.spines["right"]
right_side.set_visible(False)
top_side = ax1.spines["top"]
top_side.set_visible(False)
ax1.hlines(1, 50,1500, colors='gray', linestyles='dashed',linewidth=1.5)

ax1.text(400, np.mean(ax1.get_ylim()), "%s"%(TASK1), fontsize=26)
ax1.text(900, np.mean(ax1.get_ylim()), "%s"%(TASK2), fontsize=26)

#plt.tight_layout()

#plt.savefig('./result/figs/TE.pdf',dpi=500)

#####
colors = sns.color_palette('Dark2', n_colors=2)

X, Y = generate_gaussian_parity(750, cov_scale=0.1, angle_params=0)
Z, W = generate_gaussian_parity(750, cov_scale=0.1, angle_params=np.pi/2)

ax = fig.add_subplot(gs[:6,4:10])
ax.scatter(X[:, 0], X[:, 1], c=get_colors(colors, Y), s=50)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Gaussian XOR', fontsize=30)

plt.tight_layout()
ax.axis('off')
#plt.savefig('./result/figs/gaussian-xor.pdf')

###
colors = sns.color_palette('Dark2', n_colors=2)

ax = fig.add_subplot(gs[:6,11:16])
ax.scatter(Z[:, 0], Z[:, 1], c=get_colors(colors, W), s=50)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Gaussian N-XOR', fontsize=30)
ax.axis('off')
#plt.tight_layout()
plt.savefig('./result/figs/xor_nxor_exp_honest_trees.pdf')

# %%
