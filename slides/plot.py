#%%
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import seaborn as sns 
import matplotlib
import numpy as np
import pickle

from sklearn.model_selection import StratifiedKFold
from math import log2, ceil 

import sys
sys.path.append("../src/")
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


def generate_spirals(N, D=2, K=5, noise = 0.5, acorn = None, density=0.3):

    #N number of poinst per class
    #D number of features, 
    #K number of classes
    X = []
    Y = []
    if acorn is not None:
        np.random.seed(acorn)
    
    if K == 2:
        turns = 2
    elif K==3:
        turns = 2.5
    elif K==5:
        turns = 3.5
    elif K==7:
        turns = 4.5
    else:
        print ("sorry, can't currently surpport %s classes " %K)
        return
    
    mvt = np.random.multinomial(N, 1/K * np.ones(K))
    
    if K == 2:
#         r = np.linspace(0.01, 1, N)
        r = np.random.uniform(0,1,size=int(N/K))
        r = np.sort(r)
        t = np.linspace(0,  np.pi* 4 * turns/K, int(N/K)) + noise * np.random.normal(0, density, int(N/K))
        dx = r * np.cos(t)
        dy = r* np.sin(t)

        X.append(np.vstack([dx, dy]).T )
        X.append(np.vstack([-dx, -dy]).T)
        Y += [0] * int(N/K) 
        Y += [1] * int(N/K)
    else:    
        for j in range(1, K+1):
            r = np.linspace(0.01, 1, int(mvt[j-1]))
            t = np.linspace((j-1) * np.pi *4 *turns/K,  j* np.pi * 4* turns/K, int(mvt[j-1])) + noise * np.random.normal(0, density, int(mvt[j-1]))
            dx = r * np.cos(t)
            dy = r* np.sin(t)

            dd = np.vstack([dx, dy]).T        
            X.append(dd)
            #label
            Y += [j-1] * int(mvt[j-1])
    return np.vstack(X), np.array(Y).astype(int)

# %%
fontsize=30
labelsize=28


fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True, figsize=(12,6))

colors = sns.color_palette('Dark2', n_colors=2)

X, Y = generate_gaussian_parity(750, cov_scale=0.1, angle_params=0)
Z, W = generate_gaussian_parity(750, cov_scale=0.1, angle_params=np.pi/2)

ax[0].scatter(X[:, 0], X[:, 1], c=get_colors(colors, Y), s=50)

ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title('Gaussian XOR', fontsize=30)

plt.tight_layout()
ax[0].axis('off')
#plt.savefig('./result/figs/gaussian-xor.pdf')

#####################
ax[1].scatter(Z[:, 0], Z[:, 1], c=get_colors(colors, W), s=50)

ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title('Gaussian N-XOR', fontsize=30)
ax[1].axis('off')

plt.savefig('figs/gaussian-xor-nxor.svg')
# %%
with open('../experiments/xor_nxor_exp/result/mean_xor_nxor.pickle','rb') as f:
    mean_error = pickle.load(f)

n_xor = (100*np.arange(0.5, 7.25, step=0.25)).astype(int)
n_nxor = (100*np.arange(0.5, 7.50, step=0.25)).astype(int)

n1s = n_xor
n2s = n_nxor

ns = np.concatenate((n1s, n2s + n1s[-1]))
ls=['-', '--']
algorithms = ['XOR Forest', 'N-XOR Forest', 'Lifelong Forest', 'Naive Forest']


TASK1='XOR'
TASK2='N-XOR'

fontsize=30
labelsize=28

colors = sns.color_palette("Set1", n_colors = 2)

fig = plt.figure(constrained_layout=True,figsize=(21,14))
gs = fig.add_gridspec(14, 21)
ax1 = fig.add_subplot(gs[7:,:6])
ax1.plot(n1s, mean_error[0,:len(n1s)], label=algorithms[0], c=colors[1], ls=ls[np.sum(0 > 1).astype(int)], lw=3)
ax1.plot(ns[len(n1s):], mean_error[2, len(n1s):], label=algorithms[1], c=colors[1], ls=ls[1], lw=3)
ax1.plot(ns, mean_error[1], label=algorithms[2], c=colors[0], ls=ls[np.sum(1 > 1).astype(int)], lw=3)
ax1.plot(ns, mean_error[4], label=algorithms[3], c='g', ls=ls[np.sum(1 > 1).astype(int)], lw=3)

ax1.set_ylabel('Generalization Error (%s)'%(TASK1), fontsize=fontsize)
ax1.legend(loc='upper right', fontsize=20, frameon=False)
#ax1.set_ylim(0.09, 0.21)
ax1.set_xlabel('Total Sample Size', fontsize=fontsize)
ax1.tick_params(labelsize=labelsize)
#ax1.set_yticks([0.5,0.15, 0.25])
ax1.set_xticks([250,750,1500])
ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")
ax1.set_title('XOR', fontsize=30)

right_side = ax1.spines["right"]
right_side.set_visible(False)
top_side = ax1.spines["top"]
top_side.set_visible(False)

ax1.text(400, np.mean(ax1.get_ylim()), "%s"%(TASK1), fontsize=26)
ax1.text(900, np.mean(ax1.get_ylim()), "%s"%(TASK2), fontsize=26)

#####################################
with open('../experiments/xor_nxor_exp/result/mean_xor_nxor.pickle','rb') as f:
    mean_error = pickle.load(f)

algorithms = ['XOR Forest', 'N-XOR Forest', 'Lifelong Forest', 'Naive Forest']

TASK1='XOR'
TASK2='N-XOR'

ax1 = fig.add_subplot(gs[7:,7:13])
ax1.plot(n1s, mean_error[0,:len(n1s)], label=algorithms[0], c=colors[1], ls=ls[np.sum(0 > 1).astype(int)], lw=3)
ax1.plot(ns[len(n1s):], mean_error[2, len(n1s):], label=algorithms[1], c=colors[1], ls=ls[1], lw=3)

ax1.plot(ns[len(n1s):], mean_error[3, len(n1s):], label=algorithms[2], c=colors[0], ls=ls[1], lw=3)
ax1.plot(ns[len(n1s):], mean_error[5, len(n1s):], label=algorithms[3], c='g', ls=ls[1], lw=3)

ax1.set_ylabel('Generalization Error (%s)'%(TASK2), fontsize=fontsize)
ax1.legend(loc='upper right', fontsize=18, frameon=False)
ax1.set_xlabel('Total Sample Size', fontsize=fontsize)
ax1.tick_params(labelsize=labelsize)
ax1.set_xticks([250,750,1500])
ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")


right_side = ax1.spines["right"]
right_side.set_visible(False)
top_side = ax1.spines["top"]
top_side.set_visible(False)

ax1.text(400, np.mean(ax1.get_ylim()), "%s"%(TASK1), fontsize=26)
ax1.text(900, np.mean(ax1.get_ylim()), "%s"%(TASK2), fontsize=26)

ax1.set_title('N-XOR', fontsize=30)

#####################################
with open('../experiments/xor_nxor_exp/result/mean_te_xor_nxor.pickle','rb') as f:
    mean_te = pickle.load(f)

algorithms = ['Lifelong BTE', 'Lifelong FTE', 'Naive BTE', 'Naive FTE']

TASK1='XOR'
TASK2='N-XOR'

ax1 = fig.add_subplot(gs[7:,14:])

ax1.plot(ns, mean_te[0], label=algorithms[0], c=colors[0], ls=ls[0], lw=3)
ax1.plot(ns[len(n1s):], mean_te[1, len(n1s):], label=algorithms[1], c=colors[0], ls=ls[1], lw=3)
ax1.plot(ns, mean_te[2], label=algorithms[2], c='g', ls=ls[0], lw=3)
ax1.plot(ns[len(n1s):], mean_te[3, len(n1s):], label=algorithms[3], c='g', ls=ls[1], lw=3)

ax1.set_ylabel('Forward/Backward \n Transfer Efficiency (FTE/BTE)', fontsize=fontsize)
ax1.legend(loc='upper right', fontsize=20, frameon=False)
ax1.set_xlabel('Total Sample Size', fontsize=fontsize)
ax1.tick_params(labelsize=labelsize)
#ax1.set_yticks([0,.5,1,1.5])
ax1.set_xticks([250,750,1500])
ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")
right_side = ax1.spines["right"]
right_side.set_visible(False)
top_side = ax1.spines["top"]
top_side.set_visible(False)
ax1.hlines(1, 50,1500, colors='gray', linestyles='dashed',linewidth=1.5)

ax1.text(400, np.mean(ax1.get_ylim()), "%s"%(TASK1), fontsize=26)
ax1.text(900, np.mean(ax1.get_ylim()), "%s"%(TASK2), fontsize=26)

colors = sns.color_palette('Dark2', n_colors=2)

X, Y = generate_gaussian_parity(750, cov_scale=0.1, angle_params=0)
Z, W = generate_gaussian_parity(750, cov_scale=0.1, angle_params=np.pi/2)

ax = fig.add_subplot(gs[:6,4:10])
clr = [colors[i] for i in Y]
ax.scatter(X[:, 0], X[:, 1], c=clr, s=50)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Gaussian XOR', fontsize=30)

plt.tight_layout()
ax.axis('off')

colors = sns.color_palette('Dark2', n_colors=2)

ax = fig.add_subplot(gs[:6,11:16])
clr = [colors[i] for i in W]
ax.scatter(Z[:, 0], Z[:, 1], c=clr, s=50)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Gaussian N-XOR', fontsize=30)
ax.axis('off')
#plt.tight_layout()
plt.savefig('./figs/xor-te.svg')

# %%
fontsize=30
labelsize=28


fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(26,16))

colors = sns.color_palette('Dark2', n_colors=2)

X, Y = generate_gaussian_parity(750, cov_scale=0.1, angle_params=0)
Z, W = generate_gaussian_parity(750, cov_scale=0.1, angle_params=np.pi/4)

ax[0][0].scatter(X[:, 0], X[:, 1], c=get_colors(colors, Y), s=50)

ax[0][0].set_xticks([])
ax[0][0].set_yticks([])
ax[0][0].set_title('Gaussian XOR', fontsize=30)

#plt.tight_layout()
ax[0][0].axis('off')
#plt.savefig('./result/figs/gaussian-xor.pdf')

#####################
ax[0][1].scatter(Z[:, 0], Z[:, 1], c=get_colors(colors, W), s=50)

ax[0][1].set_xticks([])
ax[0][1].set_yticks([])
ax[0][1].set_title('Gaussian R-XOR', fontsize=30)
ax[0][1].axis('off')

#####################
mean_error = unpickle('../experiments/xor_nxor_exp/result/mean_te_xor_rxor.pickle')

algorithms = ['Lifelong BTE', 'Lifelong FTE', 'Naive BTE', 'Naive FTE']

TASK1='XOR'
TASK2='R-XOR'

ax[0][2].plot(ns, mean_te[0], label=algorithms[0], c=colors[0], ls=ls[0], lw=3)
ax[0][2].plot(ns[len(n1s):], mean_te[1, len(n1s):], label=algorithms[1], c=colors[0], ls=ls[1], lw=3)
ax[0][2].plot(ns, mean_te[2], label=algorithms[2], c='g', ls=ls[0], lw=3)
ax[0][2].plot(ns[len(n1s):], mean_te[3, len(n1s):], label=algorithms[3], c='g', ls=ls[1], lw=3)

ax[0][2].set_ylabel('Forward/Backward \n Transfer Efficiency (FTE/BTE)', fontsize=fontsize)
ax[0][2].legend(loc='upper right', fontsize=20, frameon=False)
ax[0][2].set_xlabel('Total Sample Size', fontsize=fontsize)
ax[0][2].tick_params(labelsize=labelsize)
#ax1.set_yticks([0,.5,1,1.5])
ax[0][2].set_xticks([250,750,1500])
ax[0][2].axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")
right_side = ax[0][2].spines["right"]
right_side.set_visible(False)
top_side = ax[0][2].spines["top"]
top_side.set_visible(False)
ax[0][2].hlines(1, 50,1500, colors='gray', linestyles='dashed',linewidth=1.5)

ax[0][2].text(400, np.mean(ax1.get_ylim()), "%s"%(TASK1), fontsize=26)
ax[0][2].text(900, np.mean(ax1.get_ylim()), "%s"%(TASK2), fontsize=26)



#########################################
mean_error = unpickle('../experiments/xor_rxor_spiral_exp/result/mean_spiral.pickle')
std_error = unpickle('../experiments/xor_rxor_spiral_exp/result/std_spiral.pickle')

spiral3 = (100*np.arange(0.5, 7.25, step=0.25)).astype(int)
spiral5 = (100*np.arange(0.5, 7.50, step=0.25)).astype(int)

n1s = spiral3
n2s = spiral5

ns = np.concatenate((n1s, n2s + n1s[-1]))
ls=['-', '--']
algorithms = ['Uncertainty Forest', 'Lifelong Forest']


TASK1='3 spirals'
TASK2='5 spirals'

fontsize=30
labelsize=28

colors = sns.color_palette("Set1", n_colors = 2)

#####
mean_error = unpickle('../experiments/xor_rxor_spiral_exp/result/mean_te_spiral.pickle')
std_error = unpickle('../experiments/xor_rxor_spiral_exp/result/std_te_spiral.pickle')

algorithms = ['Backward Transfer', 'Forward Transfer']

TASK1='3 spirals'
TASK2='5 spirals'

ax[1][2].plot(ns, mean_error[0], label=algorithms[0], c=colors[0], ls=ls[0], lw=3)
#ax1.fill_between(ns, 
#        mean_error[0] + 1.96*std_error[0], 
#        mean_error[0] - 1.96*std_error[0], 
#        where=mean_error[1] + 1.96*std_error[0] >= mean_error[0] - 1.96*std_error[0], 
#        facecolor=colors[0], 
#        alpha=0.15,
#        interpolate=True)

ax[1][2].plot(ns[len(n1s):], mean_error[1, len(n1s):], label=algorithms[1], c=colors[0], ls=ls[1], lw=3)
#ax1.fill_between(ns[len(n1s):], 
#        mean_error[1, len(n1s):] + 1.96*std_error[1, len(n1s):], 
#        mean_error[1, len(n1s):] - 1.96*std_error[1, len(n1s):], 
#        where=mean_error[1, len(n1s):] + 1.96*std_error[1, len(n1s):] >= mean_error[1, len(n1s):] - 1.96*std_error[1, len(n1s):], 
#        facecolor=colors[0], 
#        alpha=0.15,
#        interpolate=True)

ax[1][2].set_ylabel('Transfer Efficiency', fontsize=fontsize)
ax[1][2].legend(loc='upper right', fontsize=20, frameon=False)
ax[1][2].set_ylim(.92, 1.1)
ax[1][2].set_xlabel('Total Sample Size', fontsize=fontsize)
ax[1][2].tick_params(labelsize=labelsize)
ax[1][2].set_yticks([.92,1,1.08])
ax[1][2].set_xticks([250,750,1500])
ax[1][2].axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")
right_side = ax1.spines["right"]
right_side.set_visible(False)
top_side = ax1.spines["top"]
top_side.set_visible(False)
ax[1][2].hlines(1, 50,1500, colors='gray', linestyles='dashed',linewidth=1.5)

ax[1][2].text(150, np.mean(ax[1][2].get_ylim()), "%s"%(TASK1), fontsize=26)
ax[1][2].text(900, np.mean(ax[1][2].get_ylim()), "%s"%(TASK2), fontsize=26)

right_side = ax[1][2].spines["right"]
right_side.set_visible(False)
top_side = ax[1][2].spines["top"]
top_side.set_visible(False)
#plt.tight_layout()

#plt.savefig('./result/figs/TE.pdf',dpi=500)

#####
colors = sns.color_palette('Dark2', n_colors=5)

X, Y = generate_spirals(750, 2, 3, noise = 2.5)
Z, W = generate_spirals(750, 2, 5, noise = 2.5)

ax[1][0].scatter(X[:, 0], X[:, 1], c=get_colors(colors, Y), s=50)

ax[1][0].set_xticks([])
ax[1][0].set_yticks([])
ax[1][0].set_title('3 spirals', fontsize=30)

#plt.tight_layout()
ax[1][0].axis('off')
#plt.savefig('./result/figs/gaussian-xor.pdf')

###
colors = sns.color_palette('Dark2', n_colors=5)

ax[1][1].scatter(Z[:, 0], Z[:, 1], c=get_colors(colors, W), s=50)

ax[1][1].set_xticks([])
ax[1][1].set_yticks([])
ax[1][1].set_title('5 spirals', fontsize=30)
ax[1][1].axis('off')

plt.savefig('figs/lotsa-te.svg')
# %%
