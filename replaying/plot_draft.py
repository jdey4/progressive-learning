#%%
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
from math import ceil,log2
import pickle 

sys.path.append("../src/")
from lifelong_dnn import LifeLongDNN

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


# %% main hyperparameters
delta = 0.01
#########################

x = np.arange(-1, 1+delta, step=delta)
y = np.arange(-1, 1+delta, step=delta)
X, Y = np.meshgrid(x,y)
X = np.ravel(X).reshape(-1,1)
Y = np.ravel(Y).reshape(-1,1)
noise = np.concatenate((X,Y), axis=1)

# %%
xor, label_xor = generate_gaussian_parity(750,cov_scale=0.1,angle_params=0)
normalize_xor = np.max(np.abs(xor))
xor = xor/normalize_xor

l2f = LifeLongDNN()
l2f.new_forest(xor, label_xor, n_estimators=10,max_depth=ceil(log2(750)))

label_noise = l2f.predict(noise, representation=0,decider=0).astype(int)

#%%
sns.set_context("talk")

colors = sns.color_palette('Dark2', n_colors=2)
fig, ax = plt.subplots(ncols=2, nrows=1, sharey=True, figsize=(12,6))
ax[0].scatter(X,Y,s=1)
ax[0].set_xticks([-1, -.5, 0, .5, 1])
ax[0].set_yticks([-1, -.5, 0, .5, 1])

ax[1].scatter(X, Y, c=get_colors(colors, label_noise), s=1)
ax[1].set_xticks([-1, -.5, 0, .5, 1])
ax[1].set_yticks([-1, -.5, 0, .5, 1])

plt.savefig('result/figs/partition.pdf')
# %%
