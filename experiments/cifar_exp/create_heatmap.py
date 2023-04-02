#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %%
sns.set_context('talk')
accuracy = {}
accuracy_ = {}
forget = {}
forget_ = {}
transfer_ = {}
transfer = {}
data = {}
algorithms_ = ['SynN', 'SynF', 'ProgNN', \
            'DF-CNN', 'EWC', 'Tota Replay', \
            'Partial Replay', 'Model Zoo', \
            'SynF (constrained)', \
            'LwF', 'O-EWC', 'SI', 'ER', \
            'A-GEM', 'TAG', 'None']
algorithms = []

sorted_indx = [ 0,  2,  7,  1,  8,  6,  5, 13,  9, 12,  3, 14, 10, 11,  4, 15]

accuracy_['cifar'] = [.4, .41, .39, .17, .36, .36, .37, .46,\
           .41, .42, .36, .35, .32, .27, .15, .29]
forget_['cifar'] = [.03, .03, 0, -.09, -.01, -.03, -.01, .05,\
          .03, 0, 0, -.01, -.13, -.17, -.05, -.14]
transfer_['cifar'] = [.13, .08, .07, -.09, -.09, -.09, -.07,\
            .05, .03, -.03, -.08, -.09, -.09, -.13,\
            -.23, -.16]

accuracy_['5 dataset'] = [.79, .71, np.nan, np.nan, .68, .83, .82, .89,\
            np.nan, .8, .68, .66, .68, .67, .69, .62]
forget_['5 dataset'] = [-.02, -.01, np.nan, np.nan, -.08, -.01, -.01, .03,\
            np.nan, -.07, -.08, -.07, -.13, -.14, -.11, -.31]
transfer_['5 dataset'] = [-.03, -.02, np.nan, np.nan, -.18, -.03, -.04, .03,\
            np.nan, -.06, -.18, -.2, -.12, -.09, -.1, -.24]


accuracy_['imagenet'] = [.55, .52, np.nan, np.nan, .54, .58, .58, .6,\
            np.nan, .6, .55, .57, .58, .56, .58, .52]
forget_['imagenet'] = [.03, .02, np.nan, np.nan, -.04, -.01, -.01, .06,\
            np.nan, 0, -.03, -.02, -.07, -.07, -.06, -.12]
transfer_['imagenet'] = [.02, .04, np.nan, np.nan, -.07, -.03, -.03, .1,\
            np.nan, -.01, -.06, -.04, -.01, .05, -.04, -.1]

accuracy_['speech'] = [.91, .91, np.nan, np.nan, .73, .93, .93, .98,\
            np.nan, .76, .72, .72, np.nan, np.nan, np.nan, .73]
forget_['speech'] = [.035, .01, np.nan, np.nan, -.28, 0, 0, 0.01,\
            np.nan, -.24, -.29, -.27, np.nan, np.nan, np.nan, -.3]
transfer_['speech'] = [.16, .03, np.nan, np.nan, -.24, -.04, -.04, 0,\
            np.nan, -.21, -.25, -.25, np.nan, np.nan, np.nan, -.23]

accuracy_['food1k'] = [.45, .36, np.nan, np.nan, np.nan, np.nan, np.nan, .42,\
           np.nan, .43, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
forget_['food1k'] = [.03, .01, np.nan, np.nan, np.nan, np.nan, np.nan, .05,\
           np.nan, -.08, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
transfer_['food1k'] = [.2, .09, np.nan, np.nan, np.nan, np.nan, np.nan, .08,\
           np.nan, -.03, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

for idx in sorted_indx:
    algorithms.append(algorithms_[idx])

for key in transfer_.keys():
    accuracy[key] = []
    forget[key] = []
    transfer[key] = []
    for idx in sorted_indx:
        accuracy[key].append(accuracy_[key][idx])
        forget[key].append(forget_[key][idx])
        transfer[key].append(transfer_[key][idx])

keys = ['accuracy', 'forget', 'transfer']

data['accuracy'] = pd.DataFrame({"cifar":accuracy["cifar"], "5-dataset":accuracy["5 dataset"],\
        "imagenet":accuracy["imagenet"], "speech":accuracy["speech"], "food1k":accuracy["food1k"]})
data['accuracy'].index = algorithms
data['accuracy']['mean'] = data['accuracy'].mean(axis=1)


data['forget'] = pd.DataFrame({"cifar":forget["cifar"], "5-dataset":forget["5 dataset"],\
        "imagenet":forget["imagenet"], "speech":forget["speech"], "food1k":forget["food1k"]})
data['forget'].index = algorithms
data['forget']['mean'] = data['forget'].mean(axis=1)


data['transfer'] = pd.DataFrame({"cifar":transfer["cifar"], "5-dataset":transfer["5 dataset"],\
        "imagenet":transfer["imagenet"], "speech":transfer["speech"], "food1k":transfer["food1k"]})
data['transfer'].index = algorithms
data['transfer']['mean'] = data['transfer'].mean(axis=1)
mean_trn = list(data['transfer']['mean'])
print(np.argsort(mean_trn)[::-1])

fig, ax = plt.subplots(1, 3, figsize=(20,8), sharey=True, sharex=True)
#cbar_ax = fig.add_axes([.91, .3, .03, .4])
vmins = [0,-.3,-.3]
vmaxs = [1,.3,.3]
for i in range(3):
    sns.heatmap(data[keys[i]], yticklabels=algorithms,\
                vmin=vmins[i], vmax=vmaxs[i],
             cmap='coolwarm', ax=ax[i],)# cbar=i == 0, \
             #cbar_ax=cbar_ax if i==0 else None)
    ax[i].set_title(keys[i], fontsize=35)
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=70, fontsize=20)
    #ax[i].set_yticklabels(ax[i].get_yticklabels(), fontsize=20)

fig.tight_layout(rect=[0, 0, .9, 1])
plt.savefig('/Users/jayantadey/progressive-learning/result/figs/heatmap_performance.pdf')
# %%
