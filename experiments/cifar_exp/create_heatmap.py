#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %%
sns.set_context('talk')
accuracy = {}
forget = {}
transfer = {}
data = {}
algorithms = ['SynN', 'SynF', 'ProgNN', \
            'DF-CNN', 'EWC', 'Tota Replay', \
            'Partial Replay', 'Model Zoo', \
            'SynF \n (constrained)', \
            'LwF', 'O-EWC', 'SI', 'ER', \
            'A-GEM', 'TAG', 'None']

accuracy['cifar'] = [.4, .41, .39, .17, .36, .36, .37, .46,\
           .41, .42, .36, .35, .32, .27, .15, .29]
forget['cifar'] = [.03, .03, 0, -.09, -.01, -.03, -.01, .05,\
          .03, 0, 0, -.01, -.13, -.17, -.05, -.14]
transfer['cifar'] = [.13, .08, .07, -.09, -.09, -.09, -.07,\
            .05, .03, -.03, -.08, -.09, -.09, -.13,\
            -.23, -.16]

accuracy['5 dataset'] = [.79, .71, np.nan, np.nan, .68, .83, .82, .89,\
            np.nan, .8, .68, .66, .68, .67, .69, .62]
forget['5 dataset'] = [-.02, -.01, np.nan, np.nan, -.08, -.01, -.01, .03,\
            np.nan, -.07, -.08, -.07, -.13, -.14, -.11, -.31]
transfer['5 dataset'] = [-.03, -.02, np.nan, np.nan, -.18, -.03, -.04, .03,\
            np.nan, -.06, -.18, -.2, -.12, -.09, -.1, -.24]


accuracy['imagenet'] = [.55, .52, np.nan, np.nan, .54, .58, .58, .6,\
            np.nan, .6, .55, .57, .58, .56, .58, .52]
forget['imagenet'] = [.03, .02, np.nan, np.nan, -.04, -.01, -.01, .06,\
            np.nan, 0, -.03, -.02, -.07, -.07, -.06, -.12]
transfer['imagenet'] = [.02, .04, np.nan, np.nan, -.07, -.03, -.03, .1,\
            np.nan, -.01, -.06, -.04, -.01, .05, -.04, -.1]

accuracy['speech'] = [.91, .91, np.nan, np.nan, .73, .93, .93, np.nan,\
            np.nan, .76, .72, .72, np.nan, np.nan, np.nan, .73]
forget['speech'] = [.035, .01, np.nan, np.nan, -.28, 0, 0, np.nan,\
            np.nan, -.24, -.29, -.27, np.nan, np.nan, np.nan, -.3]
transfer['speech'] = [.16, .03, np.nan, np.nan, -.24, -.04, -.04, np.nan,\
            np.nan, -.21, -.25, -.25, np.nan, np.nan, np.nan, -.23]


keys = ['accuracy', 'forget', 'transfer']

data['accuracy'] = pd.DataFrame({"cifar":accuracy["cifar"], "5 dataset":accuracy["5 dataset"],\
        "imagenet":accuracy["imagenet"], "speech":accuracy["speech"]})
data['accuracy'].index = algorithms

data['forget'] = pd.DataFrame({"cifar":forget["cifar"], "5 dataset":forget["5 dataset"],\
        "imagenet":forget["imagenet"], "speech":forget["speech"]})
data['forget'].index = algorithms

data['transfer'] = pd.DataFrame({"cifar":transfer["cifar"], "5 dataset":transfer["5 dataset"],\
        "imagenet":transfer["imagenet"], "speech":transfer["speech"]})
data['transfer'].index = algorithms

fig, ax = plt.subplots(1, 3, figsize=(18,8), sharey=True, sharex=True)
#cbar_ax = fig.add_axes([.91, .3, .03, .4])
vmins = [0,-.3,-.3]
vmaxs = [1,.3,.3]
for i in range(3):
    sns.heatmap(data[keys[i]], yticklabels=algorithms,\
                vmin=vmins[i], vmax=vmaxs[i],
             cmap='coolwarm', ax=ax[i],)# cbar=i == 0, \
             #cbar_ax=cbar_ax if i==0 else None)
    ax[i].set_title(keys[i], fontsize=35)
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=45, fontsize=20)
    #ax[i].set_yticklabels(ax[i].get_yticklabels(), fontsize=20)

fig.tight_layout(rect=[0, 0, .9, 1])
plt.savefig('/Users/jayantadey/progressive-learning/result/figs/heatmap_performance.pdf')
# %%
