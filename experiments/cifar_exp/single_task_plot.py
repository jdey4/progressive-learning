#%%
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import numpy as np
import pandas as pd
from itertools import product
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib
#%%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_fte_bte(err, single_err):
    bte = [[] for i in range(10)]
    te = [[] for i in range(10)]
    fte = []
    
    for i in range(10):
        for j in range(i,10):
            #print(err[j][i],j,i)
            bte[i].append(err[i][i]/err[j][i])
            te[i].append(single_err[i]/err[j][i])
                
    for i in range(10):
        fte.append(single_err[i]/err[i][i])
            
            
    return fte,bte,te

def calc_mean_bte(btes,task_num=10,reps=6):
    mean_bte = [[] for i in range(task_num)]


    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            tmp += np.array(btes[i][j])
        
        tmp=tmp/reps
        mean_bte[j].extend(tmp)
            
    return mean_bte     

def calc_mean_te(tes,task_num=10,reps=6):
    mean_te = [[] for i in range(task_num)]

    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            tmp += np.array(tes[i][j])
        
        tmp=tmp/reps
        mean_te[j].extend(tmp)
            
    return mean_te 

def calc_mean_fte(ftes,task_num=10,reps=6):
    fte = np.asarray(ftes)
    
    return list(np.mean(np.asarray(fte),axis=0))

def get_error_matrix(filename):
    multitask_df, single_task_df = unpickle(filename)

    err = [[] for _ in range(10)]

    for ii in range(10):
        err[ii].extend(
            1 - np.array(
                multitask_df[multitask_df['base_task']==ii+1]['accuracy']
                )
            )
    single_err = 1 - np.array(single_task_df['accuracy'])

    return single_err, err

def sum_error_matrix(error_mat1, error_mat2):
    err = [[] for _ in range(10)]

    for ii in range(10):
        err[ii].extend(
            list(
                np.asarray(error_mat1[ii]) +
                np.asarray(error_mat2[ii])
            )
        )
    return err

#%%
### MAIN HYPERPARAMS ###
ntrees = 10
slots = 10
task_num = 10
shifts = 6

#%% code for 500 samples
reps = slots*shifts
single_task = {
               'task1': np.zeros(reps),
               'task2': np.zeros(reps),
               'task3': np.zeros(reps),
               'task4': np.zeros(reps),
               'task5': np.zeros(reps),
               'task6': np.zeros(reps),
               'task7': np.zeros(reps),
               'task8': np.zeros(reps),
               'task9': np.zeros(reps),
               'task10': np.zeros(reps),
            }

count = 0
for slot in range(slots):
    for shift in range(shifts):
            filename1 = './result/result/uf10withrep'+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
            filename2 = './result/result/uf10'+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
            _, single_task_df1 = unpickle(filename1)
            _, single_task_df2 = unpickle(filename2)
            
            for ii in range(10):
                key = 'task'+str(ii+1)
                single_task[key][count] = single_task_df2['accuracy'][ii] - single_task_df1['accuracy'][ii]
            
            count += 1

single_task_data = pd.DataFrame.from_dict(single_task)
single_task_data = pd.melt(single_task_data,var_name='Tasks', value_name='Accuracy')


# %%
fig, ax = plt.subplots(ncols=1, figsize=(8,8))
sns.set_context("talk")

marker_style = ['.', '.', '.', '.', '.', '+', 'o', '*', '.', '+']
clr = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#f781bf", "#f781bf", "#f781bf", "#f781bf", "#b15928", "#b15928"]
c = sns.color_palette(clr, n_colors=10)

sns.stripplot(
    x="Tasks", y="Accuracy", data=single_task_data, palette=c, ax=ax
    )
ax.set_xticklabels(
    ['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10'],
    fontsize=12,rotation=45,ha="right",rotation_mode='anchor'
    )
ax.set_title('without repacement - with replacement')
ax.hlines(0, 0,9, colors='grey', linestyles='dashed',linewidth=1.5)

plt.savefig('result/figs/single_task_accuracy.pdf')

# %%
