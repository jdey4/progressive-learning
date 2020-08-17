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

def stratified_scatter(te_dict,axis_handle,s,color):
    algo = list(te_dict.keys())
    total_alg = len(algo)

    total_points = len(te_dict[algo[0]])

    pivot_points = np.arange(-.25, (total_alg+1)*1, step=1)
    interval = .7/(total_points-1)

    for algo_no,alg in enumerate(algo):
        for no,points in enumerate(te_dict[alg]):
            axis_handle.scatter(
                pivot_points[algo_no]+interval*no,
                te_dict[alg][no],
                s=s,
                c='k'
                )


#%%
### MAIN HYPERPARAMS ###
ntrees = 10
slots = 10
task_num = 10
shifts = 6
total_alg_top = 9
total_alg_bottom = 5
alg_name_top = ['L2N','L2F','RC-L2F','Prog-NN', 'DF-CNN','LwF','EWC','O-EWC','SI']
model_file_top = ['dnn0','fixed_uf10','uf10','Prog_NN','DF_CNN','LwF','EWC','Online_EWC','SI']
btes_top = [[] for i in range(total_alg_top)]
ftes_top = [[] for i in range(total_alg_top)]
tes_top = [[] for i in range(total_alg_top)]

########################

#%% code for 500 samples
reps = slots*shifts

for alg in range(total_alg_top): 
    count = 0 
    bte_tmp = [[] for _ in range(reps)]
    fte_tmp = [[] for _ in range(reps)] 
    te_tmp = [[] for _ in range(reps)] 

    for slot in range(slots):
        for shift in range(shifts):
            if alg < 2 or alg == 2:
                filename = '../experiments/cifar_exp/result/result/'+model_file_top[alg]+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
            else:
                filename = '../experiments/cifar_exp/benchmarking_algorthms_result/'+model_file_top[alg]+'_'+str(shift+1)+'_'+str(slot)+'.pickle'

            multitask_df, single_task_df = unpickle(filename)

            single_err, err = get_error_matrix(filename)
            fte, bte, te = get_fte_bte(err,single_err)
            
            bte_tmp[count].extend(bte)
            fte_tmp[count].extend(fte)
            te_tmp[count].extend(te)
            count+=1
    
    btes_top[alg].extend(calc_mean_bte(bte_tmp,reps=reps))
    ftes_top[alg].extend(calc_mean_fte(fte_tmp,reps=reps))
    tes_top[alg].extend(calc_mean_te(te_tmp,reps=reps))

# %%
fig = plt.figure(constrained_layout=True,figsize=(14,6))
gs = fig.add_gridspec(6, 14)

clr_top = [ "#377eb8", "#e41a1c", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf"]
c_top = sns.color_palette(clr_top, n_colors=len(clr_top))

fontsize=25
ticksize=22
legendsize=18

ax = fig.add_subplot(gs[:6,:6])

for i, fte in enumerate(ftes_top):
    if i == 0:
        ax.plot(np.arange(1,11), fte, color=c_top[i], marker='.', markersize=12, label=alg_name_top[i], linewidth=3)
        continue

    if i == 1:
        ax.plot(np.arange(1,11), fte, color=c_top[i], marker='.', markersize=12, label=alg_name_top[i], linewidth=3)
        continue
    
    if i == 2:
        ax.plot(np.arange(1,11), fte, color=c_top[i], marker='.', linestyle='dashed', markersize=12, label=alg_name_top[i], linewidth=3)
        continue

    ax.plot(np.arange(1,11), fte, color=c_top[i], marker='.', markersize=12, label=alg_name_top[i])
    
ax.set_xticks(np.arange(1,11))
ax.set_yticks([0.9, 1, 1.1, 1.2, 1.3,1.4])
ax.set_ylim(0.89, 1.41)
ax.tick_params(labelsize=ticksize)

ax.set_ylabel('Forward Transfer Efficiency', fontsize=fontsize)
ax.set_xlabel('Number of tasks seen', fontsize=fontsize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5)
#########################################################
ax = fig.add_subplot(gs[:6,7:13])

for i in range(task_num - 1):

    et = np.zeros((total_alg_top,task_num-i))

    for j in range(0,total_alg_top):
        et[j,:] = np.asarray(btes_top[j][i])

    ns = np.arange(i + 1, task_num + 1)
    for j in range(0,total_alg_top):
        if j == 0:
            if i == 0:
                ax.plot(ns, et[j,:], marker='.', markersize=8, label=alg_name_top[j], color=c_top[j], linewidth = 3)
            else:
                ax.plot(ns, et[j,:], marker='.', markersize=8, color=c_top[j], linewidth = 3)
        elif j == 1:
            if i == 0:
                ax.plot(ns, et[j,:], marker='.', markersize=8, label = alg_name_top[j], color=c_top[j])
            else:
                ax.plot(ns, et[j,:], marker='.', markersize=8, color=c_top[j], linewidth = 3)
        elif j == 2:
            if i == 0:
                ax.plot(ns, et[j,:], linestyle='dashed', marker='.', markersize=8, label = alg_name_top[j], color=c_top[j])
            else:
                ax.plot(ns, et[j,:], marker='.', markersize=8, color=c_top[j], linewidth = 3)     
        else:
            if i == 0:
                ax.plot(ns, et[j,:], marker='.', markersize=8, label = alg_name_top[j], color=c_top[j])
            else:
                ax.plot(ns, et[j,:], marker='.', markersize=8, color=c_top[j])


ax.set_xlabel('Number of tasks seen', fontsize=fontsize)
ax.set_ylabel('Backward Transfer Efficiency', fontsize=fontsize)

ax.set_yticks([.4,.6,.8,.9,1, 1.1,1.2])
ax.set_xticks(np.arange(1,11))
ax.set_ylim(0.85, 1.23)
ax.tick_params(labelsize=ticksize)
#ax[0][1].grid(axis='x')

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5)

ax.legend(bbox_to_anchor=(1, .8),fontsize=legendsize, frameon=False)
plt.savefig('figs/cifar_benchmark.png',dpi=300)
# %%
