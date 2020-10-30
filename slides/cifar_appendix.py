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

def stratified_scatter(te_dict,axis_handle,s,color,style):
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
                c='k',
                marker=style[algo_no]
                )

   

#%%
### MAIN HYPERPARAMS ###
ntrees = 10
task_num = 10
shifts = 6
total_alg = 11
alg_name = ['L2N','L2F','Prog-NN', 'DF-CNN','LwF','EWC','O-EWC','SI', 'Replay (increasing amount)', 'Replay (fixed amount)', 'None']
model_file_5000 = ['dnn0','uf5000_40','Prog_NN','DF_CNN', 'LwF', 'EWC', 'OEWC', 'si', 'offline', 'exact', 'None']

btes_5000 = [[] for i in range(total_alg)]
ftes_5000 = [[] for i in range(total_alg)]
tes_5000 = [[] for i in range(total_alg)]
########################

#%% code for 5000 samples
reps = shifts

for alg in range(total_alg): 
    count = 0 
    te_tmp = [[] for _ in range(reps)]
    bte_tmp = [[] for _ in range(reps)]
    fte_tmp = [[] for _ in range(reps)] 

    for shift in range(shifts):
        if alg < 2:
            filename = '../experiments/cifar_exp/result/result/'+model_file_5000[alg]+'_'+str(shift+1)+'_0'+'.pickle'
        elif alg<4:
            filename = '../experiments/cifar_exp/benchmarking_algorthms_result/'+model_file_5000[alg]+'_'+str(shift+1)+'.pickle'
        else:
            filename = '../experiments/cifar_exp/benchmarking_algorthms_result/'+model_file_5000[alg]+'-'+str(shift+1)+'.pickle'

        multitask_df, single_task_df = unpickle(filename)

        single_err_, err_ = get_error_matrix(filename)
            
        if count == 0:
            single_err, err = single_err_, err_
        else:
            err = sum_error_matrix(err, err_)
            single_err = list(
                np.asarray(single_err) + np.asarray(single_err_)
            )

            count += 1
    
    fte, bte, te = get_fte_bte(err,single_err)

    btes_5000[alg].extend(bte)
    ftes_5000[alg].extend(fte)
    tes_5000[alg].extend(te)



# %%
fig = plt.figure(constrained_layout=True,figsize=(20,6))
gs = fig.add_gridspec(6, 20)


marker_style = ['.', '.', '.', '.', '.', '+', 'o', '*', '.', '+', 'o']

clr_combined = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#f781bf", "#f781bf", "#f781bf", "#f781bf", "#b15928", "#b15928", "#b15928"]
c_combined = sns.color_palette(clr_combined, n_colors=len(alg_name))

fontsize=25
ticksize=22
legendsize=14

ax = fig.add_subplot(gs[:6,:6])

for i, fte in enumerate(ftes_5000):
    if i == 0:
        ax.plot(np.arange(1,11), fte, color=c_combined[i], marker=marker_style[i], markersize=12, label=alg_name[i], linewidth=3)
        continue

    if i == 1:
        ax.plot(np.arange(1,11), fte, color=c_combined[i], marker=marker_style[i], markersize=12, label=alg_name[i], linewidth=3)
        continue
    
    ax.plot(np.arange(1,11), fte, color=c_combined[i], marker=marker_style[i], markersize=12, label=alg_name[i])
    
ax.set_xticks(np.arange(1,11))
#ax.set_yticks([0.9, 1, 1.1, 1.2, 1.3,1.4])
#ax.set_ylim(0.89, 1.41)
ax.tick_params(labelsize=ticksize)

ax.set_ylabel('Forward Transfer Efficiency (FTE)', fontsize=fontsize)
ax.set_xlabel('Number of tasks seen', fontsize=fontsize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5)

#########################################################
ax = fig.add_subplot(gs[:6,7:13])

for i in range(task_num - 1):

    et = np.zeros((total_alg,task_num-i))

    for j in range(0,total_alg):
        et[j,:] = np.asarray(btes_5000[j][i])

    ns = np.arange(i + 1, task_num + 1)
    for j in range(0,total_alg):
        if j == 0:
            if i == 0:
                ax.plot(ns, et[j,:], marker=marker_style[j], markersize=8, label = alg_name[j], color=c_combined[j], linewidth = 3)
            else:
                ax.plot(ns, et[j,:], marker=marker_style[j], markersize=8, color=c_combined[j], linewidth = 3)
        elif j == 1:
            if i == 0:
                ax.plot(ns, et[j,:], marker=marker_style[j], markersize=8, label = alg_name[j], color=c_combined[j], linewidth = 3)
            else:
                ax.plot(ns, et[j,:], marker=marker_style[j], markersize=8, color=c_combined[j], linewidth = 3)
        else:
            if i == 0:
                ax.plot(ns, et[j,:], marker=marker_style[j], markersize=8, label = alg_name[j], color=c_combined[j])
            else:
                ax.plot(ns, et[j,:], marker=marker_style[j], markersize=8, color=c_combined[j])



ax.set_xlabel('Number of tasks seen', fontsize=fontsize)
ax.set_ylabel('Backward Transfer Efficiency (BTE)', fontsize=fontsize)

#ax.set_yticks([.4,.6,.8,.9,1, 1.1,1.2])
ax.set_xticks(np.arange(1,11))
#ax.set_ylim(0.96, 1.2)
ax.tick_params(labelsize=ticksize)
#ax[0][1].grid(axis='x')

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5)

handles, labels_ = ax.get_legend_handles_labels()

fig.legend(handles, labels_, bbox_to_anchor=(1, .93), fontsize=legendsize+7, frameon=False)


plt.savefig('figs/cifar5000.pdf')
# %%
