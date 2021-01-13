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

def accuracy_mean_over_task(err, reps):
    acc = np.zeros(10,dtype=float)

    for ii, err_ in enumerate(err):
        acc[ii] = 1 - np.mean(err_)/reps

    return acc

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
slots = 10
task_num = 10
shifts = 6
total_alg = 11
alg_name = ['PLN','PLF','ProgNN', 'DF-CNN', 'LwF','EWC','O-EWC','SI', 'Full replay', 'Replay \n (fixed)', 'None']
model_file = ['dnn0withrep','fixed_uf10withrep','Prog_NN','DF_CNN', 'LwF', 'EWC', 'OEWC', 'si', 'offline', 'exact', 'None']
########################

#%% code for 500 samples
reps = slots*shifts
acc = [[] for _ in range(total_alg)]
acc_all = [[] for _ in range(total_alg)]

for alg in range(total_alg): 
    count = 0 

    for slot in range(slots):
        for shift in range(shifts):
            if alg < 2:
                filename = './result/result/'+model_file[alg]+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
            elif alg == 2 or alg == 3:
                filename = './benchmarking_algorthms_result/'+model_file[alg]+'-'+str(shift+1)+'-'+str(slot+1)+'.pickle'
            else:
                filename = './benchmarking_algorthms_result/'+model_file[alg]+'-'+str(slot+1)+'-'+str(shift+1)+'.pickle'

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
    acc_all[alg].extend(err)
    acc[alg].extend(accuracy_mean_over_task(err, reps))

# %%
fig, ax = plt.subplots(1,1, figsize=(8,8))
sns.set_context('talk')
marker_style = ['.', '.', '.', '.', '.', '+', 'o', '*', '.', '+', 'o']
clr = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#f781bf", "#f781bf", "#f781bf", "#f781bf", "#b15928", "#b15928", "#b15928"]
c = sns.color_palette(clr, n_colors=total_alg)

fontsize=25
ticksize=22
legendsize=14

sample_no = list(range(500,5500,500))
for i in range(total_alg):
    ax.plot(sample_no, acc[i], color=c[i], label=alg_name[i], marker=marker_style[i])

ax.legend()
ax.set_xlabel('Sample #')
ax.set_ylabel('Accuracy')
plt.savefig('result/figs/accuracy.pdf')
# %%
fig, ax = plt.subplots(1,1, figsize=(8,8))
sns.set_context('talk')
marker_style = ['.', '.', '.', '.', '.', '+', 'o', '*', '.', '+', 'o']
clr = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#f781bf", "#f781bf", "#f781bf", "#f781bf", "#b15928", "#b15928", "#b15928"]
c = sns.color_palette(clr, n_colors=total_alg)

algos = [0,1,4]

for alg in algos:
    for i in range(10):
        et = np.zeros(10-i,dtype=float)

        for j in range(i,10):
            et[j-i] = 1-acc_all[alg][j][i]/reps
        print(et)
        #ns = np.arange(i,10)
        if i==0:
            ax.plot(sample_no[i:10], et, c=clr[alg], label=alg_name[alg], marker = marker_style[alg])
        else:
            ax.plot(sample_no[i:10], et, c=clr[alg], marker = marker_style[alg])

ax.set_xlabel('Sample #')
ax.set_ylabel('Accuracy')
ax.legend()
plt.savefig('result/figs/accuracy_top3.pdf')
# %%
