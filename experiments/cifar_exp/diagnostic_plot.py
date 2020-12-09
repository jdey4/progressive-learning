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
slots = 10
task_num = 10
shifts = 6
total_alg_top = 4
total_alg_bottom = 8
alg_name_top = ['PLN','PLF','Prog-NN', 'DF-CNN']
alg_name_bottom = ['L2F','LwF','EWC','O-EWC','SI', 'Replay \n (increasing amount)', 'Replay \n (fixed amount)', 'None']
combined_alg_name = ['PLN','PLF','Prog-NN', 'DF-CNN','LwF','EWC','O-EWC','SI', 'Replay \n (increasing amount)', 'Replay \n (fixed amount)', 'None']
model_file_top = ['dnn0withrep','fixed_uf10withrep','Prog_NN','DF_CNN']
model_file_bottom = ['uf10withrep', 'LwF', 'EWC', 'OEWC', 'si', 'offline', 'exact', 'None']
btes_top = [[] for i in range(total_alg_top)]
ftes_top = [[] for i in range(total_alg_top)]
tes_top = [[] for i in range(total_alg_top)]
btes_bottom = [[] for i in range(total_alg_bottom)]
ftes_bottom = [[] for i in range(total_alg_bottom)]
tes_bottom = [[] for i in range(total_alg_bottom)]

#combined_alg_name = ['L2N','L2F','Prog-NN', 'DF-CNN','LwF','EWC','O-EWC','SI', 'Replay (increasing amount)', 'Replay (fixed amount)', 'None']
model_file_combined = ['dnn0withrep','fixed_uf10withrep','Prog_NN','DF_CNN', 'LwF', 'EWC', 'OEWC', 'si', 'offline', 'exact', 'None']

########################
# %%
reps = slots*shifts

for alg in range(total_alg_bottom): 
    count = 0 
    bte_tmp = [[] for _ in range(reps)]
    fte_tmp = [[] for _ in range(reps)] 
    te_tmp = [[] for _ in range(reps)]

    for slot in range(slots):
        for shift in range(shifts):
            if alg < 1:
                filename = './result/result/'+model_file_bottom[alg]+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
            else:
                filename = './benchmarking_algorthms_result/'+model_file_bottom[alg]+'-'+str(slot+1)+'-'+str(shift+1)+'.pickle'

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
    #single_err /= reps
    #err /= reps
    fte, bte, te = get_fte_bte(err,single_err)
    
    btes_bottom[alg].extend(bte)
    ftes_bottom[alg].extend(fte)
    tes_bottom[alg].extend(te)

#%%
te_500 = {'PLN':np.zeros(10,dtype=float), 'PLF':np.zeros(10,dtype=float), 'Prog-NN':np.zeros(10,dtype=float),
        'DF-CNN':np.zeros(10,dtype=float), 'LwF':np.zeros(10,dtype=float), 'EWC':np.zeros(10,dtype=float),
        'O-EWC':np.zeros(10,dtype=float), 'SI':np.zeros(10,dtype=float),
        'Replay (increasing amount)':np.zeros(10,dtype=float), 'Replay (fixed amount)':np.zeros(10,dtype=float), 'None':np.zeros(10,dtype=float)}

for count,name in enumerate(te_500.keys()):
    for i in range(10):
        if count <4:
            te_500[name][i] = tes_top[count][i][9-i]
        elif count>3:
            te_500[name][i] = tes_bottom[count-3][i][9-i]


df_500 = pd.DataFrame.from_dict(te_500)
df_500 = pd.melt(df_500,var_name='Algorithms', value_name='Transfer Efficieny')

# %%
fig = plt.figure(constrained_layout=True,figsize=(29,16))
gs = fig.add_gridspec(16, 29)

clr_top = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3"]
c_top = sns.color_palette(clr_top, n_colors=len(clr_top))

clr_bottom = ["#e41a1c", "#f781bf", "#f781bf", "#f781bf", "#f781bf", "#b15928", "#b15928", "#b15928"]
c_bottom = sns.color_palette(clr_bottom, n_colors=len(clr_bottom))

marker_style_top = ['.', '.', '.', '.']
marker_style_bottom = ['.', '.', '+', 'o', '*', '.', '+', 'o']
marker_style = ['.', '.', '.', '.', '.', '+', 'o', '*', '.', '+', 'o']

clr_combined = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#f781bf", "#f781bf", "#f781bf", "#f781bf", "#b15928", "#b15928", "#b15928"]
c_combined = sns.color_palette(clr_combined, n_colors=total_alg_top+total_alg_bottom)

fontsize=25
ticksize=22
legendsize=14


#########################################################
fig, ax = plt.subplots(1,1, figsize=(8,8))

for i, fte in enumerate(ftes_bottom):
    fte[0] = 1
    if i == 0:
        ax.plot(np.arange(1,11), fte, color=c_bottom[i], marker=marker_style_bottom[i], markersize=12, linewidth=3)
        continue

    if i == 1:
        ax.plot(np.arange(1,11), fte, color=c_bottom[i], marker=marker_style_bottom[i], markersize=12, linewidth=3)
        continue
    
    ax.plot(np.arange(1,11), fte, color=c_bottom[i], marker=marker_style_bottom[i], markersize=12)
    
ax.set_xticks(np.arange(1,11))
ax.set_yticks([0.85, 1, 1.05])
ax.set_ylim(0.8, 1.05)
ax.tick_params(labelsize=ticksize)

ax.set_ylabel('Resource Constrained FTE', fontsize=fontsize)
ax.set_xlabel('Number of tasks seen', fontsize=fontsize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5)


#plt.savefig('result/figs/cifar_exp_500_recruit_with_rep.pdf')

# %%