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
total_alg_top = 4
total_alg_bottom = 5
alg_name_top = ['L2N','L2F','Prog-NN', 'DF-CNN']
alg_name_bottom = ['L2F','LwF','EWC','O-EWC','SI']
combined_alg_name = ['L2N','L2F','Prog-NN', 'DF-CNN','LwF','EWC','O-EWC','SI']
model_file_top = ['dnn0','fixed_uf10','Prog_NN','DF_CNN']
model_file_bottom = ['uf10','LwF','EWC','Online_EWC','SI']
btes_top = [[] for i in range(total_alg_top)]
ftes_top = [[] for i in range(total_alg_top)]
tes_top = [[] for i in range(total_alg_top)]
btes_bottom = [[] for i in range(total_alg_bottom)]
ftes_bottom = [[] for i in range(total_alg_bottom)]
tes_bottom = [[] for i in range(total_alg_bottom)]

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
            if alg < 2:
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
reps = slots*shifts

for alg in range(total_alg_bottom): 
    count = 0 
    bte_tmp = [[] for _ in range(reps)]
    fte_tmp = [[] for _ in range(reps)] 
    te_tmp = [[] for _ in range(reps)]

    for slot in range(slots):
        for shift in range(shifts):
            if alg < 1:
                filename = '../experiments/cifar_exp/result/result/'+model_file_bottom[alg]+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
            else:
                filename = '../experiments/cifar_exp/benchmarking_algorthms_result/'+model_file_bottom[alg]+'_'+str(shift+1)+'_'+str(slot)+'.pickle'

            multitask_df, single_task_df = unpickle(filename)

            single_err, err = get_error_matrix(filename)
            fte, bte, te = get_fte_bte(err,single_err)
            
            bte_tmp[count].extend(bte)
            fte_tmp[count].extend(fte)
            te_tmp[count].extend(te)
            count+=1
    
    btes_bottom[alg].extend(calc_mean_bte(bte_tmp,reps=reps))
    ftes_bottom[alg].extend(calc_mean_fte(fte_tmp,reps=reps))
    tes_bottom[alg].extend(calc_mean_te(te_tmp,reps=reps))

#%%
te_500 = {'L2N':np.zeros(10,dtype=float), 'L2F':np.zeros(10,dtype=float), 'Prog-NN':np.zeros(10,dtype=float),
        'DF-CNN':np.zeros(10,dtype=float), 'LwF':np.zeros(10,dtype=float),
        'EWC':np.zeros(10,dtype=float), 'Online EWC':np.zeros(10,dtype=float), 'SI':np.zeros(10,dtype=float)}

for count,name in enumerate(te_500.keys()):
    for i in range(10):
        if count <4:
            te_500[name][i] = tes_top[count][i][9-i]
        elif count>3:
            te_500[name][i] = tes_bottom[count-3][i][9-i]

df_500 = pd.DataFrame.from_dict(te_500)
df_500 = pd.melt(df_500,var_name='Algorithms', value_name='Transfer Efficieny')

# %%
fig = plt.figure(constrained_layout=True,figsize=(22,16))
gs = fig.add_gridspec(16, 22)

clr_top = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3"]
c_top = sns.color_palette(clr_top, n_colors=len(clr_top))

clr_bottom = ["#e41a1c", "#ff7f00", "#ffff33", "#a65628", "#f781bf"]
c_bottom = sns.color_palette(clr_bottom, n_colors=len(clr_bottom))

clr_combined = [ "#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf"]
c_combined = sns.color_palette(clr_combined, n_colors=total_alg_top+total_alg_bottom)

fontsize=25
ticksize=22
legendsize=14

ax = fig.add_subplot(gs[:6,:6])

for i, fte in enumerate(ftes_top):
    if i == 0:
        ax.plot(np.arange(1,11), fte, color=c_top[i], marker='.', markersize=12, label=alg_name_top[i], linewidth=3)
        continue

    if i == 1:
        ax.plot(np.arange(1,11), fte, color=c_top[i], marker='.', markersize=12, label=alg_name_top[i], linewidth=3)
        continue
    
    ax.plot(np.arange(1,11), fte, color=c_top[i], marker='.', markersize=12, label=alg_name_top[i])
    
ax.set_xticks(np.arange(1,11))
ax.set_yticks([0.9, 1, 1.1, 1.2, 1.3,1.4])
ax.set_ylim(0.89, 1.41)
ax.tick_params(labelsize=ticksize)

ax.set_ylabel('Forward Transfer Efficiency (FTE)', fontsize=fontsize)
ax.set_xlabel('Number of tasks seen', fontsize=fontsize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5)

#########################################################
ax = fig.add_subplot(gs[7:13,:6])

for i, fte in enumerate(ftes_bottom):
    if i == 0:
        ax.plot(np.arange(1,11), fte, color=c_bottom[i], marker='.', markersize=12, linewidth=3)
        continue

    if i == 1:
        ax.plot(np.arange(1,11), fte, color=c_bottom[i], marker='.', markersize=12, linewidth=3)
        continue
    
    ax.plot(np.arange(1,11), fte, color=c_bottom[i], marker='.', markersize=12)
    
ax.set_xticks(np.arange(1,11))
ax.set_yticks([0.95, 1, 1.05])
ax.set_ylim(0.95, 1.05)
ax.tick_params(labelsize=ticksize)

ax.set_ylabel('Resource Constrained FTE', fontsize=fontsize)
ax.set_xlabel('Number of tasks seen', fontsize=fontsize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5)

for i in range(0,total_alg_top+total_alg_bottom-1):
    ax.plot(1,0,color=c_combined[i], marker='.', markersize=8,label=combined_alg_name[i])

#handles, labels_ = ax.get_legend_handles_labels()
#fig.legend(handles, labels_, bbox_to_anchor=(.99, .93), fontsize=legendsize+12, frameon=False)

##################################################

clr_top = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3"]
c_top = sns.color_palette(clr_top, n_colors=len(clr_top))

clr_bottom = ["#e41a1c", "#ff7f00", "#ffff33", "#a65628", "#f781bf"]
c_bottom = sns.color_palette(clr_bottom, n_colors=len(clr_bottom))

clr_combined = [ "#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf"]
c_combined = sns.color_palette(clr_combined, n_colors=total_alg_top+total_alg_bottom)

fontsize=25
ticksize=22
legendsize=14


#ax[0][0].grid(axis='x')
ax = fig.add_subplot(gs[:6,7:14])

for i in range(task_num - 1):

    et = np.zeros((total_alg_top,task_num-i))

    for j in range(0,total_alg_top):
        et[j,:] = np.asarray(btes_top[j][i])

    ns = np.arange(i + 1, task_num + 1)
    for j in range(0,total_alg_top):
        if j == 0:
            if i == 0:
                ax.plot(ns, et[j,:], marker='.', markersize=8, label = alg_name_top[j], color=c_top[j], linewidth = 3)
            else:
                ax.plot(ns, et[j,:], marker='.', markersize=8, color=c_top[j], linewidth = 3)
        elif j == 1:
            if i == 0:
                ax.plot(ns, et[j,:], marker='.', markersize=8, label = alg_name_top[j], color=c_top[j], linewidth = 3)
            else:
                ax.plot(ns, et[j,:], marker='.', markersize=8, color=c_top[j], linewidth = 3)
        else:
            if i == 0:
                ax.plot(ns, et[j,:], marker='.', markersize=8, label = alg_name_top[j], color=c_top[j])
            else:
                ax.plot(ns, et[j,:], marker='.', markersize=8, color=c_top[j])


for i in range(total_alg_top,total_alg_top+total_alg_bottom-1):
    ax.plot(1,0,color=c_combined[i], marker='.', markersize=8,label=combined_alg_name[i])

ax.set_xlabel('Number of tasks seen', fontsize=fontsize)
ax.set_ylabel('Backward Transfer Efficiency (BTE)', fontsize=fontsize)

ax.set_yticks([.4,.6,.8,.9,1, 1.1,1.2])
ax.set_xticks(np.arange(1,11))
ax.set_ylim(0.99, 1.2)
ax.tick_params(labelsize=ticksize)
#ax[0][1].grid(axis='x')

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5)

handles, labels_ = ax.get_legend_handles_labels()
ax.legend(loc='center left', bbox_to_anchor=(.99, 0.3), fontsize=legendsize+12,frameon=False)


#########################################################
ax = fig.add_subplot(gs[7:13,7:14])

for i in range(task_num - 1):

    et = np.zeros((total_alg_bottom,task_num-i))

    for j in range(0,total_alg_bottom):
        et[j,:] = np.asarray(btes_bottom[j][i])

    ns = np.arange(i + 1, task_num + 1)
    for j in range(0,total_alg_bottom):
        if j == 0:
            if i == 0:
                ax.plot(ns, et[j,:], marker='.', markersize=8, label=None, color=c_bottom[j], linewidth = 3)
            else:
                ax.plot(ns, et[j,:], marker='.', markersize=8, color=c_bottom[j], linewidth = 3)
        elif j == 1:
            if i == 0:
                ax.plot(ns, et[j,:], marker='.', markersize=8, label = alg_name_bottom[j], color=c_bottom[j])
            else:
                ax.plot(ns, et[j,:], marker='.', markersize=8, color=c_bottom[j], linewidth = 3)
        else:
            if i == 0:
                ax.plot(ns, et[j,:], marker='.', markersize=8, label = alg_name_bottom[j], color=c_bottom[j])
            else:
                ax.plot(ns, et[j,:], marker='.', markersize=8, color=c_bottom[j])


ax.set_xlabel('Number of tasks seen', fontsize=fontsize)
ax.set_ylabel('Resource Constrained BTE', fontsize=fontsize)

ax.set_yticks([.4,.6,.8,.9,1, 1.1,1.2])
ax.set_xticks(np.arange(1,11))
ax.set_ylim(0.85, 1.17)
ax.tick_params(labelsize=ticksize)
#ax[0][1].grid(axis='x')

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5)


#fig.legend(handles, labels_, bbox_to_anchor=(.8, 1), fontsize=legendsize+12, frameon=False)


fontsize=30
labelsize=28

ax = fig.add_subplot(gs[7:13,15:21])
ax.tick_params(labelsize=22)
ax_ = sns.boxplot(
    x="Algorithms", y="Transfer Efficieny", data=df_500, palette=c_combined, whis=np.inf,
    ax=ax, showfliers=False, notch=1
    )
ax.hlines(1, -1,8, colors='grey', linestyles='dashed',linewidth=1.5)
#sns.boxplot(x="Algorithms", y="Transfer Efficieny", data=mean_df, palette=c, linewidth=3, ax=ax[1][1])
#ax_=sns.pointplot(x="Algorithms", y="Transfer Efficieny", data=df_500, join=False, color='grey', linewidth=1.5, ci='sd',ax=ax)
#ax_.set_yticks([.4,.6,.8,1, 1.2,1.4])
ax_.set_xlabel('', fontsize=fontsize)
ax.set_ylabel('Transfer Efficiency after 10 Tasks', fontsize=fontsize-5)
ax_.set_xticklabels(
    ['L2N','L2F','Prog-NN','DF-CNN','LwF','EWC','O-EWC','SI'],
    fontsize=20,rotation=45,ha="right",rotation_mode='anchor'
    )

stratified_scatter(te_500,ax,16,c_combined)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

plt.savefig('figs/darpa.pdf')

# %%
