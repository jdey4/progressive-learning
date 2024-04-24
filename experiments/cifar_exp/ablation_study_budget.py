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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import register_cmap
#%%
def register_palette(name, clr):
    # relative positions of colors in cmap/palette 
    pos = [0.0,1.0]

    colors=['#FFFFFF', clr]
    cmap = LinearSegmentedColormap.from_list("", list(zip(pos, colors)))
    register_cmap(name, cmap)

def calc_forget(err, reps, total_task=10):
#Tom Vient et al
    forget = 0
    for ii in range(total_task-1):
        forget += err[ii][ii] - err[total_task-1][ii]

    forget /= (total_task-1)
    return forget/reps

def calc_transfer(err, single_err, reps, total_task=10):
#Tom Vient et al
    transfer = np.zeros(total_task,dtype=float)

    for ii in range(total_task):
        transfer[ii] = (single_err[ii] - err[total_task-1][ii])/reps

    return np.mean(transfer)

def calc_acc(err, reps, total_task=10):
#Tom Vient et al
    acc = 0
    for ii in range(total_task):
        acc += (1-err[total_task-1][ii]/reps)
    return acc/total_task

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def calc_final_acc(err, reps):
    avg_acc = []
    for err_ in err[-1][::-1]:
        avg_acc.append(1-err_/reps)
    
    return avg_acc

def calc_avg_acc(err, reps):
    avg_acc = np.zeros(10, dtype=float)
    avg_var = np.zeros(10, dtype=float)
    for i in range(10):
        avg_acc[i] = (1*(i+1) - np.sum(err[i])/reps + (9-i)*.1)/10
        avg_var[i] = np.var(1-np.array(err[i])/reps)
    return avg_acc, avg_var

def calc_avg_single_acc(err, reps):
    avg_acc = np.zeros(10, dtype=float)
    avg_var = np.zeros(10, dtype=float)
    for i in range(10):
        avg_acc[i] = (1*(i+1) - np.sum(err[:i+1])/reps + (9-i)*.1)/10
        avg_var[i] = np.var(1-np.array(err[:i+1])/reps)
    return avg_acc, avg_var

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
                c='k'
                )
            
#%%
### MAIN HYPERPARAMS ###
ntrees = 10
slots = 10
task_num = 10
shifts = 6
total_alg_top = 11
alg_name_top = ['SiLLy-N-4', 'EWC', 'Total Replay', 'Partial Replay', 'LwF','O-EWC','SI', 'ER', 'A-GEM', 'TAG', 'None']
model_file_top = ['dnn_budget', 'EWC', 'offline', 'exact', 'LwF', 'OEWC', 'si', 'er', 'agem', 'tag', 'None']
btes_top = [[] for i in range(total_alg_top)]
ftes_top = [[] for i in range(total_alg_top)]
tes_top = [[] for i in range(total_alg_top)]
acc_top = [[] for i in range(total_alg_top)]

########################
# %%
reps = slots*shifts

for alg in range(total_alg_top): 
    count = 0 
    bte_tmp = [[] for _ in range(reps)]
    fte_tmp = [[] for _ in range(reps)] 
    te_tmp = [[] for _ in range(reps)]

    for slot in range(slots):
        for shift in range(shifts):
            if alg < 1:
                filename = './result/result/'+model_file_top[alg]+'_'+str(shift)+'_'+str(slot)+'.pickle'
            else:
                filename = './benchmarking_algorthms_result/'+model_file_top[alg]+'-'+str(slot+1)+'-'+str(shift+1)+'.pickle'

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
    avg_acc = calc_final_acc(err, reps)

    btes_top[alg].extend(bte)
    ftes_top[alg].extend(fte)
    tes_top[alg].extend(te)
    acc_top[alg].extend(avg_acc)
#%%
fte_top_end = {'SiLLy-N-4':np.zeros(10,dtype=float), 'EWC':np.zeros(10,dtype=float),
               'Total Replay':np.zeros(10,dtype=float),
               'Partial Replay':np.zeros(10,dtype=float), 'LwF':np.zeros(10,dtype=float),
               'O-EWC':np.zeros(10,dtype=float), 'SI':np.zeros(10,dtype=float),
               'ER':np.zeros(10,dtype=float), 'A-GEM':np.zeros(10,dtype=float),
               'TAG':np.zeros(10,dtype=float), 'None':np.zeros(10,dtype=float)}

task_order = []
t = 1
for count,name in enumerate(fte_top_end.keys()):
    #print(name, count)
    for i in range(10):
        fte_top_end[name][9-i] = np.log(ftes_top[count][i])
        task_order.append(t)
        t += 1

tmp_fle = {}
for id in fte_top_end.keys():
    tmp_fle[id] = fte_top_end[id]

df_fle = pd.DataFrame.from_dict(tmp_fle)
df_fle = pd.melt(df_fle,var_name='Algorithms', value_name='Forward Transfer Efficieny')
df_fle.insert(2, "Task ID", task_order)
#%%
bte_end = {'SiLLy-N-4':np.zeros(10,dtype=float), 'EWC':np.zeros(10,dtype=float),
           'Total Replay':np.zeros(10,dtype=float),
           'Partial Replay':np.zeros(10,dtype=float), 'LwF':np.zeros(10,dtype=float),
           'O-EWC':np.zeros(10,dtype=float), 'SI':np.zeros(10,dtype=float),
           'ER':np.zeros(10,dtype=float), 'A-GEM':np.zeros(10,dtype=float),
           'TAG':np.zeros(10,dtype=float), 'None':np.zeros(10,dtype=float)}


for count,name in enumerate(bte_end.keys()):
    #print(name, count)
    for i in range(10):
        bte_end[name][9-i] = np.log(btes_top[count][i][9-i])

tmp_ble = {}
for id in alg_name_top:
    tmp_ble[id] = bte_end[id]

df_ble = pd.DataFrame.from_dict(tmp_ble)
df_ble = pd.melt(df_ble,var_name='Algorithms', value_name='Backward Transfer Efficieny')
df_ble.insert(2, "Task ID", task_order)

#%%
te_end = {'SiLLy-N-4':np.zeros(10,dtype=float), 'EWC':np.zeros(10,dtype=float),
           'Total Replay':np.zeros(10,dtype=float),
           'Partial Replay':np.zeros(10,dtype=float), 'LwF':np.zeros(10,dtype=float),
           'O-EWC':np.zeros(10,dtype=float), 'SI':np.zeros(10,dtype=float),
           'ER':np.zeros(10,dtype=float), 'A-GEM':np.zeros(10,dtype=float),
           'TAG':np.zeros(10,dtype=float), 'None':np.zeros(10,dtype=float)}


for count,name in enumerate(bte_end.keys()):
    #print(name, count)
    for i in range(10):
        te_end[name][9-i] = np.log(tes_top[count][i][9-i])

tmp_le = {}
for id in alg_name_top:
    tmp_le[id] = te_end[id]

df_le = pd.DataFrame.from_dict(tmp_le)
df_le = pd.melt(df_le,var_name='Algorithms', value_name='Transfer Efficiency')
df_le.insert(2, "Task ID", task_order)
#%%
acc_end = {'SiLLy-N-4':np.zeros(10,dtype=float), 'EWC':np.zeros(10,dtype=float),
           'Total Replay':np.zeros(10,dtype=float),
           'Partial Replay':np.zeros(10,dtype=float), 'LwF':np.zeros(10,dtype=float),
           'O-EWC':np.zeros(10,dtype=float), 'SI':np.zeros(10,dtype=float),
           'ER':np.zeros(10,dtype=float), 'A-GEM':np.zeros(10,dtype=float),
           'TAG':np.zeros(10,dtype=float), 'None':np.zeros(10,dtype=float)}


for count,name in enumerate(bte_end.keys()):
    #print(name, count)
    for i in range(10):
        acc_end[name][i] = acc_top[count][i]

tmp_acc = {}
for id in alg_name_top:
    tmp_acc[id] = acc_end[id]

df_acc = pd.DataFrame.from_dict(tmp_acc)
df_acc = pd.melt(df_acc,var_name='Algorithms', value_name='Accuracy')
df_acc.insert(2, "Task ID", task_order)
#%%
### MAIN HYPERPARAMS ###
ntrees = 10
slots = 10
task_num = 10
shifts = 6
total_alg_top_replay = 4
total_alg_bottom_replay = 4
alg_name_top_replay = ['0.4', '0.6', '0.8', '1.0']
combined_alg_name_replay = ['0.4', '0.6', '0.8', '1.0']
model_file_top_replay = ['dnn0']
model_file_bottom_replay = ['uf10']
samples_to_replay = [.4,.6,.8,1]

btes_top_replay = [[] for i in range(total_alg_top_replay)]
ftes_top_replay = [[] for i in range(total_alg_top_replay)]
tes_top_replay = [[] for i in range(total_alg_top_replay)]
acc_top_replay = [[] for i in range(total_alg_top_replay)]

model_file_combined_replay = ['dnn0']

########################

#%% code for 500 samples
reps = slots*shifts

for alg, samples in enumerate(samples_to_replay):

    count = 0 
    bte_tmp = [[] for _ in range(reps)]
    fte_tmp = [[] for _ in range(reps)] 
    te_tmp = [[] for _ in range(reps)]

    for slot in range(slots):
        for shift in range(shifts):
            filename = './controlled_replay_result/'+model_file_top_replay[0]+'_'+str(shift+1)+'_'+str(slot)+'_'+str(samples)+'.pickle'

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
    avg_acc = calc_final_acc(err, reps)
    
    btes_top_replay[alg].extend(bte)
    ftes_top_replay[alg].extend(fte)
    tes_top_replay[alg].extend(te)
    acc_top_replay[alg].extend(avg_acc)
    
#%%
fte_replay_end = {'.4':np.zeros(10,dtype=float), '.6':np.zeros(10,dtype=float), 
                '.8':np.zeros(10,dtype=float), '1':np.zeros(10,dtype=float)}

task_order = []
t = 1
for count,name in enumerate(fte_replay_end.keys()):
    print(name, count)
    for i in range(10):
        fte_replay_end[name][9-i] = np.log(ftes_top_replay[0][i])
        task_order.append(t+1)
        t += 1
    

tmp_fle = {}
for id in fte_replay_end.keys():
    tmp_fle[id] = fte_replay_end[id]

df_fle_replay = pd.DataFrame.from_dict(tmp_fle)
df_fle_replay = pd.melt(df_fle_replay,var_name='Algorithms', value_name='Forward Transfer Efficieny')
df_fle_replay.insert(2, "Task ID", task_order)
#%%
bte_end_replay = {'.4':np.zeros(10,dtype=float), '.6':np.zeros(10,dtype=float), 
                '.8':np.zeros(10,dtype=float), '1':np.zeros(10,dtype=float)}


for count,name in enumerate(bte_end_replay.keys()):
    #print(name, count)
    for i in range(10):
        bte_end_replay[name][9-i] = np.log(btes_top_replay[count][i][9-i])
        
tmp_ble = {}
for id in bte_end_replay.keys():
    tmp_ble[id] = bte_end_replay[id]

df_ble_replay = pd.DataFrame.from_dict(tmp_ble)
df_ble_replay = pd.melt(df_ble_replay,var_name='Algorithms', value_name='Backward Transfer Efficieny')
df_ble_replay.insert(2, "Task ID", task_order)
#%%
te_500_replay = {'.4':np.zeros(10,dtype=float), '.6':np.zeros(10,dtype=float), 
                '.8':np.zeros(10,dtype=float), '1':np.zeros(10,dtype=float)}

for count,name in enumerate(te_500_replay.keys()):
    #print(name, count)
    for i in range(10):
        te_500_replay[name][9-i] = np.log(tes_top_replay[count][i][9-i])



df_500_replay = pd.DataFrame.from_dict(te_500_replay)
df_500_replay = pd.melt(df_500_replay,var_name='Algorithms', value_name='Learning Efficieny')
df_500_replay.insert(2, "Task ID", task_order)

#%%
acc_replay_end = {'.4':np.zeros(10,dtype=float), '.6':np.zeros(10,dtype=float), 
                '.8':np.zeros(10,dtype=float), '1':np.zeros(10,dtype=float)}

task_order = []
t = 1
for count,name in enumerate(acc_replay_end.keys()):
    print(name, count)
    for i in range(10):
        acc_replay_end[name][9-i] = acc_top_replay[count][i]
        task_order.append(t+1)
        t += 1
    

tmp_acc = {}
for id in acc_replay_end.keys():
    tmp_acc[id] = acc_replay_end[id]

df_acc_replay = pd.DataFrame.from_dict(tmp_acc)
df_acc_replay = pd.melt(df_acc_replay,var_name='Algorithms', value_name='Accuracy')
df_acc_replay.insert(2, "Task ID", task_order)
#%%
universal_clr_dict = {'SiLLy-N-4': 'r',
                      'SiLLy-N': 'r',
                      'EWC': '#4daf4a',
                      'Total Replay': '#b15928',
                      'Partial Replay': '#f47835',
                      'LwF': '#f781bf',
                      'O-EWC': '#83d0c9',
                      'SI': '#f781bf',
                      'ER': '#b15928',
                      'A-GEM': '#8b8589',
                      'TAG': '#f781bf',
                      'None': '#4c516d'}
for ii, name in enumerate(universal_clr_dict.keys()):
    print(name)
    register_palette(name, universal_clr_dict[name])
    
# %%
fig = plt.figure(constrained_layout=True,figsize=(46,32))
gs = fig.add_gridspec(32,46)

#c_top = sns.color_palette('Reds', n_colors=10)
c_top = []
for name in alg_name_top:
    c_top.extend(
        sns.color_palette(
            name, 
            n_colors=task_num
            )
        )
    
fontsize=40
ticksize=38
legendsize=16


ax = fig.add_subplot(gs[2:10,10:18])

ax_ = sns.stripplot(x='Algorithms', y='Forward Transfer Efficieny', hue='Task ID', data=df_fle, palette=c_top, ax=ax, size=25, legend=None)
ax_.set_xticklabels(
    alg_name_top,
    fontsize=fontsize,rotation=65,ha="right",rotation_mode='anchor'
    )
for xtick, color in zip(ax_.get_xticklabels(), alg_name_top):
        xtick.set_color(universal_clr_dict[color])

#ax.set_title('Resource Constrained FL', fontsize=fontsize)

ax_.set_yticks([-.3,0,.1])
ax.set_ylabel('Forward Transfer', fontsize=fontsize)
ax.set_xlabel('', fontsize=fontsize)
ax.tick_params(labelsize=ticksize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.hlines(0, 0,10, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')

handles_task, labels_task = ax.get_legend_handles_labels()

#########################################################
#ax[0][0].grid(axis='x')
ax = fig.add_subplot(gs[2:10,20:28])

ax_ = sns.stripplot(x='Algorithms', y='Backward Transfer Efficieny', hue='Task ID', data=df_ble, palette=c_top, ax=ax, size=25, legend=None)
ax_.set_xticklabels(
    alg_name_top,
    fontsize=fontsize,rotation=65,ha="right",rotation_mode='anchor'
    )
for xtick, color in zip(ax_.get_xticklabels(), alg_name_top):
        xtick.set_color(universal_clr_dict[color])

ax_.set_yticks([-.4,0,.1])
#ax.set_title('Resource Constrained BL', fontsize=fontsize)
ax.set_ylabel('Backward Transfer', fontsize=fontsize)
ax.hlines(0, 0,10, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')
ax.set_xlabel('')
ax.tick_params(labelsize=ticksize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
########################################################
ax = fig.add_subplot(gs[2:10,:8])

ax_ = sns.stripplot(x='Algorithms', y='Transfer Efficiency', hue='Task ID', data=df_le, palette=c_top, ax=ax, size=25, legend=None)
ax_.set_xticklabels(
    alg_name_top,
    fontsize=fontsize,rotation=65,ha="right",rotation_mode='anchor'
    )
for xtick, color in zip(ax_.get_xticklabels(), alg_name_top):
        xtick.set_color(universal_clr_dict[color])

ax_.set_yticks([0,.5])
#ax.set_title('Resource Constrained BL', fontsize=fontsize)
ax.set_ylabel('Transfer', fontsize=fontsize)
ax.hlines(0, 0,10, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')
ax.set_xlabel('')
ax.tick_params(labelsize=ticksize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)


##########################################################
ax = fig.add_subplot(gs[2:10,30:38])

ax_ = sns.stripplot(x='Algorithms', y='Accuracy', hue='Task ID', data=df_acc, palette=c_top, ax=ax, size=25, legend=None)
ax_.set_xticklabels(
    alg_name_top,
    fontsize=fontsize,rotation=65,ha="right",rotation_mode='anchor'
    )
for xtick, color in zip(ax_.get_xticklabels(), alg_name_top):
        xtick.set_color(universal_clr_dict[color])

ax_.set_yticks([0,.5])
#ax.set_title('Resource Constrained BL', fontsize=fontsize)
ax.set_ylabel('Accuracy', fontsize=fontsize)
ax.set_xlabel('')
ax.tick_params(labelsize=ticksize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

'''mean_error, std_error = unpickle('../recruitment_exp/result/recruitment_exp_500.pickle')
ns = 10*np.array([10, 50, 100, 200, 350, 500])
clr = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
colors = sns.color_palette(clr, n_colors=len(clr))

#labels = ['recruiting', 'Uncertainty Forest', 'hybrid', '50 Random', 'BF', 'building']
labels = ['SynF (building)', 'RF (new)', 'recycling', 'hybrid']
algo = ['building', 'UF', 'recruiting', 'hybrid']
adjust = 0
for i,key in enumerate(algo):
    err = np.array(mean_error[key])
    ax.plot(ns, err, c=colors[i], label=labels[i], linewidth=3)


#ax.set_title('CIFAR Recruitment Experiment', fontsize=30)
ax.set_xscale('log')
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_ylabel('Generalization Error', fontsize=fontsize)
ax.set_xlabel('')
ax.tick_params(labelsize=ticksize)
#ax.set_ylim(0.325, 0.575)
ax.set_title("Resource Recycling (C)",fontsize=fontsize+15)
ax.set_xticks([])
ax.set_yticks([0.45, 0.55, 0.65,0.75])
#ax.set_ylim([0.43,0.62])
#ax.text(50, 1, "50", fontsize=ticksize)
ax.text(100, 0.424, "100", fontsize=ticksize-2)
ax.text(500, 0.424, "500", fontsize=ticksize-2)
ax.text(5000, 0.424, "5000", fontsize=ticksize-2)
ax.text(120, 0.39, "Number of Task 10 Samples", fontsize=fontsize-1)

ax.legend(loc='lower left',fontsize=legendsize+10, frameon=False)
#ax.set_title('Recruitment Experiment on Task 10', fontsize=fontsize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
'''

#########################################################

#fig.text(.35, 0.85, "CIFAR 10X10 (Controlled Replay)", fontsize=fontsize+10)
#fig.legend(handles_bottom, labels_bottom, bbox_to_anchor=(.9, .9), fontsize=legendsize+14, frameon=False)
#########################################################
c_top = []
for name in range(4):
    c_top.extend(
        sns.color_palette(
            'SiLLy-N', 
            n_colors=task_num
            )
        )
    
ax = fig.add_subplot(gs[13:21,10:18])
ax_ = sns.stripplot(x='Algorithms', y='Forward Transfer Efficieny', hue='Task ID', data=df_fle_replay, palette=c_top, ax=ax, size=30, legend=None)
ax_.set_xticklabels(
    combined_alg_name_replay,
    fontsize=fontsize,rotation=65,ha="right",rotation_mode='anchor'
    )
        
#ax.set_title('Forward Learning (FL)', fontsize=fontsize)
ax_.set_yticks([0,.1])
ax.set_ylabel('Forward Transfer', fontsize=fontsize)
ax.set_xlabel('', fontsize=fontsize)
ax.tick_params(labelsize=ticksize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.hlines(0, 0, 3, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')

handles_top, labels_top = ax.get_legend_handles_labels()


#####################################################
ax = fig.add_subplot(gs[13:21,20:28])
ax_ = sns.stripplot(x='Algorithms', y='Backward Transfer Efficieny', data=df_ble_replay, hue='Task ID', palette=c_top, ax=ax, size=30, legend=None)
ax_.set_xticklabels(
    combined_alg_name_replay,
    fontsize=fontsize,rotation=65,ha="right",rotation_mode='anchor'
    )
ax_.set_yticks([0,.1])
ax.set_ylabel('Backward Transfer', fontsize=fontsize)
ax.set_xlabel('', fontsize=fontsize)
ax.tick_params(labelsize=ticksize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.hlines(0, 0, 3, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')

############################

ax = fig.add_subplot(gs[13:21,:8])
ax_ = sns.stripplot(x='Algorithms', y='Learning Efficieny', data=df_500_replay, hue='Task ID', palette=c_top, ax=ax, size=30, legend=None)
ax_.set_xticklabels(
    combined_alg_name_replay,
    fontsize=fontsize,rotation=65,ha="right",rotation_mode='anchor'
    )

#ax.set_title('Overall Learning', fontsize=fontsize)
ax_.set_yticks([0,.2])
ax.set_ylabel('Transfer', fontsize=fontsize)
ax.set_xlabel('', fontsize=fontsize)
ax.tick_params(labelsize=ticksize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.hlines(0, 0, 3, colors='grey', linestyles='dashed',linewidth=1.5)

#########################################################
ax = fig.add_subplot(gs[13:21,30:38])
ax_ = sns.stripplot(x='Algorithms', y='Accuracy', data=df_acc_replay, hue='Task ID', palette=c_top, ax=ax, size=30, legend=None)
ax_.set_xticklabels(
    combined_alg_name_replay,
    fontsize=fontsize,rotation=65,ha="right",rotation_mode='anchor'
    )

#ax.set_title('Overall Learning', fontsize=fontsize)
ax_.set_yticks([0,.5])
ax.set_ylabel('Accuracy', fontsize=fontsize)
ax.set_xlabel('', fontsize=fontsize)
ax.tick_params(labelsize=ticksize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

fig.text(.38,.96, 'Constrained Resource', fontsize=fontsize+15)
fig.text(0.28, 0.56, 'Controlled Replay (Resource Growing)', fontsize=fontsize+15)
fig.text(.33, 1, "CIFAR 10X10 (Ablation Study)", fontsize=fontsize+20)

fig.text(.35, 0.25, "Fraction of old data replayed", fontsize=fontsize+10)
#fig.legend(handles_top, labels_top, bbox_to_anchor=(.9, .55), fontsize=legendsize+14, frameon=False)
#fig.legend(handles_task, labels_task, bbox_to_anchor=(.9, .45), fontsize=legendsize+14, frameon=False)

plt.savefig('result/figs/cifar_ablation.pdf', bbox_inches='tight')

# %%
