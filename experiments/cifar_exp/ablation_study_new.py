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
alg_name_top = ['SynF', 'EWC', 'Total Replay', 'Partial Replay', 'LwF','O-EWC','SI', 'ER', 'A-GEM', 'TAG', 'None']
model_file_top = ['uf10withrep', 'EWC', 'offline', 'exact', 'LwF', 'OEWC', 'si', 'er', 'agem', 'tag', 'None']
btes_top = [[] for i in range(total_alg_top)]
ftes_top = [[] for i in range(total_alg_top)]
tes_top = [[] for i in range(total_alg_top)]

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
                filename = './result/result/'+model_file_top[alg]+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
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

    btes_top[alg].extend(bte)
    ftes_top[alg].extend(fte)
    tes_top[alg].extend(te)

#%%
fte_top_end = {'SynF':np.zeros(10,dtype=float), 'EWC':np.zeros(10,dtype=float),
               'Total Replay':np.zeros(10,dtype=float),
               'Partial Replay':np.zeros(10,dtype=float), 'LwF':np.zeros(10,dtype=float),
               'O-EWC':np.zeros(10,dtype=float), 'SI':np.zeros(10,dtype=float),
               'ER':np.zeros(10,dtype=float), 'A-GEM':np.zeros(10,dtype=float),
               'TAG':np.zeros(10,dtype=float), 'None':np.zeros(10,dtype=float)}


for count,name in enumerate(fte_top_end.keys()):
    #print(name, count)
    for i in range(10):
        fte_top_end[name][i] = np.log(ftes_top[count][i])

tmp_fle = {}
for id in fte_top_end.keys():
    tmp_fle[id] = fte_top_end[id]

df_fle = pd.DataFrame.from_dict(tmp_fle)
df_fle = pd.melt(df_fle,var_name='Algorithms', value_name='Forward Transfer Efficieny')
#%%
bte_end = {'SynF':np.zeros(10,dtype=float), 'EWC':np.zeros(10,dtype=float),
           'Total Replay':np.zeros(10,dtype=float),
           'Partial Replay':np.zeros(10,dtype=float), 'LwF':np.zeros(10,dtype=float),
           'O-EWC':np.zeros(10,dtype=float), 'SI':np.zeros(10,dtype=float),
           'ER':np.zeros(10,dtype=float), 'A-GEM':np.zeros(10,dtype=float),
           'TAG':np.zeros(10,dtype=float), 'None':np.zeros(10,dtype=float)}


for count,name in enumerate(bte_end.keys()):
    #print(name, count)
    for i in range(10):
        bte_end[name][i] = np.log(btes_top[count][i][9-i])

tmp_ble = {}
for id in alg_name_top:
    tmp_ble[id] = bte_end[id]

df_ble = pd.DataFrame.from_dict(tmp_ble)
df_ble = pd.melt(df_ble,var_name='Algorithms', value_name='Backward Transfer Efficieny')

#%%
### MAIN HYPERPARAMS ###
ntrees = 10
slots = 10
task_num = 10
shifts = 6
total_alg_top_replay = 4
total_alg_bottom_replay = 4
alg_name_top_replay = ['SynN (0.4)', 'SynN (0.6)', 'SynN (0.8)', 'SynN (1.0)']
alg_name_bottom_replay = ['SynF (0.4)', 'SynF (0.6)', 'SynF (0.8)', 'SynF (1.0)']
combined_alg_name_replay = ['SynN (0.4)', 'SynN (0.6)', 'SynN (0.8)', 'SynN (1.0)', 'SynF (0.4)', 'SynF (0.6)', 'SynF (0.8)', 'SynF (1.0)']
model_file_top_replay = ['dnn0']
model_file_bottom_replay = ['uf10']
samples_to_replay = [.4,.6,.8,1]

btes_top_replay = [[] for i in range(total_alg_top_replay)]
ftes_top_replay = [[] for i in range(total_alg_top_replay)]
tes_top_replay = [[] for i in range(total_alg_top_replay)]
btes_bottom_replay = [[] for i in range(total_alg_bottom_replay)]
ftes_bottom_replay = [[] for i in range(total_alg_bottom_replay)]
tes_bottom_replay = [[] for i in range(total_alg_bottom_replay)]

model_file_combined_replay = ['dnn0','uf10']

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
    avg_acc, avg_var = calc_avg_acc(err, reps)
    avg_single_acc, avg_single_var = calc_avg_single_acc(single_err, reps)

    btes_top_replay[alg].extend(bte)
    ftes_top_replay[alg].extend(fte)
    tes_top_replay[alg].extend(te)
    

# %%
reps = slots*shifts

for alg, samples in enumerate(samples_to_replay):
    count = 0 
    bte_tmp = [[] for _ in range(reps)]
    fte_tmp = [[] for _ in range(reps)] 
    te_tmp = [[] for _ in range(reps)]

    for slot in range(slots):
        for shift in range(shifts):
            filename = './controlled_replay_result/'+model_file_bottom_replay[0]+'_'+str(shift+1)+'_'+str(slot)+'_'+str(samples)+'.pickle'
            
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

    btes_bottom_replay[alg].extend(bte)
    ftes_bottom_replay[alg].extend(fte)
    tes_bottom_replay[alg].extend(te)
#%%
fte_replay_end = {'SynN (.4)':np.zeros(10,dtype=float), 'SynN (.6)':np.zeros(10,dtype=float), 
          'SynN (.8)':np.zeros(10,dtype=float),
          'SynN (1)':np.zeros(10,dtype=float), 'SynF (.4)':np.zeros(10,dtype=float), 
          'SynF (.6)':np.zeros(10,dtype=float),'SynF (.8)':np.zeros(10,dtype=float),
          'SynF (1)':np.zeros(10,dtype=float)}

for count,name in enumerate(fte_replay_end.keys()):
    print(name, count)
    if count < 4:
        for i in range(10):
            fte_replay_end[name][i] = np.log(ftes_top_replay[0][i])
    else:
        for i in range(10):
            fte_replay_end[name][i] = np.log(ftes_bottom_replay[0][i])

tmp_fle = {}
for id in fte_replay_end.keys():
    tmp_fle[id] = fte_replay_end[id]

df_fle_replay = pd.DataFrame.from_dict(tmp_fle)
df_fle_replay = pd.melt(df_fle_replay,var_name='Algorithms', value_name='Forward Transfer Efficieny')

#%%
bte_end_replay = {'SynN (.4)':np.zeros(10,dtype=float), 'SynN (.6)':np.zeros(10,dtype=float), 
          'SynN (.8)':np.zeros(10,dtype=float),
          'SynN (1)':np.zeros(10,dtype=float), 'SynF (.4)':np.zeros(10,dtype=float), 
          'SynF (.6)':np.zeros(10,dtype=float),'SynF (.8)':np.zeros(10,dtype=float),
          'SynF (1)':np.zeros(10,dtype=float)}


for count,name in enumerate(bte_end_replay.keys()):
    #print(name, count)
    for i in range(10):
        if count <4:
            bte_end_replay[name][i] = np.log(tes_top_replay[count][i][9-i])
        else:
            bte_end_replay[name][i] = np.log(tes_bottom_replay[count-4][i][9-i])

tmp_ble = {}
for id in bte_end_replay.keys():
    tmp_ble[id] = bte_end_replay[id]

df_ble_replay = pd.DataFrame.from_dict(tmp_ble)
df_ble_replay = pd.melt(df_ble_replay,var_name='Algorithms', value_name='Backward Transfer Efficieny')

#%%
te_500_replay = {'SynN (.4)':np.zeros(10,dtype=float), 'SynN (.6)':np.zeros(10,dtype=float), 
          'SynN (.8)':np.zeros(10,dtype=float),
          'SynN (1)':np.zeros(10,dtype=float), 'SynF (.4)':np.zeros(10,dtype=float), 
          'SynF (.6)':np.zeros(10,dtype=float),'SynF (.8)':np.zeros(10,dtype=float),
          'SynF (1)':np.zeros(10,dtype=float)}

for count,name in enumerate(te_500_replay.keys()):
    #print(name, count)
    for i in range(10):
        if count <4:
            te_500_replay[name][i] = np.log(tes_top_replay[count][i][9-i])
        else:
            te_500_replay[name][i] = np.log(tes_bottom_replay[count-4][i][9-i])


df_500_replay = pd.DataFrame.from_dict(te_500_replay)
df_500_replay = pd.melt(df_500_replay,var_name='Algorithms', value_name='Learning Efficieny')


# %%
fig = plt.figure(constrained_layout=True,figsize=(46,32))
gs = fig.add_gridspec(32,46)

clr_replay = ["#984ea3", "#984ea3", "#984ea3", "#984ea3", "#4daf4a", "#4daf4a", "#4daf4a", "#4daf4a"]

clr_top = ["#e41a1c", "#b15928", "#b15928", "#984ea3", "#f781bf", "#f781bf", "#f781bf", "#b15928", "#b15928", "#b15928", "#b15928"]
c_top = sns.color_palette(clr_top, n_colors=len(clr_top))

marker_style = ['.', '.', '+', 'v', '.', 'o', '*', '.', '+', 'x', 'o']

fontsize=40
ticksize=34
legendsize=16

ax = fig.add_subplot(gs[4:12,:8])
ax.plot([0], [0], color=[1,1,1], label='Resource Constrained')
ax_ = sns.stripplot(x='Algorithms', y='Forward Transfer Efficieny', data=df_fle, hue='Algorithms', palette=c_top, ax=ax, size=20, legend=None, alpha=.3)
ax_.set_xticklabels(
    alg_name_top,
    fontsize=fontsize,rotation=65,ha="right",rotation_mode='anchor'
    )
for xtick, color in zip(ax_.get_xticklabels(), c_top):
        xtick.set_color(color)

ax.set_title('Resource Constrained FL', fontsize=fontsize)

ax_.set_yticks([-.3,0,.1])
ax.set_ylabel('Forward Transfer', fontsize=fontsize)
ax.set_xlabel('', fontsize=fontsize)
ax.tick_params(labelsize=ticksize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

for i in range(0,total_alg_top):
    ax.plot(1,0,color=c_top[i], markersize=8,label=alg_name_top[i])
ax.hlines(0, 0,10, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')

handles_bottom, labels_bottom = ax.get_legend_handles_labels()

#########################################################
#ax[0][0].grid(axis='x')
ax = fig.add_subplot(gs[4:12,12:20])
ax.plot([0], [0], color=[1,1,1], label='Resource Constrained')
ax_ = sns.stripplot(x='Algorithms', y='Backward Transfer Efficieny', data=df_ble, hue='Algorithms', palette=c_top, ax=ax, size=20, legend=None, alpha=.3)
ax_.set_xticklabels(
    alg_name_top,
    fontsize=fontsize,rotation=65,ha="right",rotation_mode='anchor'
    )
for xtick, color in zip(ax_.get_xticklabels(), c_top):
        xtick.set_color(color)

ax_.set_yticks([-.4,0,.1])
ax.set_title('Resource Constrained BL', fontsize=fontsize)
ax.set_ylabel('Backward Transfer', fontsize=fontsize)
ax.hlines(0, 0,10, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')
ax.set_xlabel('')
ax.tick_params(labelsize=ticksize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

##########################################################
ax = fig.add_subplot(gs[4:12,24:32])

mean_error, std_error = unpickle('../recruitment_exp/result/recruitment_exp_500.pickle')
ns = 10*np.array([10, 50, 100, 200, 350, 500])
clr = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
colors = sns.color_palette(clr, n_colors=len(clr))

#labels = ['recruiting', 'Uncertainty Forest', 'hybrid', '50 Random', 'BF', 'building']
labels = ['SynF (building)', 'UF (new)', 'recycling', 'hybrid']
algo = ['building', 'UF', 'recruiting', 'hybrid']
adjust = 0
for i,key in enumerate(algo):
    err = np.array(mean_error[key])
    ax.plot(ns, err, c=colors[i], label=labels[i])


#ax.set_title('CIFAR Recruitment Experiment', fontsize=30)
ax.set_xscale('log')
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_ylabel('Generalization Error', fontsize=fontsize)
ax.set_xlabel('')
ax.tick_params(labelsize=ticksize)
#ax.set_ylim(0.325, 0.575)
ax.set_title("Resource Recycling",fontsize=fontsize)
ax.set_xticks([])
ax.set_yticks([0.45, 0.55, 0.65,0.75])
#ax.set_ylim([0.43,0.62])
#ax.text(50, 1, "50", fontsize=ticksize)
ax.text(100, 0.424, "100", fontsize=ticksize-2)
ax.text(500, 0.424, "500", fontsize=ticksize-2)
ax.text(5000, 0.424, "5000", fontsize=ticksize-2)
ax.text(120, 0.39, "Number of Task 10 Samples", fontsize=fontsize-1)

ax.legend(loc='lower left',fontsize=legendsize+6, frameon=False)
#ax.set_title('Recruitment Experiment on Task 10', fontsize=fontsize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)


#########################################################

#fig.text(.35, 0.85, "CIFAR 10X10 (Controlled Replay)", fontsize=fontsize+10)
fig.legend(handles_bottom, labels_bottom, bbox_to_anchor=(.9, .9), fontsize=legendsize+14, frameon=False)
#########################################################
color = ['b','b','b','b','r','r','r','r']

ax = fig.add_subplot(gs[13:21,:9])
ax.plot([0], [0], color=[1,1,1], label='Controlled Replay')
ax_ = sns.stripplot(x='Algorithms', y='Forward Transfer Efficieny', data=df_fle_replay, hue='Algorithms', palette=color, ax=ax, size=20, legend=None, alpha=.3)
ax_.set_xticklabels(
    combined_alg_name_replay,
    fontsize=fontsize,rotation=65,ha="right",rotation_mode='anchor'
    )
for xtick, c in zip(ax_.get_xticklabels(), color):
        xtick.set_color(c)

ax.set_title('Forward Learning (FL)', fontsize=fontsize)
ax_.set_yticks([0,.1])
ax.set_ylabel('Forward Transfer', fontsize=fontsize)
ax.set_xlabel('', fontsize=fontsize)
ax.tick_params(labelsize=ticksize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.plot(1,0,color=color[0], markersize=8,label='SynN')
ax.plot(1,0,color=color[5], markersize=8,label='SynF')

ax.hlines(0, 1,8, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')

handles_top, labels_top = ax.get_legend_handles_labels()


#####################################################
ax = fig.add_subplot(gs[13:21,12:20])
ax_ = sns.stripplot(x='Algorithms', y='Backward Transfer Efficieny', data=df_ble_replay, hue='Algorithms', palette=color, ax=ax, size=20, legend=None, alpha=.3)
ax_.set_xticklabels(
    combined_alg_name_replay,
    fontsize=fontsize,rotation=65,ha="right",rotation_mode='anchor'
    )
for xtick, c in zip(ax_.get_xticklabels(), color):
        xtick.set_color(c)

ax.set_title('Backward Learning (BL)', fontsize=fontsize)
ax_.set_yticks([0,.2])
ax.set_ylabel('Backward Transfer', fontsize=fontsize)
ax.set_xlabel('', fontsize=fontsize)
ax.tick_params(labelsize=ticksize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

for i in range(0,total_alg_top):
    ax.plot(1,0,color=c_top[i], markersize=8,label=alg_name_top[i])
ax.hlines(0, 0,8, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')

############################

ax = fig.add_subplot(gs[13:21,24:32])
ax_ = sns.stripplot(x='Algorithms', y='Learning Efficieny', data=df_500_replay, hue='Algorithms', palette=color, ax=ax, size=20, legend=None, alpha=.3)
ax_.set_xticklabels(
    combined_alg_name_replay,
    fontsize=fontsize,rotation=65,ha="right",rotation_mode='anchor'
    )
for xtick, c in zip(ax_.get_xticklabels(), color):
        xtick.set_color(c)

ax.set_title('Overall Learning', fontsize=fontsize)
ax_.set_yticks([0,.2])
ax.set_ylabel('Oveall transfer after 10 Tasks', fontsize=fontsize)
ax.set_xlabel('', fontsize=fontsize)
ax.tick_params(labelsize=ticksize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

for i in range(0,total_alg_top):
    ax.plot(1,0,color=c_top[i], markersize=8,label=alg_name_top[i])
ax.hlines(0, 0,8, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')

#########################################################

fig.text(.35, .93, "CIFAR 10X10 (Ablation Study)", fontsize=fontsize+20)
fig.legend(handles_top, labels_top, bbox_to_anchor=(.9, .55), fontsize=legendsize+14, frameon=False)

plt.savefig('result/figs/cifar_ablation.pdf', dpi=300)

# %%
