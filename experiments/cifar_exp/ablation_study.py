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
total_alg_bottom = 11
alg_name_bottom = ['SynF', 'EWC', 'Total Replay', 'Partial Replay', 'LwF','O-EWC','SI', 'ER', 'A-GEM', 'TAG', 'None']
model_file_bottom = ['uf10withrep', 'EWC', 'offline', 'exact', 'LwF', 'OEWC', 'si', 'er', 'agem', 'tag', 'None']
btes_bottom = [[] for i in range(total_alg_bottom)]
ftes_bottom = [[] for i in range(total_alg_bottom)]
tes_bottom = [[] for i in range(total_alg_bottom)]

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
##############################
#############################
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
te_500_replay = {'SynN (.4)':np.zeros(10,dtype=float), 'SynN (.6)':np.zeros(10,dtype=float), 
          'SynN (.8)':np.zeros(10,dtype=float),
          'SynN (1)':np.zeros(10,dtype=float), 'SynF (.4)':np.zeros(10,dtype=float), 
          'SynF (.6)':np.zeros(10,dtype=float),'SynF (.8)':np.zeros(10,dtype=float),
          'SynF (1)':np.zeros(10,dtype=float)}

for count,name in enumerate(te_500_replay.keys()):
    print(name, count)
    for i in range(10):
        if count <4:
            te_500_replay[name][i] = np.log(tes_top_replay[count][i][9-i])
        else:
            te_500_replay[name][i] = np.log(tes_bottom_replay[count-8][i][9-i])


df_500_replay = pd.DataFrame.from_dict(te_500_replay)
df_500_replay = pd.melt(df_500_replay,var_name='Algorithms', value_name='Learning Efficieny')


# %%
fig = plt.figure(constrained_layout=True,figsize=(46,32))
gs = fig.add_gridspec(32,46)

clr_replay = ["#984ea3", "#984ea3", "#984ea3", "#984ea3", "#4daf4a", "#4daf4a", "#4daf4a", "#4daf4a"]

clr_bottom = ["#e41a1c", "#b15928", "#b15928", "#984ea3", "#f781bf", "#f781bf", "#f781bf", "#b15928", "#b15928", "#b15928", "#b15928"]
c_bottom = sns.color_palette(clr_bottom, n_colors=len(clr_bottom))

marker_style_bottom = ['.', '.', '+', 'v', '.', 'o', '*', '.', '+', 'x', 'o']

fontsize=38
ticksize=34
legendsize=16

ax = fig.add_subplot(gs[4:12,:8])
ax.plot([0], [0], color=[1,1,1], label='Resource Constrained')

for i, fte in enumerate(ftes_bottom):
    fte[0] = 1
    if i == 0:
        ax.plot(np.arange(1,11), fte, color=c_bottom[i], marker=marker_style_bottom[i], markersize=12, label = alg_name_bottom[i], linewidth=3)
        continue

    if i == 1:
        ax.plot(np.arange(1,11), fte, color=c_bottom[i], marker=marker_style_bottom[i], markersize=12, label = alg_name_bottom[i], linewidth=3)
        continue
    
    ax.plot(np.arange(1,11), fte, color=c_bottom[i], marker=marker_style_bottom[i], markersize=12, label = alg_name_bottom[i])
    
ax.set_xticks(np.arange(1,11))
ax.set_yticks([0.85, 1, 1.1])
ax.set_ylim(0.8, 1.15)
ax.tick_params(labelsize=ticksize)
ax.set_title('Resource Constrained FL', fontsize=fontsize)

ax.set_ylabel('log FLE', fontsize=fontsize)
ax.set_xlabel('Tasks seen', fontsize=fontsize)

log_lbl = np.round(
    np.log([0.85,1,1.1]),
    1
)
labels = [item.get_text() for item in ax.get_yticklabels()]

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax.set_yticklabels(labels)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')

handles_bottom, labels_bottom = ax.get_legend_handles_labels()
'''for i in range(0,total_alg_top+total_alg_bottom-1):
    ax.plot(1,0,color=c_combined[i], marker=marker_style[i], markersize=8,label=combined_alg_name[i])'''

#########################################################
#ax[0][0].grid(axis='x')
ax = fig.add_subplot(gs[2:15,8:27], projection='3d')
#cmap = sns.color_palette("coolwarm", as_cmap=True)
color = ['b', 'r']
for i in range(task_num - 1):

    et = np.zeros((total_alg_bottom,task_num-i))

    for j in range(0,total_alg_bottom):
        et[j,:] = np.asarray(btes_bottom[j][i])

    ns = np.arange(i + 1, task_num + 1)
    ns_new = np.linspace(ns.min(), ns.max(), 300)

    for j in range(0,total_alg_bottom):
        y_interp = np.interp(ns_new, ns, et[j,:])
        idx = np.where(y_interp < 1.0)[0]
        supper = y_interp.copy()
        supper[idx] = np.nan

        idx = np.where(y_interp >= 1.0)[0]
        slower = y_interp.copy()
        slower[idx] = np.nan

        ax.plot(ns_new, supper, zs=j, zdir='y', color='r', linewidth=3)
        ax.plot(ns_new, slower, zs=j, zdir='y', color='b', linewidth=3)
        

xs = np.linspace(0, 10, 10)
zs = np.linspace(0, total_alg_bottom-1, 10)
X, Y = np.meshgrid(xs, zs)
Z = np.ones(X.shape)

ax.plot_surface(X, Y, Z, color='grey', alpha=.3)

for ii in range(total_alg_bottom):
    zs = np.linspace(ii-.05,ii+.05,10)
    X, Y = np.meshgrid(xs, zs)
    Z = np.ones(X.shape)

    ax.plot_surface(X, Y, Z, color='grey', alpha=1)

ax.view_init(elev=10., azim=15, roll=0)

'''for i in range(total_alg_top,total_alg_top+total_alg_bottom-1):
    ax.plot(1,0,color=c_combined[i], marker=marker_style[i], markersize=8,label=combined_alg_name[i])'''

ax.set_xlabel('Tasks seen', fontsize=30, labelpad=15)
ax.set_zlabel('log BLE', fontsize=30, labelpad=15)

ax.set_zticks([.8,.9,1, 1.1,1.2])
ax.set_xticks(np.arange(2,11,4))
ax.set_yticks(list(range(total_alg_bottom)))
ax.set_zlim(0.76, 1.25)
ax.set_ylim([0,total_alg_bottom-1])
log_lbl = np.round(
    np.log([.8,.9,1,1.1,1.2]),
    1
)
labels = [item.get_text() for item in ax.get_zticklabels()]

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax.text(.9, .5, 1.4, 'Resource Constrained BL', fontsize=fontsize)
ax.set_zticklabels(labels)
ax.set_yticklabels(alg_name_bottom, rotation=80)
ax.tick_params(labelsize=ticksize-8)
#ax[0][1].grid(axis='x')
ax.invert_xaxis()

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
#ax.hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')

for ytick, color in zip(ax.get_yticklabels(), clr_bottom):
    ytick.set_color(color)


##########################################################
ax = fig.add_subplot(gs[4:12,28:36])

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
    #ax.fill_between(ns, 
    #        acc + 1.96*np.array(std_error[key]), 
    #        acc - 1.96*np.array(std_error[key]), 
    #        where=acc + 1.96*np.array(std_error[key]) >= acc - 1.96*np.array(std_error[key]), 
    #        facecolor=colors[i], 
    #        alpha=0.15,
    #        interpolate=False)


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
fig.legend(handles_bottom, labels_bottom, bbox_to_anchor=(.9, .8), fontsize=legendsize+14, frameon=False)
#########################################################
ax = fig.add_subplot(gs[19:27,:8])
ax.plot([0], [0], color=[1,1,1], label='Controlled Replay')
fte = ftes_top_replay[3]
fte[0] = 1
ax.plot(np.arange(1,11), fte, color=clr_replay[0], label='SynN', linewidth=3)

fte = ftes_bottom_replay[3]
fte[0] = 1
ax.plot(np.arange(1,11), fte, color=clr_replay[4], label='SynF', linewidth=3)

ax.set_title('Forward Learning (FL)', fontsize=fontsize)
ax.set_xticks(np.arange(1,11))
ax.set_yticks([0.9, 1, 1.1, 1.2, 1.3])
#ax.set_yticks([])
#ax.text(0, np.mean(ax.get_ylim()), "%s" % str(0), fontsize=26)
#ax.yaxis.set_major_locator(plt.LogLocator(subs=(0.9, 1, 1.1, 1.2, 1.3)))
ax.set_ylim(0.9, 1.31)

log_lbl = np.round(
    np.log([0.9,1,1.1,1.2,1.3]),
    1
)
labels = [item.get_text() for item in ax.get_yticklabels()]

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax.set_yticklabels(labels)

ax.tick_params(labelsize=ticksize)

ax.set_ylabel('log FLE', fontsize=fontsize)
ax.set_xlabel('Tasks seen', fontsize=fontsize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')

handles_top, labels_top = ax.get_legend_handles_labels()


#####################################################
ax = fig.add_subplot(gs[17:30,8:27], projection='3d')

color = ['b', 'r']
for i in range(task_num - 1):

    et = np.zeros((total_alg_top_replay,task_num-i))

    for j in range(0,total_alg_top_replay):
        et[j,:] = np.asarray(btes_top_replay[j][i])

    ns = np.arange(i + 1, task_num + 1)
    ns_new = np.linspace(ns.min(), ns.max(), 60)

    for j in range(0,total_alg_top_replay):
        y_interp = np.interp(ns_new, ns, et[j,:])
        idx = np.where(y_interp < 1.0)[0]
        supper = y_interp.copy()
        supper[idx] = np.nan

        idx = np.where(y_interp >= 1.0)[0]
        slower = y_interp.copy()
        slower[idx] = np.nan

        ax.plot(ns_new, supper, zs=j, zdir='y', color='r', linewidth=3)
        ax.plot(ns_new, slower, zs=j, zdir='y', color='b', linewidth=3)
        



for i in range(task_num - 1):

    et = np.zeros((total_alg_bottom_replay,task_num-i))

    for j in range(0,total_alg_bottom_replay):
        et[j,:] = np.asarray(btes_bottom_replay[j][i])

    ns = np.arange(i + 1, task_num + 1)
    ns_new = np.linspace(ns.min(), ns.max(), 60)

    for j in range(0,total_alg_bottom_replay):
        y_interp = np.interp(ns_new, ns, et[j,:])
        idx = np.where(y_interp < 1.0)[0]
        supper = y_interp.copy()
        supper[idx] = np.nan

        idx = np.where(y_interp >= 1.0)[0]
        slower = y_interp.copy()
        slower[idx] = np.nan

        ax.plot(ns_new, supper, zs=j+4, zdir='y', color='r', linewidth=3)
        ax.plot(ns_new, slower, zs=j+4, zdir='y', color='b', linewidth=3)
        

xs = np.linspace(0, 10, 10)
zs = np.linspace(0, 7, 10)
X, Y = np.meshgrid(xs, zs)
Z = np.ones(X.shape)

ax.plot_surface(X, Y, Z, color='grey', alpha=.3)

for ii in range(total_alg_top_replay):
    zs = np.linspace(ii-.05,ii+.05,10)
    X, Y = np.meshgrid(xs, zs)
    Z = np.ones(X.shape)

    ax.plot_surface(X, Y, Z, color='grey', alpha=1)

for ii in range(4,4+total_alg_bottom_replay):
    zs = np.linspace(ii-.05,ii+.05,10)
    X, Y = np.meshgrid(xs, zs)
    Z = np.ones(X.shape)

    ax.plot_surface(X, Y, Z, color='grey', alpha=1)


ax.view_init(elev=10., azim=15, roll=0)

'''for i in range(total_alg_top,total_alg_top+total_alg_bottom-1):
    ax.plot(1,0,color=c_combined[i], marker=marker_style[i], markersize=8,label=combined_alg_name[i])'''

ax.text(.9, .5, 1.35, 'Backward Learning (BL)', fontsize=fontsize)
ax.set_xlabel('Tasks seen', fontsize=30, labelpad=15)
ax.set_zlabel('log BLE', fontsize=30, labelpad=15)

ax.set_zticks([.9,1, 1.1,1.2])
ax.set_xticks(np.arange(2,11,4))
ax.set_zlim(0.9, 1.25)
ax.set_ylim([0,7])
log_lbl = np.round(
    np.log([.9,1,1.1,1.2]),
    1
)
labels = [item.get_text() for item in ax.get_zticklabels()]

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax.set_zticklabels(labels)
ax.set_yticks(np.arange(0,8,1))
ax.set_yticklabels(combined_alg_name_replay, rotation=80)
ax.tick_params(labelsize=ticksize-8)
#ax[0][1].grid(axis='x')
ax.invert_xaxis()

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
#ax.hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')

for ytick, color in zip(ax.get_yticklabels(), clr_replay):
    ytick.set_color(color)


############################

ax = fig.add_subplot(gs[19:27,28:36])

ax.set_title('Overall Learning', fontsize=fontsize)
ax.tick_params(labelsize=22)
ax_ = sns.boxplot(
    x="Algorithms", y="Learning Efficieny", data=df_500_replay, palette=clr_replay, whis=np.inf,
    ax=ax, showfliers=False, notch=1
    )
ax.hlines(0, -1,8, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')
#sns.boxplot(x="Algorithms", y="Transfer Efficieny", data=mean_df, palette=c, linewidth=3, ax=ax[1][1])
#ax_=sns.pointplot(x="Algorithms", y="Transfer Efficieny", data=df_500, join=False, color='grey', linewidth=1.5, ci='sd',ax=ax)
ax_.set_yticks([0,.1,.25])
ax_.set_xlabel('', fontsize=fontsize)
ax.set_ylabel('log LE after 10 Tasks', fontsize=fontsize-5)
ax_.set_xticklabels(
    ['SynN (0.4)','SynN (0.6)', 'SynN (0.8)', 'SynN (1.0)','SynF (0.4)', 'SynF (0.6)', 'SynF (0.8)', 'SynF (1.0)'],
    fontsize=22,rotation=80,ha="right",rotation_mode='anchor'
    )

stratified_scatter(te_500_replay,ax,16,clr_replay)
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

#########################################################

fig.text(.35, 0.85, "CIFAR 10X10 (Ablation Study)", fontsize=fontsize+10)
fig.legend(handles_top, labels_top, bbox_to_anchor=(.9, .45), fontsize=legendsize+14, frameon=False)

plt.savefig('result/figs/cifar_ablation.pdf', dpi=300)

# %%
