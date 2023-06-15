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
total_alg_top = 6
total_alg_bottom = 8
alg_name_top = ['SynN','SynF', 'Model Zoo','ProgNN', 'LMC', 'DF-CNN']
alg_name_bottom = ['SynF','LwF','O-EWC','SI', 'ER', 'A-GEM', 'TAG', 'None']
combined_alg_name = ['SynN','SynF', 'Model Zoo','ProgNN', 'DF-CNN','EWC', 'Total Replay', 'Partial Replay', 'LwF', 'O-EWC','SI', 'ER', 'A-GEM', 'TAG', 'None']
model_file_top = ['dnn0withrep','fixed_uf10withrep', 'model_zoo','Prog_NN', 'LMC', 'DF_CNN']
model_file_bottom = ['uf10withrep', 'LwF', 'OEWC', 'si', 'er', 'agem', 'tag', 'None']
btes_top = [[] for i in range(total_alg_top)]
ftes_top = [[] for i in range(total_alg_top)]
tes_top = [[] for i in range(total_alg_top)]
btes_bottom = [[] for i in range(total_alg_bottom)]
ftes_bottom = [[] for i in range(total_alg_bottom)]
tes_bottom = [[] for i in range(total_alg_bottom)]
avg_acc_top = [[] for i in range(total_alg_top)]
avg_var_top = [[] for i in range(total_alg_top)]
avg_acc_bottom = [[] for i in range(total_alg_bottom)]
avg_var_bottom = [[] for i in range(total_alg_bottom)]

avg_single_acc_top = [[] for i in range(total_alg_top)]
avg_single_var_top = [[] for i in range(total_alg_top)]
avg_single_acc_bottom = [[] for i in range(total_alg_bottom)]
avg_single_var_bottom = [[] for i in range(total_alg_bottom)]

#combined_alg_name = ['L2N','L2F','Prog-NN', 'DF-CNN','LwF','EWC','O-EWC','SI', 'Replay (increasing amount)', 'Replay (fixed amount)', 'None']
model_file_combined = ['dnn0withrep','fixed_uf10withrep', 'model_zoo','Prog_NN','DF_CNN', 'LwF', 'EWC', 'OEWC', 'si', 'offline', 'exact', 'er', 'agem', 'tag', 'None']

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
                filename = './result/result/'+model_file_top[alg]+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
            elif alg == 3 or alg == 5:
                filename = './benchmarking_algorthms_result/'+model_file_top[alg]+'-'+str(shift+1)+'-'+str(slot+1)+'.pickle'
            elif alg == 2:
                filename = './benchmarking_algorthms_result/'+model_file_top[alg]+'_'+str(slot+1)+'_'+str(shift+1)+'.pickle'
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
    avg_acc, avg_var = calc_avg_acc(err, reps)
    avg_single_acc, avg_single_var = calc_avg_single_acc(single_err, reps)

    btes_top[alg].extend(bte)
    ftes_top[alg].extend(fte)
    tes_top[alg].extend(te)
    avg_acc_top[alg]= avg_acc
    avg_var_top[alg] = avg_var
    avg_single_acc_top[alg]= avg_single_acc
    avg_single_var_top[alg] = avg_single_var

    print('Algo name:' , alg_name_top[alg])
    print('Accuracy', np.round(calc_acc(err,reps),2))
    print('forget', np.round(calc_forget(err, reps),2))
    print('transfer', np.round(calc_transfer(err, single_err, reps),2))
    

# %%
fig = plt.figure(constrained_layout=True,figsize=(18,8))
gs = fig.add_gridspec(8,18)

clr_top = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#984ea3", "#f781bf", "#b15928", "#b15928", "#984ea3"]
c_top = sns.color_palette(clr_top, n_colors=len(clr_top))

marker_style_top = ['.', '.', '.', '.', 'o', '+']

fontsize=30
ticksize=26
legendsize=16

ax = fig.add_subplot(gs[1:6,:6])

for i, fte in enumerate(ftes_top):
    fte[0] = 1
    if i == 0:
        ax.plot(np.arange(1,11), fte, color=c_top[i], marker=marker_style_top[i], markersize=12, label=alg_name_top[i], linewidth=3)
        continue

    if i == 1:
        ax.plot(np.arange(1,11), fte, color=c_top[i], marker=marker_style_top[i], markersize=12, label=alg_name_top[i], linewidth=3)
        continue
    
    ax.plot(np.arange(1,11), fte, color=c_top[i], marker=marker_style_top[i], markersize=12, label=alg_name_top[i])

ax.set_title('Forward Learning (FL)', fontsize=fontsize)
ax.set_xticks(np.arange(1,11))
ax.set_yticks([0.8, 1, 1.3])
#ax.set_yticks([])
#ax.text(0, np.mean(ax.get_ylim()), "%s" % str(0), fontsize=26)
#ax.yaxis.set_major_locator(plt.LogLocator(subs=(0.9, 1, 1.1, 1.2, 1.3)))
ax.set_ylim(0.8, 1.31)

log_lbl = np.round(
    np.log([0.8,1,1.3]),
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


#########################################################
#ax[0][0].grid(axis='x')
ax = fig.add_subplot(gs[:7,6:16], projection='3d')
#cmap = sns.color_palette("coolwarm", as_cmap=True)
color = ['b', 'r']
for i in range(task_num - 1):

    et = np.zeros((total_alg_top,task_num-i))

    for j in range(0,total_alg_top):
        et[j,:] = np.asarray(btes_top[j][i])

    ns = np.arange(i + 1, task_num + 1)
    ns_new = np.linspace(ns.min(), ns.max(), 60)

    for j in range(0,total_alg_top):
        y_interp = np.interp(ns_new, ns, et[j,:])
        idx = np.where(y_interp < 1.0)[0]
        supper = y_interp.copy()
        supper[idx] = np.nan

        idx = np.where(y_interp >= 1.0)[0]
        slower = y_interp.copy()
        slower[idx] = np.nan

        ax.plot(ns_new, supper, zs=j, zdir='y', color='r', linewidth=3)
        ax.plot(ns_new, slower, zs=j, zdir='y', color='b', linewidth=3)
        

xs = np.linspace(0, 11, 10)
zs = np.linspace(0, 5, 10)
X, Y = np.meshgrid(xs, zs)
Z = np.ones(X.shape)

ax.plot_surface(X, Y, Z, color='grey', alpha=.3)

for ii in range(total_alg_top):
    zs = np.linspace(ii-.05,ii+.05,10)
    X, Y = np.meshgrid(xs, zs)
    Z = np.ones(X.shape)

    ax.plot_surface(X, Y, Z, color='grey', alpha=1)


ax.view_init(elev=10., azim=15, roll=0)

'''for i in range(total_alg_top,total_alg_top+total_alg_bottom-1):
    ax.plot(1,0,color=c_combined[i], marker=marker_style[i], markersize=8,label=combined_alg_name[i])'''

ax.text(.6, .6, 1.4, 'Backward Learning (BL)', fontsize=fontsize)
ax.set_xlabel('Tasks seen', fontsize=30, labelpad=15)
ax.set_zlabel('log BLE', fontsize=30, labelpad=15)

ax.set_zticks([.8,1,1.2])
ax.set_yticks([0,1,2,3,4,5])
ax.set_xticks(np.arange(2,11,4))
ax.set_zlim(0.76, 1.25)
ax.set_ylim([0,5])
log_lbl = np.round(
    np.log([.8,1,1.2]),
    1
)
labels = [item.get_text() for item in ax.get_zticklabels()]

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax.set_zticklabels(labels)
ax.set_yticklabels(alg_name_top, rotation=80)
ax.tick_params(labelsize=ticksize-8)
#ax[0][1].grid(axis='x')
ax.invert_xaxis()

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
#ax.hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')

for ytick, color in zip(ax.get_yticklabels(), clr_top):
    ytick.set_color(color)


##########################################################
fig.text(.25, .9, "CIFAR 10X10 (500 samples)", fontsize=fontsize+5)
fig.legend(handles_top, labels_top, bbox_to_anchor=(1, .8), fontsize=legendsize+8, frameon=False)

plt.savefig('result/figs/cifar_exp_500.pdf', dpi=300)
# %%
