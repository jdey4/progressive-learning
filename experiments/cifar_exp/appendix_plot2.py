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
alg_name_top = ['Odin','Odif','ProgNN', 'DF-CNN']
alg_name_bottom = ['Odif','LwF','EWC','O-EWC','SI', 'Full replay', 'Partial replay', 'None']
combined_alg_name = ['Odin','Odif','ProgNN', 'DF-CNN','LwF','EWC','O-EWC','SI', 'Total Replay', 'Partial Replay', 'None']
model_file_top = ['dnn0','fixed_uf10','Prog_NN','DF_CNN']
model_file_bottom = ['uf10', 'LwF', 'EWC', 'OEWC', 'si', 'offline', 'exact', 'None']
btes_top = [[] for i in range(total_alg_top)]
ftes_top = [[] for i in range(total_alg_top)]
tes_top = [[] for i in range(total_alg_top)]
btes_bottom = [[] for i in range(total_alg_bottom)]
ftes_bottom = [[] for i in range(total_alg_bottom)]
tes_bottom = [[] for i in range(total_alg_bottom)]

#combined_alg_name = ['L2N','L2F','Prog-NN', 'DF-CNN','LwF','EWC','O-EWC','SI', 'Replay (increasing amount)', 'Replay (fixed amount)', 'None']
model_file_combined = ['dnn0','fixed_uf10','Prog_NN','DF_CNN', 'LwF', 'EWC', 'OEWC', 'si', 'offline', 'exact', 'None']

########################

#%% code for 5000 samples
reps = slots*shifts

for alg in range(total_alg_top): 
    count = 0 
    bte_tmp = [[] for _ in range(reps)]
    fte_tmp = [[] for _ in range(reps)] 
    te_tmp = [[] for _ in range(reps)]

    for shift in range(shifts):
        if alg < 2:
            filename = './result/result/'+model_file_top[alg]+'_'+str(shift+1)+'.pickle'
        elif alg <4:
            filename = './benchmarking_algorthms_result/'+model_file_top[alg]+'_'+str(shift+1)+'.pickle'

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

# %%
reps = slots*shifts

for alg in range(total_alg_bottom): 
    count = 0 
    bte_tmp = [[] for _ in range(reps)]
    fte_tmp = [[] for _ in range(reps)] 
    te_tmp = [[] for _ in range(reps)]

    for shift in range(shifts):
        if alg < 1:
            filename = './result/result/'+model_file_bottom[alg]+'_'+str(shift+1)+'.pickle'
        else:
            filename = './benchmarking_algorthms_result/'+model_file_bottom[alg]+'-'+str(shift+1)+'.pickle'

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
te_5000 = {'Odin':np.zeros(10,dtype=float), 'Odif':np.zeros(10,dtype=float), 'Prog-NN':np.zeros(10,dtype=float),
          'DF-CNN':np.zeros(10,dtype=float), 'Odif (constrained)':np.zeros(10,dtype=float), 'LwF':np.zeros(10,dtype=float),
          'EWC':np.zeros(10,dtype=float), 'O-EWC':np.zeros(10,dtype=float), 'SI':np.zeros(10,dtype=float),
          'Total Replay':np.zeros(10,dtype=float), 'Partial Replay':np.zeros(10,dtype=float), 'None':np.zeros(10,dtype=float)}

for count,name in enumerate(te_5000.keys()):
    for i in range(10):
        if count < 4:
            te_5000[name][i] = np.log(tes_top[count][i][9-i])
        else:
            te_5000[name][i] = np.log(tes_bottom[count-4][i][9-i])


df_5000 = pd.DataFrame.from_dict(te_5000)
df_5000 = pd.melt(df_5000,var_name='Algorithms', value_name='Transfer Efficieny')

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
marker_style_scatter = ['.', '.', '.', '.', '.', '.', '+', 'o', '*', '.', '+', 'o']


clr_combined = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#f781bf", "#f781bf", "#f781bf", "#f781bf", "#b15928", "#b15928", "#b15928"]
c_combined = sns.color_palette(clr_combined, n_colors=total_alg_top+total_alg_bottom)

clr_combined_ = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#e41a1c", "#f781bf", "#f781bf", "#f781bf", "#f781bf", "#b15928", "#b15928", "#b15928"]
c_combined_ = sns.color_palette(clr_combined_, n_colors=total_alg_top+total_alg_bottom+1)

fontsize=29
ticksize=26
legendsize=14

ax = fig.add_subplot(gs[:7,:7])

for i, fte in enumerate(ftes_top):
    fte[0] = 1
    if i == 0:
        ax.plot(np.arange(1,11), fte, color=c_top[i], marker=marker_style_top[i], markersize=12, label=alg_name_top[i], linewidth=3)
        continue

    if i == 1:
        ax.plot(np.arange(1,11), fte, color=c_top[i], marker=marker_style_top[i], markersize=12, label=alg_name_top[i], linewidth=3)
        continue
    
    ax.plot(np.arange(1,11), fte, color=c_top[i], marker=marker_style_top[i], markersize=12, label=alg_name_top[i])
    
ax.set_xticks(np.arange(1,11))
ax.set_yticks([0.9, 1, 1.1, 1.2, 1.3])
ax.set_ylim(0.89, 1.31)

log_lbl = np.round(
    np.log([0.9,1,1.1,1.2,1.3]),
    2
)
labels = [item.get_text() for item in ax.get_yticklabels()]

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax.set_yticklabels(labels)

ax.tick_params(labelsize=ticksize)

ax.set_ylabel('log Forward TE', fontsize=fontsize)
ax.set_xlabel('Number of tasks seen', fontsize=fontsize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5)



#ax[0][0].grid(axis='x')
ax = fig.add_subplot(gs[:7,8:15])

for i in range(task_num - 1):

    et = np.zeros((total_alg_top,task_num-i))

    for j in range(0,total_alg_top):
        et[j,:] = np.asarray(btes_top[j][i])

    ns = np.arange(i + 1, task_num + 1)
    for j in range(0,total_alg_top):
        if j == 0:
            if i == 0:
                ax.plot(ns, et[j,:], marker=marker_style_top[j], markersize=8, label = alg_name_top[j], color=c_top[j], linewidth = 3)
            else:
                ax.plot(ns, et[j,:], marker=marker_style_top[j], markersize=8, color=c_top[j], linewidth = 3)
        elif j == 1:
            if i == 0:
                ax.plot(ns, et[j,:], marker=marker_style_top[j], markersize=8, label = alg_name_top[j], color=c_top[j], linewidth = 3)
            else:
                ax.plot(ns, et[j,:], marker=marker_style_top[j], markersize=8, color=c_top[j], linewidth = 3)
        else:
            if i == 0:
                ax.plot(ns, et[j,:], marker=marker_style_top[j], markersize=8, label = alg_name_top[j], color=c_top[j])
            else:
                ax.plot(ns, et[j,:], marker=marker_style_top[j], markersize=8, color=c_top[j])


for i in range(total_alg_top,total_alg_top+total_alg_bottom-1):
    ax.plot(1,0,color=c_combined[i], marker=marker_style[i], markersize=8,label=combined_alg_name[i])

ax.set_xlabel('Number of tasks seen', fontsize=fontsize)
ax.set_ylabel('log Backward TE', fontsize=fontsize)

ax.set_yticks([.8,.9,1, 1.1,1.2])
ax.set_xticks(np.arange(1,11))
ax.set_ylim(0.76, 1.25)

log_lbl = np.round(
    np.log([.8,.9,1,1.1,1.2]),
    2
)
labels = [item.get_text() for item in ax.get_yticklabels()]

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax.set_yticklabels(labels)

ax.tick_params(labelsize=ticksize)
#ax[0][1].grid(axis='x')

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5)

handles, labels_ = ax.get_legend_handles_labels()
#ax.legend(loc='center left', bbox_to_anchor=(.8, 0.5), fontsize=legendsize+16)




ax = fig.add_subplot(gs[:7,16:23])
ax.tick_params(labelsize=22)
ax_ = sns.boxplot(
    x="Algorithms", y="Transfer Efficieny", data=df_5000, palette=c_combined_, whis=np.inf,
    ax=ax, showfliers=False, notch=1
    )
ax.hlines(0, -1,11, colors='grey', linestyles='dashed',linewidth=1.5)
#sns.boxplot(x="Algorithms", y="Transfer Efficieny", data=mean_df, palette=c, linewidth=3, ax=ax[1][1])
#ax_=sns.pointplot(x="Algorithms", y="Transfer Efficieny", data=df_500, join=False, color='grey', linewidth=1.5, ci='sd',ax=ax)
#ax_.set_yticks([.4,.6,.8,1, 1.2,1.4])
ax_.set_xlabel('', fontsize=fontsize)
ax.set_ylabel('log TE after 10 Tasks', fontsize=fontsize-5)
ax_.set_xticklabels(
    ['Odin','Odif', 'ProgNN','DF-CNN', 'Odif (constrained)','LwF','EWC','O-EWC','SI','Total Replay','Partial Replay', 'None'],
    fontsize=18,rotation=65,ha="right",rotation_mode='anchor'
    )

stratified_scatter(te_5000,ax,16,c_combined_,marker_style_scatter)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

#########################################################
ax = fig.add_subplot(gs[8:15,:7])

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
ax.set_yticks([0.75, 1, 1.25])
ax.set_ylim(0.7, 1.3)

log_lbl = np.round(
    np.log([.75,1,1.25]),
    2
)
labels = [item.get_text() for item in ax.get_yticklabels()]

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax.set_yticklabels(labels)

ax.tick_params(labelsize=ticksize)

ax.set_ylabel('Resource Constrained log FTE', fontsize=fontsize)
ax.set_xlabel('Number of tasks seen', fontsize=fontsize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5)

for i in range(0,total_alg_top+total_alg_bottom-1):
    ax.plot(1,0,color=c_combined[i], marker=marker_style[i], markersize=8,label=combined_alg_name[i])

handles, labels_ = ax.get_legend_handles_labels()


#ax[0][0].grid(axis='x')
ax = fig.add_subplot(gs[8:15,8:15])

for i in range(task_num - 1):

    et = np.zeros((total_alg_bottom,task_num-i))

    for j in range(0,total_alg_bottom):
        et[j,:] = np.asarray(btes_bottom[j][i])

    ns = np.arange(i + 1, task_num + 1)
    for j in range(0,total_alg_bottom):
        if j == 0:
            if i == 0:
                ax.plot(ns, et[j,:], marker=marker_style_bottom[j], markersize=8, label=None, color=c_bottom[j], linewidth = 3)
            else:
                ax.plot(ns, et[j,:], marker=marker_style_bottom[j], markersize=8, color=c_bottom[j], linewidth = 3)
        elif j == 1:
            if i == 0:
                ax.plot(ns, et[j,:], marker=marker_style_bottom[j], markersize=8, label = alg_name_bottom[j], color=c_bottom[j])
            else:
                ax.plot(ns, et[j,:], marker=marker_style_bottom[j], markersize=8, color=c_bottom[j], linewidth = 3)
        else:
            if i == 0:
                ax.plot(ns, et[j,:], marker=marker_style_bottom[j], markersize=8, label = alg_name_bottom[j], color=c_bottom[j])
            else:
                ax.plot(ns, et[j,:], marker=marker_style_bottom[j], markersize=8, color=c_bottom[j])


ax.set_xlabel('Number of tasks seen', fontsize=fontsize)
ax.set_ylabel('Resource Constrained log BTE', fontsize=fontsize)

ax.set_yticks([.8, 1, 1.25])
ax.set_xticks(np.arange(1,11))
ax.set_ylim(0.75, 1.28)

log_lbl = np.round(
    np.log([.8,1,1.25]),
    2
)
labels = [item.get_text() for item in ax.get_yticklabels()]

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax.set_yticklabels(labels)

ax.tick_params(labelsize=ticksize)
#ax[0][1].grid(axis='x')

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5)

############################

ax = fig.add_subplot(gs[8:15,16:23])
mean_error, std_error = unpickle('../recruitment_exp/result/recruitment_exp_5000.pickle')
ns = 10*np.array([10, 50, 100, 200, 350, 500])
clr = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
colors = sns.color_palette(clr, n_colors=len(clr))

#labels = ['recruiting', 'Uncertainty Forest', 'hybrid', '50 Random', 'BF', 'building']
labels = ['Odif (building)', 'UF (new)', 'recruiting', 'hybrid']
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
#ax.set_title("CIFAR Recruitment",fontsize=titlesize)
ax.set_xticks([])
ax.set_yticks([0.45, 0.55, 0.65,0.75])
#ax.set_ylim([0.43,0.62])
#ax.text(50, 1, "50", fontsize=ticksize)
ax.text(100, 0.410, "100", fontsize=ticksize-2)
ax.text(500, 0.410, "500", fontsize=ticksize-2)
ax.text(5000, 0.410, "5000", fontsize=ticksize-2)
ax.text(120, 0.380, "Number of Task 10 Samples", fontsize=fontsize-1)

ax.legend(loc='lower left',fontsize=legendsize+6, frameon=False)
ax.set_title('Recruitment Experiment on Task 10', fontsize=fontsize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

fig.legend(handles, labels_, bbox_to_anchor=(.97, .95), fontsize=legendsize+12, frameon=False)
#plt.savefig('result/figs/cifar_exp_500_recruit_with_rep.pdf')
plt.savefig('result/figs/benchmark_5000.pdf', dpi=500)
# %%