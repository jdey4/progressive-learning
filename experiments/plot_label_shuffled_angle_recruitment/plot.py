#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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
    multitask_df = unpickle(filename)
    err = []
    for ii in range(10):
        tmp = 1 - np.array(multitask_df[multitask_df['task']==ii+1]['task_1_accuracy'])
        err.append(tmp)

    return err

#%%
alg_name = ['Odin','Odif','Prog_NN', 'DF_CNN','LwF','EWC','O-EWC','SI', 'ER', 'A-GEM', 'TAG', 'Total Replay', 'Partial Replay', 'None']
model_file = ['dnn0','uf10','Prog_NN','DF_CNN', 'LwF', 'EWC', 'OEWC', 'SI', 'er', 'agem', 'tag', 'offline', 'exact', 'None']
total_alg = 14
slots = 10
shifts = 6

#%% claculate TE for label shuffle
reps = slots*shifts
tes_label_shuffle = [[] for i in range(total_alg)]

for alg in range(total_alg): 
    err_ = np.zeros(10,dtype=float)

    for slot in range(slots):
        for shift in range(shifts):
            if alg < 2:
                filename = './label_shuffle_result/'+model_file[alg]+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
            elif alg == 2 or alg == 3:
                filename = './label_shuffle_result/'+model_file[alg]+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
            else:
                filename = './label_shuffle_result/'+model_file[alg]+'-'+str(slot+1)+'-'+str(shift+1)+'.pickle'

            err_ += np.ravel(np.array(get_error_matrix(filename)))
    
    err_ /= reps
    te = err_[0] / err_

    if alg == 2:
        tes_label_shuffle[alg].extend([1]*10)
    else:
        tes_label_shuffle[alg].extend(te)

#%% calculate TE for rotation experiment
alg_name = ['Odin','Odif','LwF','EWC','O-EWC','SI', 'ER', 'A-GEM', 'TAG', 'Total Replay', 'Partial Replay', 'None']
model_file = ['dnn','uf', 'LwF', 'EWC', 'OEWC', 'si', 'er', 'agem', 'tag', 'offline', 'exact', 'None']
total_alg = 12
angles = range(0,182,4)
tes_angle = [[] for i in range(total_alg)]

for alg in range(total_alg): 
    for angle in angles:
        if alg < 2:
            filename = '../rotation_cifar/results/'+model_file[alg]+'-'+str(angle)+'.pickle'
        else:
            filename = '../rotation_cifar/benchmarking_algorthms_result/'+model_file[alg]+'-'+str(angle)+'.pickle'

        err = unpickle(filename)
        tes_angle[alg].extend([err[0]/err[1]])
# %%
fontsize=24
ticksize=22
fig = plt.figure(constrained_layout=True,figsize=(18,6))
gs = fig.add_gridspec(6, 18)

clr = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#f781bf", "#f781bf", "#f781bf", "#f781bf", "#b15928", "#b15928", "#b15928", "#b15928", "#b15928", "#b15928"]
c = sns.color_palette(clr, n_colors=len(clr))
marker_style = ['.', '.', '.', '.', '.', '+', 'o', '*', '.', '+', 'x', '.', '+', 'o']

ax = fig.add_subplot(gs[:6,:6])

for alg_no,alg in enumerate(alg_name):
    if alg_no<2:
        ax.plot(np.arange(1,11),tes_label_shuffle[alg_no], c=c[alg_no], label=alg_name[alg_no], linewidth=3, marker=marker_style[alg_no])
    else:
        ax.plot(np.arange(1,11),tes_label_shuffle[alg_no], c=c[alg_no], label=alg_name[alg_no], marker=marker_style[alg_no])

ax.set_yticks([.8,.9,1,1.1,1.2])
ax.set_ylim([0.79,1.21])
ax.set_xticks(np.arange(1,11))

log_lbl = np.round(
    np.log([0.8,0.9, 1, 1.1, 1.2]),
    2
)
labels = [item.get_text() for item in ax.get_yticklabels()]

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax.set_yticklabels(labels)


ax.tick_params(labelsize=ticksize)
ax.set_xlabel('Number of tasks seen', fontsize=fontsize)
ax.set_ylabel('log Backward LE', fontsize=fontsize)
ax.set_title("A. Label Shuffled CIFAR", fontsize = fontsize)
ax.hlines(1,1,10, colors='grey', linestyles='dashed',linewidth=1.5)
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
plt.tight_layout()


ax = fig.add_subplot(gs[:6,8:14])
angles = np.arange(0,184,4)
#alg_name = ['SynN','SynF','LwF','EWC','O-EWC','SI', 'Total Replay', 'Partial Replay', 'None']
#clr = ["#377eb8", "#e41a1c", "#f781bf", "#f781bf", "#f781bf", "#f781bf", "#b15928", "#b15928", "#b15928"]
#c = sns.color_palette(clr, n_colors=len(clr))
#marker_style = ['.', '.', '.', '+', 'o', '*', '.', '+', 'o']

for alg_no,alg in enumerate(alg_name):
    if alg_no<2:
        ax.plot(angles,tes_angle[alg_no], c=c[alg_no], label=alg_name[alg_no], linewidth=3, marker=marker_style[alg_no])
    else:
        ax.plot(angles,tes_angle[alg_no], c=c[alg_no], label=alg_name[alg_no], marker=marker_style[alg_no])

ax.set_yticks([.6,.7,.8,.9,1,1.1])
ax.set_ylim([0.6,1.1])
ax.set_xticks([0,30,60,90,120,150,180])
ax.hlines(1,0,180, colors='grey', linestyles='dashed',linewidth=1.5)

log_lbl = np.round(
    np.log([.6,.7,.8,.9,1,1.1]),
    2
)
labels = [item.get_text() for item in ax.get_yticklabels()]

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax.set_yticklabels(labels)

ax.tick_params(labelsize=ticksize)
ax.set_xlabel('Angle of Rotation (Degrees)', fontsize=fontsize)
ax.set_ylabel('log Backward LE', fontsize=fontsize)
ax.set_title("B. Rotation Experiment", fontsize=fontsize)
handles, labels_ = ax.get_legend_handles_labels()
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
plt.tight_layout()

fig.legend(handles, labels_, bbox_to_anchor=(.99, .93), fontsize=20, frameon=False)
plt.savefig('figs/adversary.pdf', dpi=500)

# %%
fig, ax = plt.subplots(1,1, figsize=(8,8))
mean_error = unpickle('recruitment_result/recruitment_mean.pickle')
std_error = unpickle('recruitment_result/recruitment_std.pickle')
ns = 10*np.array([50, 100, 200, 350, 500])
colors = sns.color_palette('Set1', n_colors=mean_error.shape[0]+2)

#labels = ['recruiting', 'Uncertainty Forest', 'hybrid', '50 Random', 'BF', 'building']
labels = ['hybrid', 'building', 'recruiting','50 Random', 'BF', 'Uncertainty Forest' ]
not_included = ['BF', '50 Random']
    
adjust = 0
for i, error_ in enumerate(mean_error[:-1]):
    if labels[i] in not_included:
        adjust +=1
        continue
    ax.plot(ns, mean_error[i], c=colors[i+1-adjust], label=labels[i])
    ax.fill_between(ns, 
            mean_error[i] + 1.96*std_error[i], 
            mean_error[i] - 1.96*std_error[i], 
            where=mean_error[i] + 1.96*std_error[i] >= mean_error[i] - 1.96*std_error[i], 
            facecolor=colors[i+1-adjust], 
            alpha=0.15,
            interpolate=False)

ax.plot(ns, mean_error[-1], c=colors[0], label=labels[-1])
ax.fill_between(ns, 
        mean_error[-1] + 1.96*std_error[-1], 
        mean_error[-1] - 1.96*std_error[-1], 
        where=mean_error[-1] + 1.96*std_error[i] >= mean_error[-1] - 1.96*std_error[-1], 
        facecolor=colors[0], 
        alpha=0.15,
        interpolate=False)


#ax.set_title('CIFAR Recruitment Experiment', fontsize=30)
ax.set_ylabel('Accuracy', fontsize=28)
ax.set_xlabel('Number of Task 10 Samples', fontsize=30)
ax.tick_params(labelsize=28)
ax.set_ylim(0.325, 0.575)
ax.set_title("CIFAR Recruitment",fontsize=30)
ax.set_xticks([500, 2000, 5000])
ax.set_yticks([0.35, 0.45, 0.55])

ax.legend(fontsize=12)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

plt.savefig('figs/recruit.pdf', dpi=500)

# %% convert specificc pickles
'''alg_name = ['PLN','PLF']
model_file = ['dnn0','uf10']
total_alg = 2
slots = 10
shifts = 6


reps = slots*shifts

for alg in range(total_alg): 
    count = 0 
    bte_tmp = [[] for _ in range(reps)]
    fte_tmp = [[] for _ in range(reps)] 
    te_tmp = [[] for _ in range(reps)]

    for slot in range(slots):
        for shift in range(shifts):
            filename = './label_shuffle_result/'+model_file[alg]+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
            print(filename)
            #df = unpickle(filename)
            df = pd.DataFrame()
            print(filename)
            err_ = get_error_matrix_(filename)
            
            df['data_fold'] = [shift]*10
            df['task'] = list(range(1,11))
            df['task_1_accuracy'] = err_

            with open(filename, 'wb') as f:
                pickle.dump(df, f)'''
# %%
'''alg_name = ['LwF', 'EWC', 'OEWC', 'si', 'offline', 'exact', 'None']
model_file = ['LwF', 'EWC', 'OEWC', 'si', 'offline', 'exact', 'None']
total_alg = 7
slots = 10
shifts = 6


reps = slots*shifts

for alg in range(total_alg): 
    for slot in range(slots):
        for shift in range(shifts):
            filename = './label_shuffle_result/'+model_file[alg]+'-'+str(slot+1)+'-'+str(shift+1)+'.pickle'
            print(filename)
            #df = unpickle(filename)
            df = pd.DataFrame()
            print(filename)
            err_ = get_error_matrix_(filename)
            
            df['data_fold'] = [shift]*10
            df['task'] = list(range(1,11))
            df['task_1_accuracy'] = err_

            with open(filename, 'wb') as f:
                pickle.dump(df, f)'''
# %%
