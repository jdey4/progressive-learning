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

    colors=['#FFFFFF',clr]
    cmap = LinearSegmentedColormap.from_list("", list(zip(pos, colors)))
    register_cmap(name, cmap)

def calc_forget(err, total_task, reps):
#Tom Vient et al
    forget = 0
    for ii in range(total_task-1):
        forget += err[ii][ii] - err[total_task-1][ii]

    forget /= (total_task-1)
    return forget/reps

def calc_transfer(err, single_err, total_task, reps):
#Tom Vient et al
    transfer = np.zeros(total_task,dtype=float)

    for ii in range(total_task):
        transfer[ii] = (single_err[ii] - err[total_task-1][ii])/reps

    return np.mean(transfer)

def calc_acc(err, total_task, reps):
#Tom Vient et al
    acc = 0
    for ii in range(total_task):
        acc += (1-err[total_task-1][ii]/reps)
    return acc/total_task

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def calc_avg_acc(err, total_task, reps):
    avg_acc = np.zeros(total_task, dtype=float)
    avg_var = np.zeros(total_task, dtype=float)
    for i in range(total_task):
        avg_acc[i] = (1*(i+1) - np.sum(err[i])/reps + (4-i)*.1)/10
        avg_var[i] = np.var(1-np.array(err[i])/reps)
    return avg_acc, avg_var

def calc_avg_single_acc(err, total_task, reps):
    avg_acc = np.zeros(total_task, dtype=float)
    avg_var = np.zeros(total_task, dtype=float)
    for i in range(total_task):
        avg_acc[i] = (1*(i+1) - np.sum(err[:i+1])/reps + (total_task-1-i)*.1)/10
        avg_var[i] = np.var(1-np.array(err[:i+1])/reps)
    return avg_acc, avg_var
    
def get_fte_bte(err, single_err, total_task):
    bte = [[] for i in range(total_task)]
    te = [[] for i in range(total_task)]
    fte = []
    
    for i in range(total_task):
        for j in range(i,total_task):
            #print(err[j][i],j,i)
            bte[i].append((err[i][i]+1e-2)/(err[j][i]+1e-2))
            te[i].append((single_err[i]+1e-2)/(err[j][i]+1e-2))
                
    for i in range(total_task):
        fte.append((single_err[i]+1e-2)/(err[i][i]+1e-2))
            
            
    return fte,bte,te

def calc_mean_bte(btes,task_num,reps=10):
    mean_bte = [[] for i in range(task_num)]


    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            tmp += np.array(btes[i][j])
        
        tmp=tmp/reps
        mean_bte[j].extend(tmp)
            
    return mean_bte     

def calc_mean_te(tes,task_num,reps=10):
    mean_te = [[] for i in range(task_num)]

    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            tmp += np.array(tes[i][j])
        
        tmp=tmp/reps
        mean_te[j].extend(tmp)
                                             
    return mean_te 

def calc_mean_fte(ftes,task_num,reps=1):
    fte = np.asarray(ftes)
    
    return list(np.mean(np.asarray(fte),axis=0))

def get_error_matrix(filename, total_task):
    multitask_df, single_task_df = unpickle(filename)

    err = [[] for _ in range(total_task)]

    for ii in range(total_task):
        err[ii].extend(
            1 - np.array(
                multitask_df[multitask_df['base_task']==ii+1]['accuracy']
                )
            )
    single_err = 1 - np.array(single_task_df['accuracy'])

    return single_err, err

def sum_error_matrix(error_mat1, error_mat2, total_task):
    err = [[] for _ in range(total_task)]

    for ii in range(total_task):
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
labels = []
btes_all = {}
ftes_all = {}
tes_all = {}
te_scatter = {}
df_all = {}

fle = {}
ble = {}
le = {}
ordr = []

# %%
### MAIN HYPERPARAMS ###
ntrees = 10
slots = 10
task_num = 10
shifts = 6
total_alg = 7
alg_name = ['SynN','SynF', 'Model Zoo','ProgNN', 'LMC', 'DF-CNN', 'CoSCL']

model_file = ['dnn0withrep','fixed_uf10withrep', 'model_zoo','Prog_NN', 'LMC', 'DF_CNN', 'CoSCL']

btes = [[] for i in range(total_alg)]
ftes = [[] for i in range(total_alg)]
tes = [[] for i in range(total_alg)]

########################

#%% code for 500 samples
reps = slots*shifts

for alg in range(total_alg): 
    count = 0 
    bte_tmp = [[] for _ in range(reps)]
    fte_tmp = [[] for _ in range(reps)] 
    te_tmp = [[] for _ in range(reps)]

    for slot in range(slots):
        for shift in range(shifts):
            if alg < 2:
                filename = '/Users/jayantadey/ProgLearn/benchmarks/cifar_exp/result/result/'+model_file[alg]+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
            elif alg == 3 or alg == 5:
                filename = '/Users/jayantadey/ProgLearn/benchmarks/cifar_exp/benchmarking_algorthms_result/'+model_file[alg]+'-'+str(shift+1)+'-'+str(slot+1)+'.pickle'
            elif alg == 2:
                filename = '/Users/jayantadey/ProgLearn/benchmarks/cifar_exp/benchmarking_algorthms_result/'+model_file[alg]+'_'+str(slot+1)+'_'+str(shift+1)+'.pickle'
            elif alg == 6:
                filename = './benchmarking_algorthms_result/'+model_file[alg]+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
            else:
                filename = '/Users/jayantadey/ProgLearn/benchmarks/cifar_exp/benchmarking_algorthms_result/'+model_file[alg]+'-'+str(slot+1)+'-'+str(shift+1)+'.pickle'

            multitask_df, single_task_df = unpickle(filename)

            single_err_, err_ = get_error_matrix(filename, task_num)

            if count == 0:
                single_err, err = single_err_, err_
            else:
                err = sum_error_matrix(err, err_, task_num)
                single_err = list(
                    np.asarray(single_err) + np.asarray(single_err_)
                )

            count += 1
    #single_err /= reps
    #err /= reps
    fte, bte, te = get_fte_bte(err,single_err,task_num)
    avg_acc, avg_var = calc_avg_acc(err, task_num, reps)
    avg_single_acc, avg_single_var = calc_avg_single_acc(single_err, task_num, reps)

    btes[alg].extend(bte)
    ftes[alg].extend(fte)
    tes[alg].extend(te)
    
#%%
te_500 = {'SynN':np.zeros(10,dtype=float), 'SynF':np.zeros(10,dtype=float),
          'Model Zoo':np.zeros(10,dtype=float),
          'ProgNN':np.zeros(10,dtype=float), 'LMC':np.zeros(10,dtype=float),
          'DF-CNN':np.zeros(10,dtype=float),'CoSCL':np.zeros(10,dtype=float)}


task_order = []
t = 1
for count,name in enumerate(te_500.keys()):
    #print(name, count)
    for i in range(10):
        te_500[name][i] = np.log(tes[count][i][9-i])
        task_order.append(t+1)
        t += 1

mean_val = []
for name in te_500.keys():
    mean_val.append(np.mean(te_500[name]))
    print(name, np.round(np.mean(te_500[name]),2), np.round(np.std(te_500[name], ddof=1),2))

arg = np.argsort(mean_val)[::-1]
ordr.append(arg)
algos = list(te_500.keys())
combined_alg_name = []

for ii in arg:
    combined_alg_name.append(
        algos[ii]
    )
    
tmp_te = {}
for id in combined_alg_name:
    tmp_te[id] = te_500[id]

df_le = pd.DataFrame.from_dict(tmp_te)
df_le = pd.melt(df_le,var_name='Algorithms', value_name='Transfer Efficieny')
df_le.insert(2, "Task ID", task_order)
# %%
fle['cifar'] = np.mean(np.log(ftes), axis=1)
ble['cifar'] = []
le['cifar'] = []

bte_end = {'SynN':np.zeros(10,dtype=float), 'SynF':np.zeros(10,dtype=float),
          'Model Zoo':np.zeros(10,dtype=float),
          'ProgNN':np.zeros(10,dtype=float), 'LMC':np.zeros(10,dtype=float),
          'DF-CNN':np.zeros(10,dtype=float),'CoSCL':np.zeros(10,dtype=float)}


for count,name in enumerate(bte_end.keys()):
    #print(name, count)
    for i in range(10):
        bte_end[name][i] = np.log(btes[count][i][9-i])

tmp_ble = {}
for id in combined_alg_name:
    tmp_ble[id] = bte_end[id]

df_ble = pd.DataFrame.from_dict(tmp_ble)
df_ble = pd.melt(df_ble,var_name='Algorithms', value_name='Backward Transfer Efficieny')
df_ble.insert(2, "Task ID", task_order)

#%%
fte_end = {'SynN':np.zeros(10,dtype=float), 'SynF':np.zeros(10,dtype=float),
          'Model Zoo':np.zeros(10,dtype=float),
          'ProgNN':np.zeros(10,dtype=float), 'LMC':np.zeros(10,dtype=float),
          'DF-CNN':np.zeros(10,dtype=float),'CoSCL':np.zeros(10,dtype=float)}


for count,name in enumerate(fte_end.keys()):
    #print(name, count)
    for i in range(10):
        fte_end[name][i] = np.log(ftes[count][i])

tmp_fle = {}
for id in combined_alg_name:
    tmp_fle[id] = fte_end[id]

df_fle = pd.DataFrame.from_dict(tmp_fle)
df_fle = pd.melt(df_fle,var_name='Algorithms', value_name='Forward Transfer Efficieny')
df_fle.insert(2, "Task ID", task_order)
#%%
btes_all['cifar'] = df_ble
ftes_all['cifar'] = df_fle
tes_all['cifar'] = df_le
labels.append(combined_alg_name)

#%%
universal_clr_dict = {'SynN': '#377eb8',
                      'SynF': '#e41a1c',
                      'ProgNN': '#4daf4a',
                      'Model Zoo': '#984ea3',
                      'LMC': '#83d0c9',
                      'CoSCL': '#f781bf',
                      'DF-CNN': '#b15928',
                    }

for ii, name in enumerate(universal_clr_dict.keys()):
    print(name)
    register_palette(name, universal_clr_dict[name])
    
#%%
font=30
datasets = ['CIFAR 10X10']
FLE_yticks = [[-.3,0,.3], [-1.5,0,1], [-.1,0,.4], [-0.4,0,.6], [-1.5,0,.3]]
BLE_yticks = [[-.4,0,.2], [-3,0,2], [-.3,0,.3], [-0.6,0,.2], [-2.5,0,.5]]
LE_yticks = [[-.4,0,.2], [-3,0,2], [-.3,0,.4], [-0.6,0,.6], [-2.5,0,.4]]

#c_top = sns.color_palette('Reds', n_colors=10)

fig, ax = plt.subplots(len(tes_all.keys()), 3, figsize=(24,10))
sns.set_context('talk')

clr_ = []
for name in universal_clr_dict.keys():
    clr_.extend(
        sns.color_palette(
            name, 
            n_colors=task_num
            )
        )
    
for ii, data in enumerate(tes_all.keys()):
    ax_ = sns.stripplot(x='Algorithms', y='Forward Transfer Efficieny', data=ftes_all[data], hue='Task ID', palette=clr_, ax=ax[0], size=25, legend=None)
    ax_.set_xticklabels(
    labels[ii],
    fontsize=font,rotation=65,ha="right",rotation_mode='anchor'
    )
    for xtick, color in zip(ax_.get_xticklabels(), universal_clr_dict.values()):
        xtick.set_color(color)
    ax_.hlines(0, -1,len(labels[ii]), colors='grey', linestyles='dashed',linewidth=1.5, label='chance')

    ax_.set_xlabel('')
    ax_.set_yticks(FLE_yticks[ii])
    ax_.tick_params('y',labelsize=30)
    ax_.set_ylabel('Forward Transfer', fontsize=font+5)

    right_side = ax_.spines["right"]
    right_side.set_visible(False)
    top_side = ax_.spines["top"]
    top_side.set_visible(False)

    ax_ = sns.stripplot(x='Algorithms', y='Backward Transfer Efficieny', data=btes_all[data], hue='Task ID', palette=clr_, ax=ax[1], size=25, legend=None)
    ax_.set_xticklabels(
    labels[ii],
    fontsize=font,rotation=65,ha="right",rotation_mode='anchor'
    )
    for xtick, color in zip(ax_.get_xticklabels(), universal_clr_dict.values()):
        xtick.set_color(color)
    #ax_.set_xticklabels([])
    #ax_.set_xlim([0, len(labels[ii])])
    ax_.hlines(0, -1,len(labels[ii]), colors='grey', linestyles='dashed',linewidth=1.5, label='chance')

    ax_.set_xlabel('')
    ax_.set_yticks(BLE_yticks[ii])
    ax_.tick_params('y', labelsize=30)
    ax_.set_ylabel('Backward Transfer', fontsize=font+5)

    ax_.set_title('CIFAR 10X10 (500 samples)', fontsize=45)
    right_side = ax_.spines["right"]
    right_side.set_visible(False)
    top_side = ax_.spines["top"]
    top_side.set_visible(False)

    ax_ = sns.stripplot(x='Algorithms', y='Transfer Efficieny', data=tes_all[data], hue='Task ID', palette=clr_, ax=ax[2], size=25, legend=None)
    
    ax_.set_xticklabels(
    labels[ii],
    fontsize=font,rotation=65,ha="right",rotation_mode='anchor'
    )
    for xtick, color in zip(ax_.get_xticklabels(), universal_clr_dict.values()):
        xtick.set_color(color)

    ax_.hlines(0, -1,len(labels[ii]), colors='grey', linestyles='dashed',linewidth=1.5, label='chance')

    ax_.set_xlabel('')
    ax_.set_yticks(LE_yticks[ii])
    ax_.tick_params('y', labelsize=30)
    ax_.set_ylabel('Transfer', fontsize=font+5)

    right_side = ax_.spines["right"]
    right_side.set_visible(False)
    top_side = ax_.spines["top"]
    top_side.set_visible(False)

plt.savefig('stripplot_summary_cifar10_500.pdf')
# %%
