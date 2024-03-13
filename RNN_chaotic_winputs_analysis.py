# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 15:33:58 2021

@author: Administrator
"""

import sys
import os

path2 = ['C:/Users/yuriy/Desktop/stuff/RNN_stuff/',
         'C:/Users/ys2605/Desktop/stuff/RNN_stuff/',
         'C:/Users/shymk/Desktop/stuff/RNN_stuff/']

for path3 in path2:
    if os.path.isdir(path3):
        path1 = path3;

#sys.path.append('C:\\Users\\ys2605\\Desktop\\stuff\\mesto\\');
#sys.path.append('/Users/ys2605/Desktop/stuff/RNN_stuff/RNN_scripts');
sys.path.append(path1 + 'RNN_scripts')


from f_analysis import f_plot_rates2, f_plot_rates_only # seriation, 
from f_RNN import f_RNN_test, f_RNN_test_spont, f_gen_ob_dset, f_gen_cont_dset #, f_trial_ave_pad, f_gen_equal_freq_space
from f_RNN_process import f_trial_ave_ctx_pad, f_trial_ave_ctx_pad2, f_trial_sort_data_pad, f_trial_sort_data_ctx_pad, f_label_redundants, f_get_rdc_trav, f_gather_dev_trials, f_gather_red_trials, f_analyze_trial_vectors, f_analyze_cont_trial_vectors # , f_euc_dist, f_cos_sim
from f_RNN_dred import f_run_dred, f_run_dred_wrap, f_proj_onto_dred
from f_RNN_plots import f_plot_dred_rates, f_plot_dred_rates2, f_plot_dred_rates3, f_plot_dred_rates3d, f_plot_traj_speed, f_plot_resp_distances, f_plot_mmn, f_plot_mmn2, f_plot_mmn_dist, f_plot_mmn_freq, f_plot_dred_pcs, f_plot_rnn_weights2, f_plot_run_dist, f_plot_cont_vec_data, f_plot_rd_vec_data, f_plot_ctx_vec_data, f_plot_ctx_vec_dir, f_plot_trial_ave_pca # , f_plot_shadederrorbar
from f_RNN_chaotic import RNN_chaotic
from f_RNN_utils import f_gen_stim_output_templates, f_gen_cont_seq, f_gen_oddball_seq, f_gen_input_output_from_seq, f_plot_examle_inputs, f_plot_train_loss, f_plot_train_test_loss, f_gen_name_tag, f_cut_reshape_rates_wrap, f_plot_exp_var, f_plot_freq_space_distances_control, f_plot_freq_space_distances_oddball, f_save_fig # , f_reshape_rates
from f_RNN_decoder import f_make_cv_groups, f_sample_trial_data_dec, f_run_binwise_dec, f_plot_binwise_dec, f_run_one_shot_dec, f_plot_one_shot_dec_bycat, f_plot_one_shot_dec_iscat

import time
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.patches as patches
#from matplotlib import gridspec
#from matplotlib import colors
import matplotlib.cm as cm
#from random import sample, random
import torch
import torch.nn as nn

sys.path.append(path1 + '../' + 'python_dependencies/dPCA-master/python/dPCA')
from dPCA import dPCA

from sklearn.decomposition import PCA
from sklearn import svm
#from sklearn.model_selection import cross_val_score #, ShuffleSplit, train_test_split
#from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import Isomap
#from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform #, cdist, squareform
#from scipy.sparse import diags
#from scipy import signal
from scipy import linalg
#from scipy.io import loadmat, savemat
#import skimage.io



#%%

#fname_RNN_load = 'test_20k_std3'
#fname_RNN_load = '50k_20stim_std3';

#fname_RNN_ob = 'oddball2_2ctx_20000train_samp_20trials_10stim_64batch_1e-03lr_2023_5_28_13h_15m_RNN'                              # 2 ctx     no train out ;no activation in params (tanh)
#fname_RNN_ob = 'oddball2_2ctx_20000train_samp_20trials_10stim_64batch_1e-03lr_2023_5_28_22h_32m_RNN'                            # 2 ctx     no activation in params (tanh)
#fname_RNN_ob = 'oddball2_2ctx_20000train_samp_20trials_10stim_64batch_1e-03lr_2023_5_28_22h_33m_RNN'                            # 2 ctx     no activation in params (tanh)
#fname_RNN_ob = 'oddball2_2ctx_20000train_samp_20trials_10stim_64batch_1e-03lr_2023_6_5_0h_35m_RNN'                              # 2 ctx     no activation in params (tanh)
#fname_RNN_ob = 'oddball2_2ctx_20000train_samp_ReLU_20trials_10stim_64batch_1e-03lr_2023_6_16_11h_6m_RNN'                        # 2 ctx
#fname_RNN_ob = 'oddball2_2ctx_20000train_samp_ReLU_20trials_10stim_64batch_1e-03lr_2023_6_17_15h_18m_RNN'                       # 2 ctx
#fname_RNN_ob = 'oddball2_2ctx_40000train_samp_ReLU_20trials_10stim_64batch_1e-03lr_2023_6_19_12h_12m_RNN'                       # 2 ctx
#fname_RNN_ob = 'oddball2_2ctx_40000train_samp_ReLU_20trials_20stim_64batch_1e-03lr_2023_6_20_12h_23m_RNN'                       # 2 ctx
#fname_RNN_ob = 'oddball2_2ctx_20000train_samp_tanh_20trials_10stim_64batch_1e-03lr_2023_6_27_13h_44m_RNN'                       # 2 ctx
#fname_RNN_ob = 'oddball2_2ctx_20000train_samp_tanh_20trials_10stim_64batch_1e-03lr_2023_6_28_14h_22m_RNN'                       # 2 ctx
#fname_RNN_ob = 'oddball2_2ctx_20000train_samp_tanh_20trials_10stim_64batch_1e-03lr_2023_6_28_16h_31m_RNN'                       # 2 ctx
#fname_RNN_ob = 'oddball2_2ctx_20000train_samp_tanh_20trials_10stim_64batch_5e-03lr_2023_6_30_12h_30m_RNN'                       # 2 ctx
#fname_RNN_ob = 'oddball2_2ctx_20000train_samp_tanh_20trials_10stim_64batch_1e-02lr_2023_6_30_15h_11m_RNN'                       # 2 ctx
#fname_RNN_ob = 'oddball2_2ctx_20000trainsamp_10neurons_tanh_20trials_10stim_64batch_1e-02lr_2023_7_3_10h_18m_RNN'               # 2 ctx
#fname_RNN_ob = 'oddball2_2ctx_20000trainsamp_20neurons_ReLU_20trials_10stim_64batch_5e-03lr_2023_7_9_14h_12m_RNN'               # 2 ctx
#fname_RNN_ob = 'oddball2_2ctx_20000trainsamp_20neurons_tanh_20trials_10stim_64batch_5e-03lr_2023_7_13_14h_8m_RNN'               # 2 ctx
#fname_RNN_ob = 'oddball2_2ctx_30000trainsamp_25neurons_ReLU_20trials_10stim_64batch_2e-03lr_2023_7_15_14h_42m_RNN'              # 2 ctx
#fname_RNN_ob = 'oddball2_2ctx_40000trainsamp_25neurons_ReLU_20trials_10stim_64batch_1e-03lr_2023_7_15_21h_49m_RNN'              # 2 ctx
#fname_RNN_ob = 'oddball2_2ctx_40000trainsamp_25neurons_ReLU_20trials_10stim_64batch_2e-03lr_2023_7_17_9h_24m_RNN'               # 2 ctx
#fname_RNN_ob = 'oddball2_2ctx_40000trainsamp_25neurons_ReLU_20trials_50stim_64batch_2e-03lr_2023_7_30_21h_33m_RNN'              # 2 ctx
#fname_RNN_ob = 'oddball2_2ctx_40000trainsamp_25neurons_ReLU_20trials_50stim_64batch_1e-03lr_2023_7_31_13h_14m_RNN'              # 2 ctx
#fname_RNN_ob = 'oddball2_2ctx_40000trainsamp_25neurons_ReLU_20trials_50stim_200batch_1e-03lr_2023_7_31_16h_10m_RNN'             # 2 ctx
#fname_RNN_ob = 'oddball2_2ctx_60000trainsamp_25neurons_ReLU_20trials_50stim_200batch_1e-03lr_2023_8_4_17h_41m_RNN'              # 2 ctx
#fname_RNN_ob = 'oddball2_2ctx_80000trainsamp_25neurons_ReLU_20trials_50stim_100batch_1e-03lr_2023_8_5_13h_59m_RNN'              # 2 ctx dset, V shaped learning curve, there is dd axis, not bad
#fname_RNN_ob = 'oddball2_1ctx_20000trainsamp_25neurons_ReLU_20trials_50stim_100batch_1e-03lr_2023_8_14_13h_42m_RNN'             # high loss.... 1 ctx, but still 2 stepish like learning curce
#fname_RNN_ob = 'oddball2_2ctx_80000trainsamp_25neurons_ReLU_20trials_50stim_100batch_1e-03lr_2023_8_15_13h_23m_RNN'             # 2 ctx dist not desired
#fname_RNN_ob = 'oddball2_1ctx_80000trainsamp_25neurons_ReLU_20trials_50stim_100batch_1e-03lr_2023_8_16_18h_48m_RNN'             # 1 ctx, exp decay curve, 1 step
#fname_RNN_ob = 'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_2023_9_11_14h_19m_RNN'     # 1 ctx, tau=0.5
#fname_RNN_ob = 'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise0_2023_9_11_14h_19m_ext_2024_3_6_16h_56m_RNN' # continuation
#fname_RNN_ob = 'oddball2_1ctx_80000trainsamp_25neurons_ReLU_100tau_20trials_50stim_100batch_1e-03lr_2023_9_13_11h_41m_RNN'     # 1 ctx, tau=0.1
#fname_RNN_ob = 'oddball2_1ctx_80000trainsamp_25neurons_ReLU_1000tau_20trials_50stim_100batch_1e-03lr_2023_9_14_11h_58m_RNN'     # 1 ctx, tau=1
#fname_RNN_ob = 'oddball2_1ctx_80000trainsamp_25neurons_ReLU_2000tau_20trials_50stim_100batch_1e-03lr_2023_9_15_12h_18m_RNN'     # 1 ctx, tau=2
#fname_RNN_ob = 'oddball2_1ctx_160000trainsamp_25neurons_ReLU_1000tau_20trials_50stim_100batch_1e-03lr_2023_9_16_13h_10m_RNN'    # 1 ctx, tau=1
#fname_RNN_ob = 'oddball2_1ctx_160000trainsamp_25neurons_ReLU_300tau_20trials_50stim_100batch_1e-03lr_2023_9_17_14h_33m_RNN'    # 1 ctx, tau=.3
#fname_RNN_ob = 'oddball2_1ctx_160000trainsamp_25neurons_ReLU_400tau_20trials_50stim_100batch_1e-03lr_2023_9_18_9h_17m_RNN'     # 1 ctx, tau=.4
#fname_RNN_ob = 'oddball2_1ctx_160000trainsamp_25neurons_ReLU_100tau_20trials_50stim_100batch_1e-03lr_2023_9_18_16h_17m_RNN'    # 1 ctx, tau=.1
#fname_RNN_ob = 'oddball2_1ctx_80000trainsamp_25neurons_ReLU_50tau_5dt_20trials_50stim_100batch_1e-03lr_2023_9_21_11h_15m_RNN'    # 1 ctx, tau=.05
#fname_RNN_ob = 'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_2023_10_4_17h_16m_RNN'    # 1 ctx, tau=.5
#fname_RNN_ob = 'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise0_2023_10_4_17h_16m_ext_2024_3_7_12h_9m_RNN'  # continuation
#fname_RNN_ob = 'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_2023_10_5_10h_54m_RNN'    # 1 ctx, tau=.5
#fname_RNN_ob = 'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise0_2023_10_5_10h_54m_ext_2024_3_8_11h_57m_RNN'  # continuation * exploding??
#fname_RNN_ob = 'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_2023_10_6_17h_3m_RNN'    # 1 ctx, tau=.5
#fname_RNN_ob = 'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2023_12_19_16h_20m_RNN'    # ***** 1 ctx, tau=.5 with noise
#fname_RNN_ob = 'oddball2_1ctx_80000trainsamp_250neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_1_5_13h_15m_RNN'    # 1 ctx, tau=.1 with noise, 250 neurons
#fname_RNN_ob = 'oddball2_1ctx_120000trainsamp_250neurons_ReLU_100tau_10dt_20trials_50stim_100batch_4e-05lr_noise1_2024_1_22_12h_49m_RNN'    # 1 ctx, tau=.1 with noise, 250 neurons
#fname_RNN_ob = 'oddball2_1ctx_120000trainsamp_100neurons_ReLU_100tau_10dt_20trials_50stim_100batch_4e-05lr_noise1_2024_2_1_11h_45m_RNN'    # 1 ctx, tau=.1 with noise, 100 neurons
#fname_RNN_ob = 'oddball2_1ctx_140000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_2_19_10h_15m_RNN'    # 1 ctx, tau=.5 with noise, 25 neurons
#fname_RNN_ob = 'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_2_24_13h_23m_RNN'    #didn't reach low enough loss
#fname_RNN_ob = 'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_2_28_11h_32m_RNN'    #didn't reach low enough loss
#fname_RNN_ob = 'oddball2_1ctx_250000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_3_3_13h_54m_RNN'     #didn't reach low enough loss
#fname_RNN_ob = 'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-03lr_noise1_2024_3_6_16h_3m_RNN'
#fname_RNN_ob = 'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_3_7_12h_8m_RNN'
#fname_RNN_ob = 'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_2e-03lr_noise1_2024_3_10_4h_38m_RNN'
#fname_RNN_ob = 'oddball2_1ctx_200000trainsamp_50neurons_ReLU_500tau_50dt_20trials_50stim_100batch_2e-03lr_noise1_2024_3_10_21h_27m_RNN'
#fname_RNN_ob = 'oddball2_1ctx_200000trainsamp_100neurons_ReLU_500tau_50dt_20trials_50stim_100batch_2e-03lr_noise1_2024_3_11_14h_58m_RNN'
fname_RNN_ob = 'oddball2_1ctx_200000trainsamp_75neurons_ReLU_500tau_50dt_20trials_50stim_100batch_2e-03lr_noise1_2024_3_11_16h_47m_RNN'

 
#fname_RNN_freq = 'freq2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise0_2023_12_20_0h_34m_RNN'      # freq train no noise
fname_RNN_freq = 'freq2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2023_12_20_0h_34m_RNN'      # ***** freq train with noise
#fname_RNN_freq = 'freq2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_1_4_13h_14m_RNN'       # freq train with noise
#fname_RNN_freq = 'freq2_1ctx_80000trainsamp_250neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-04lr_noise1_2024_1_10_11h_28m_RNN'     # freq train with noise, 250 neurons
#fname_RNN_freq = 'freq2_1ctx_120000trainsamp_250neurons_ReLU_500tau_50dt_20trials_50stim_100batch_4e-05lr_noise1_2024_1_11_11h_33m_RNN'    # freq train with noise, 250 neurons


#fname_RNN_save = 'test_50k_std4'
#fname_RNN_save = '50k_20stim_std3'

#%%

params = np.load(path1 + '/RNN_data/' + fname_RNN_ob[:-4] + '_params.npy', allow_pickle=True).item()

params_freq = np.load(path1 + '/RNN_data/' + fname_RNN_freq[:-4] + '_params.npy', allow_pickle=True).item()

if 'train_add_noise' not in params.keys():
    params['train_add_noise'] = 0
    
if 'train_add_noise' not in params_freq.keys():
    params_freq['train_add_noise'] = 0
    
 
#%%

name_tag1, name_tag2 = f_gen_name_tag(params)

name_tag  = name_tag1 + '_' + name_tag2
fname_RNN_save = name_tag


name_tag1_freq, name_tag2_freq = f_gen_name_tag(params_freq)

name_tag_freq  = name_tag1_freq + '_' + name_tag2_freq

#%% generate train data

#plt.close('all')

# generate stim templates
stim_templates = f_gen_stim_output_templates(params)
trial_len = round((params['stim_duration'] + params['isi_duration'])/params['dt'])

# shape (seq_len, batch_size, input_size/output_size, num_samples)
# train control trials 
#trials_train_cont = f_gen_cont_seq(params['num_freq_stim'], params['train_trials_in_sample'], params['train_batch_size'], params['train_num_samples_freq'])
#input_train_cont, output_train_cont = f_gen_input_output_from_seq(trials_train_cont, stim_templates['freq_input'], stim_templates['freq_output'], params)


#trials_train_cont2 = f_gen_cont_seq(params['num_freq_stim'], params['train_trials_in_sample'], params['train_batch_size'], 1)
#input_train_cont2, output_train_cont2 = f_gen_input_output_from_seq(trials_train_cont2, stim_templates['freq_input'], stim_templates['freq_output'], params)


# train oddball trials 
#trials_train_oddball_freq, trials_train_oddball_ctx = f_gen_oddball_seq(params['oddball_stim'], params['train_trials_in_sample'], params['dd_frac'], params['num_ctx'], params['train_batch_size'], params['train_num_samples_ctx'])

#input_train_oddball, output_train_oddball_freq = f_gen_input_output_from_seq(trials_train_oddball_freq, stim_templates['freq_input'], stim_templates['freq_output'], params)
#_, output_train_oddball_ctx = f_gen_input_output_from_seq(trials_train_oddball_ctx, stim_templates['freq_input'], stim_templates['ctx_output'], params)


#%% initialize RNN 

output_size = params['num_freq_stim'] + 1
output_size_ctx = params['num_ctx'] + 1
hidden_size = params['hidden_size'];
alpha = params['dt']/params['tau'];         

rnn = RNN_chaotic(params['input_size'], params['hidden_size'], output_size, output_size_ctx, alpha, params['train_add_noise'], activation=params['activation']).to(params['device'])
rnn.init_weights(params['g'])

# make version of untrained rnn
rnn0 = RNN_chaotic(params['input_size'], params['hidden_size'], output_size, output_size_ctx, alpha, params['train_add_noise'], activation=params['activation']).to('cpu')
rnn0.init_weights(params['g'])

rnnf = RNN_chaotic(params['input_size'], params_freq['hidden_size'], output_size, output_size_ctx, alpha, params['train_add_noise'], activation=params['activation']).to(params_freq['device'])
rnnf.init_weights(params['g'])

#%%
if 'train_loss_weights' not in params.keys():
    params['train_loss_weights'] = [0.1, 0.1, 0.9]

#loss = nn.NLLLoss()

loss_freq = nn.CrossEntropyLoss()

loss_ctx = nn.CrossEntropyLoss(weight = torch.tensor(params['train_loss_weights']).to(params['device']))

# if params['num_ctx'] > 1:
#     loss_ctx = nn.CrossEntropyLoss(weight = torch.tensor(params['train_loss_weights']).to(params['device']))  #1e-10
# else:
#     loss_ctx = nn.BCELoss(weight = torch.tensor(params['train_loss_weights']).to(params['device']))

#%%
print('Loading RNN %s' % fname_RNN_ob)
rnn.load_state_dict(torch.load(path1 + '/RNN_data/' + fname_RNN_ob))

train_out = np.load(path1 + '/RNN_data/' + fname_RNN_ob[:-4] + '_train_out.npy', allow_pickle=True).item()

print('Loading RNN %s' % fname_RNN_freq)
rnnf.load_state_dict(torch.load(path1 + '/RNN_data/' + fname_RNN_freq))

train_out_freq = np.load(path1 + '/RNN_data/' + fname_RNN_freq[:-4] + '_train_out.npy', allow_pickle=True).item()


#%%
figs = f_plot_train_loss(train_out, name_tag1, name_tag2)

figs = f_plot_train_loss(train_out_freq, name_tag1_freq, name_tag2_freq)
  
#%%
params['device'] = 'cpu'
params_freq['device'] = 'cpu'

rnn.cpu()
rnnf.cpu()
rnn0.cpu()

loss_freq.cpu()
loss_ctx.cpu()

#%% create colormap jet 
colors1 = cm.jet(np.linspace(0,1,params['num_freq_stim']))
if 0:
    plt.figure()
    plt.imshow(colors1[:,:3].reshape((50,1,3)), aspect=.2)
    plt.ylabel('color map')
    plt.xticks([])
    
#%% plot RNN params
# plt.close('all')
f_plot_rnn_weights2([rnn, rnnf, rnn0], ('oddball trained', 'freq trained', 'untrained'))

#%% create test inputs
dred_subtr_mean = 0
dred_met = 2
num_skip_trials = 90

num_prepend_zeros = 100


num_dev_stim = 20       # 20
num_red_stim = 20       # 20
num_cont_stim = 50


num_ob_runs = 400
num_ob_trials = 100

num_cont_runs = 50
num_cont_trials = 100

num_const_stim = 50


# oddball data
params_ob = params.copy()
params_ob['dd_frac'] = 0.1


ob_data1 = f_gen_ob_dset(params_ob, stim_templates, num_trials=num_ob_trials, num_runs=num_ob_runs, num_dev_stim=num_dev_stim, num_red_stim=num_red_stim, num_freqs=params['num_freq_stim'], stim_sample='equal', ob_type='one_deviant', freq_selection='sequential', can_be_same = False, can_have_no_dd = True, prepend_zeros=num_prepend_zeros)       # stim_sample= 'random' or 'equal'; ob_type='one_deviant' or 'many_deviant', '100plus1'


# const inputs data
params_const = params.copy()
params_const['isi_duration'] = 0.5  # for const set to zero
params_const['stim_duration'] = 0.5  # for const set to 1
params_const['dd_frac'] = 0
stim_templates_const = f_gen_stim_output_templates(params_const)


ob_data_const = f_gen_ob_dset(params_const, stim_templates_const, num_trials=num_ob_trials, num_runs=num_const_stim, num_freqs=params['num_freq_stim'], num_dev_stim=1, num_red_stim=num_const_stim, stim_sample='equal', ob_type='one_deviant', freq_selection='sequential', can_be_same = False, can_have_no_dd = True, prepend_zeros=num_prepend_zeros)       # stim_sample= 'random' or 'equal'; ob_type='one_deviant' or 'many_deviant', '100plus1'

# make control data
cont_data = f_gen_cont_dset(params, stim_templates, num_trials=num_cont_trials, num_runs=num_cont_runs, num_cont_stim=num_cont_stim, num_freqs=params['num_freq_stim'], prepend_zeros=num_prepend_zeros)

# plt.figure()
# plt.imshow(cont_data['input_control'][:,0,:].T, aspect='auto')

trials_oddball_ctx_cut = ob_data1['trials_oddball_ctx'][num_skip_trials:,:]
trials_oddball_freq_cut = ob_data1['trials_oddball_freq'][num_skip_trials:,:]

trials_const_ctx_cut = ob_data_const['trials_oddball_ctx3'][num_skip_trials:,:]
trials_const_freq_cut = ob_data_const['trials_oddball_freq'][num_skip_trials:,:]

trials_cont_cut = cont_data['trials_control_freq'][num_skip_trials:,:]

red_dd_seq = ob_data1['red_dd_seq']
red_stim_const = ob_data_const['red_dd_seq'][0,:]
test_cont_stim = cont_data['control_stim']

#%% run test data
#test_oddball_freq = f_RNN_test(rnn, loss_freq, input_test_oddball, output_test_oddball_freq, params, paradigm='freq')

test_oddball_ctx = f_RNN_test(rnn, loss_ctx, ob_data1['input_oddball'], ob_data1['target_oddball_ctx'], paradigm='ctx')
testf_oddball_ctx = f_RNN_test(rnnf, loss_ctx, ob_data1['input_oddball'], ob_data1['target_oddball_ctx'], paradigm='ctx')
test0_oddball_ctx = f_RNN_test(rnn0, loss_ctx, ob_data1['input_oddball'], ob_data1['target_oddball_ctx'], paradigm='ctx')

test_const_ctx = f_RNN_test(rnn, loss_ctx, ob_data_const['input_oddball'], ob_data_const['target_oddball_ctx'], paradigm='ctx')
testf_const_ctx = f_RNN_test(rnnf, loss_ctx, ob_data_const['input_oddball'], ob_data_const['target_oddball_ctx'], paradigm='ctx')
test0_const_ctx = f_RNN_test(rnn0, loss_ctx, ob_data_const['input_oddball'], ob_data_const['target_oddball_ctx'], paradigm='ctx')

test_cont_freq = f_RNN_test(rnn, loss_freq, cont_data['input_control'], cont_data['target_control'], paradigm='freq')
testf_cont_freq = f_RNN_test(rnnf, loss_freq, cont_data['input_control'], cont_data['target_control'], paradigm='freq')
test0_cont_freq = f_RNN_test(rnn0, loss_freq, cont_data['input_control'], cont_data['target_control'], paradigm='freq')


#%% process test

# f_plot_rates_only(test_oddball_ctx, 'ctx', num_plot_batches = 1, num_plot_cells = 20, preprocess = True, norm_std_fac = 6, start_from = num_skip_trials*trial_len, plot_extra = 0)

# oddball inputs
f_cut_reshape_rates_wrap(test_oddball_ctx, params_ob, num_skip_trials = num_skip_trials)
f_cut_reshape_rates_wrap(testf_oddball_ctx, params_ob, num_skip_trials = num_skip_trials)
f_cut_reshape_rates_wrap(test0_oddball_ctx, params_ob, num_skip_trials = num_skip_trials)

# constant inputs
f_cut_reshape_rates_wrap(test_const_ctx, params_ob, num_skip_trials = num_skip_trials)
f_cut_reshape_rates_wrap(testf_const_ctx, params_ob, num_skip_trials = num_skip_trials)
f_cut_reshape_rates_wrap(test0_const_ctx, params_ob, num_skip_trials = num_skip_trials)

# control intputs
f_cut_reshape_rates_wrap(test_cont_freq, params_ob, num_skip_trials = num_skip_trials)
f_cut_reshape_rates_wrap(testf_cont_freq, params_ob, num_skip_trials = num_skip_trials)
f_cut_reshape_rates_wrap(test0_cont_freq, params_ob, num_skip_trials = num_skip_trials)


#%% PCA stuff full data

# run dred
f_run_dred_wrap(test_oddball_ctx, subtr_mean=dred_subtr_mean, method=dred_met)
f_run_dred_wrap(testf_oddball_ctx, subtr_mean=dred_subtr_mean, method=dred_met)
f_run_dred_wrap(test0_oddball_ctx, subtr_mean=dred_subtr_mean, method=dred_met)

f_run_dred_wrap(test_cont_freq, subtr_mean=dred_subtr_mean, method=dred_met)
f_run_dred_wrap(testf_cont_freq, subtr_mean=dred_subtr_mean, method=dred_met)
f_run_dred_wrap(test0_cont_freq, subtr_mean=dred_subtr_mean, method=dred_met)

f_run_dred_wrap(test_const_ctx, subtr_mean=dred_subtr_mean, method=dred_met)
f_run_dred_wrap(testf_const_ctx, subtr_mean=dred_subtr_mean, method=dred_met)
f_run_dred_wrap(test0_const_ctx, subtr_mean=dred_subtr_mean, method=dred_met)

# project onto components from oddball dred
f_proj_onto_dred(test_const_ctx, test_oddball_ctx['dred_comp'])
f_proj_onto_dred(testf_const_ctx, testf_oddball_ctx['dred_comp'])
f_proj_onto_dred(test0_const_ctx, test0_oddball_ctx['dred_comp'])

f_proj_onto_dred(test_cont_freq, test_oddball_ctx['dred_comp'])
f_proj_onto_dred(testf_cont_freq, testf_oddball_ctx['dred_comp'])
f_proj_onto_dred(test0_cont_freq, test0_oddball_ctx['dred_comp'])


#f_plot_exp_var([exp_var, exp_varf, exp_var0], ['oddball trained', 'freq trained', 'untrained'], 'oddball inputs')
        
#f_plot_exp_var([exp_var_cont, exp_varf_cont, exp_var0_cont], ['oddball trained', 'freq trained', 'untrained'], 'control inputs')

f_plot_exp_var([test_oddball_ctx['exp_var'], test_cont_freq['exp_var'], testf_oddball_ctx['exp_var'], testf_cont_freq['exp_var'], test0_oddball_ctx['exp_var'], test0_cont_freq['exp_var']], ['ob trained ob in', 'ob trained freq in', 'freq trained ob in', 'freq trained freq in', 'untrained ob in', 'untrained freq in'], max_comps_plot=6)

#%% analyze rates during oddball  
# plt.close('all')

pl_params = {}

pl_params['num_runs_plot'] = 5
pl_params['plot_trials'] = 200
pl_params['color_ctx'] = 1             # 0 = red; 1 = dd
pl_params['mark_red'] = 0
pl_params['mark_dd'] = 0
pl_params['plot_pc'] = [[1, 2], [3, 4], [5, 6]] # , [5, 6], [7, 8]

f_plot_dred_rates(trials_oddball_ctx_cut, test_oddball_ctx['dred_rates4d'], red_dd_seq, pl_params, params, title_tag='ob trained RNN, ob in')

f_plot_dred_rates(trials_oddball_ctx_cut, testf_oddball_ctx['dred_rates4d'], red_dd_seq, pl_params, params, title_tag='freq trained RNN, ob in')

f_plot_dred_rates(trials_oddball_ctx_cut, test0_oddball_ctx['dred_rates4d'], red_dd_seq, pl_params, params, title_tag='untrained RNN, ob in')



f_plot_dred_rates(trials_const_ctx_cut, test_const_ctx['dred_rates4d'], red_dd_seq, pl_params, params, title_tag='ob trained RNN, const in')

f_plot_dred_rates(trials_const_ctx_cut, testf_const_ctx['dred_rates4d'], red_dd_seq, pl_params, params, title_tag='freq trained RNN, const in')



f_plot_dred_rates2(trials_oddball_ctx_cut, test_oddball_ctx['dred_rates4d'], plot_pcs=[[1, 2], [3, 4], [5, 6], [7, 8]], num_runs_plot=10, num_trials_plot=200, run_labels=red_dd_seq[1,:], mark_red=False, mark_dev=True, title_tag='ob trained RNN, ob in') # 


# no stim on off marks
f_plot_dred_rates2(trials_const_ctx_cut, test_const_ctx['dred_rates4d'], plot_pcs=[[3, 4]], num_runs_plot=20, num_trials_plot=200, run_labels=[], mark_red=False, mark_dev=False, title_tag='ob trained RNN, const in') # 

f_plot_dred_rates3d(trials_const_ctx_cut, test_const_ctx['dred_rates4d'], plot_pcs=[[1, 2, 3]], num_runs_plot=20, num_trials_plot=200, el_az_ro=[20, -60, 0], run_labels=[], mark_red=False, mark_dev=False, title_tag='ob trained RNN, const in') # 


# with stim on off marks
f_plot_dred_rates3(trials_const_freq_cut, test_const_ctx['dred_rates4d'], plot_pcs=[[1, 2], [3, 4], [5, 6]], num_runs_plot=50, num_trials_plot=200, rescale_colors=False, title_tag='ob trained RNN, const in')

# control intputs projected onto samecomponents
f_plot_dred_rates3(trials_cont_cut, test_cont_freq['dred_proj_rates4d'], plot_pcs=[[1, 2], [3, 4]], num_runs_plot=2, num_trials_plot=100, rescale_colors=True, title_tag='ob trained RNN, cont in, ob proj') # 
# control intputs 
f_plot_dred_rates3(trials_cont_cut, test_cont_freq['dred_rates4d'], plot_pcs=[[1, 2], [3, 4]], num_runs_plot=2, num_trials_plot=100, rescale_colors=True, title_tag='ob trained RNN, cont in') # 



f_plot_dred_rates3(trials_oddball_freq_cut, test_oddball_ctx['dred_rates4d'], plot_pcs=[[1, 2], [3, 4]], num_runs_plot=100, num_trials_plot=100, rescale_colors=False, title_tag='ob trained RNN') # 


base1 = np.mean(test_cont_freq['dred_proj_rates4d'][:5,:,:,:], axis=0)
dred_proj_rates4d_bs = test_cont_freq['dred_proj_rates4d'] - base1

f_plot_dred_rates3(trials_cont_cut, dred_proj_rates4d_bs, plot_pcs=[[1, 2], [3, 4]], num_runs_plot=2, num_trials_plot=100, rescale_colors=True, title_tag='ob trained RNN, cont in, ob proj') # 


#plt.rcParams['svg.fonttype'] = 'none'

#%%

f_plot_trial_ave_pca(trials_oddball_ctx_cut, test_oddball_ctx['rates4d_cut'], trials_cont_cut, test_cont_freq['rates4d_cut'], params, red_dd_seq, colors1, pc_plot = [[1,2]], baseline_subtract=True)


fig_path = 'C:/Users/ys2605/Desktop/stuff/papers/AC_paper/fig8_RNN_mech'
# f_save_fig(plt.figure(3), path=fig_path, name_tag='')


#%%
# plt.close('all')

f_plot_mmn(test_oddball_ctx['rates4d_cut'], trials_oddball_ctx_cut, params, title_tag='ob trained RNN')

f_plot_mmn(testf_oddball_ctx['rates4d_cut'], trials_oddball_ctx_cut, params, title_tag='freq trained RNN')

f_plot_mmn(test0_oddball_ctx['rates4d_cut'], trials_oddball_ctx_cut, params, title_tag='untrained RNN')


#%%

base_sub = True
split_pos_neg = False

f_plot_mmn2(trials_oddball_ctx_cut, test_oddball_ctx['rates4d_cut'], trials_cont_cut, test_cont_freq['rates4d_cut'], params, red_dd_seq, title_tag='ob trained RNN', baseline_subtract=base_sub, split_pos_cells=split_pos_neg)

f_plot_mmn2(trials_oddball_ctx_cut, testf_oddball_ctx['rates4d_cut'], trials_cont_cut, testf_cont_freq['rates4d_cut'], params_freq, red_dd_seq, title_tag='freq trained RNN', baseline_subtract=base_sub, split_pos_cells=split_pos_neg)

f_plot_mmn2(trials_oddball_ctx_cut, test0_oddball_ctx['rates4d_cut'], trials_cont_cut, test0_cont_freq['rates4d_cut'], params, red_dd_seq, title_tag='untrained RNN', baseline_subtract=base_sub, split_pos_cells=split_pos_neg)


#%%

base_sub = True

f_plot_mmn_dist(trials_oddball_ctx_cut, test_oddball_ctx['rates4d_cut'], trials_cont_cut, test_cont_freq['rates4d_cut'], params, red_dd_seq, title_tag='ob trained RNN', baseline_subtract=base_sub)

f_plot_mmn_dist(trials_oddball_ctx_cut, testf_oddball_ctx['rates4d_cut'], trials_cont_cut, testf_cont_freq['rates4d_cut'], params_freq, red_dd_seq, title_tag='freq trained RNN', baseline_subtract=base_sub)

f_plot_mmn_dist(trials_oddball_ctx_cut, test0_oddball_ctx['rates4d_cut'], trials_cont_cut, test0_cont_freq['rates4d_cut'], params, red_dd_seq, title_tag='untrained RNN', baseline_subtract=base_sub)


# rates4d_cut = test_oddball_ctx['rates4d_cut']
# rates_cont_freq4d_cut = test_cont_freq['rates4d_cut']

#%%

base_sub = True

f_plot_mmn_freq(trials_oddball_ctx_cut, test_oddball_ctx['rates4d_cut'], trials_cont_cut, test_cont_freq['rates4d_cut'], params, red_dd_seq, title_tag='ob trained RNN', baseline_subtract=base_sub)

f_plot_mmn_freq(trials_oddball_ctx_cut, testf_oddball_ctx['rates4d_cut'], trials_cont_cut, testf_cont_freq['rates4d_cut'], params_freq, red_dd_seq, title_tag='freq trained RNN', baseline_subtract=base_sub)

f_plot_mmn_freq(trials_oddball_ctx_cut, test0_oddball_ctx['rates4d_cut'], trials_cont_cut, test0_cont_freq['rates4d_cut'], params, red_dd_seq, title_tag='untrained RNN', baseline_subtract=base_sub)

#%%

zero_trials = 10
ymax = 300

f_plot_run_dist(test_oddball_ctx['rates4d_cut'], plot_runs=50, plot_trials=100, zero_trials=zero_trials, stim_ave_win=[], run_labels = red_dd_seq[0,:], ymax = ymax, title_tag='ob inputs')

f_plot_run_dist(test_const_ctx['rates4d_cut'], plot_runs=50, plot_trials=100, zero_trials=zero_trials, stim_ave_win=[], run_labels = red_stim_const, ymax = ymax, title_tag='const inputs')

f_plot_run_dist(test_cont_freq['rates4d_cut'], plot_runs=20, plot_trials=100, zero_trials=zero_trials, stim_ave_win=[], run_labels = np.zeros((100)), ymax = ymax, title_tag='control inputs')

#%% plot some vectors by freq


trial_len, num_trials, num_runs, num_cells = test_oddball_ctx['rates4d_cut'].shape

plot_t1 = (np.arange(trial_len)-trial_len/4)*params['dt']

freqs_all = np.unique(red_dd_seq)
num_freqs = freqs_all.shape[0]
colors_freq = cm.jet(np.linspace(0,1,num_freqs))

trial_ave_rdc = f_get_rdc_trav(trials_oddball_ctx_cut, test_oddball_ctx['rates4d_cut'], trials_cont_cut, test_cont_freq['rates4d_cut'], plot_t1, red_dd_seq, baseline_subtract=False)

base_time = plot_t1<0
on_time = np.logical_and(plot_t1>.200, plot_t1<.5)

trial_ave_use = trial_ave_rdc[:,0,:,:]

base_act = np.mean(trial_ave_use[base_time,:,:], axis=0)
on_act = np.mean(trial_ave_use[on_time,:,:], axis=0)

freq_dist = np.zeros((num_freqs))
freq_dir = np.zeros((num_freqs, num_cells))
for n_freq in range(num_freqs):
    vec1 = on_act[n_freq,:] - base_act[n_freq,:]
    norm1 = pdist((base_act[n_freq,:], on_act[n_freq,:]))[0]
    #norm1 = np.sqrt(np.sum(vec1**2))
    freq_dist[n_freq] = norm1

    freq_dir[n_freq,:] = vec1/norm1

plt.figure()
plt.plot(freq_dist)
plt.xlabel('freqs')
plt.ylabel('magnitudes')


plt.figure()
plt.imshow(freq_dir)
plt.xlabel('cells')
plt.ylabel('freqs')

dist_cos = squareform(pdist(freq_dir, 'cosine'))
plt.figure()
plt.imshow(1 - dist_cos)
plt.xlabel('freq')
plt.ylabel('freq')
plt.colorbar()
plt.title('cosine similarity between freq directions')

# projected directions of control activity
proj_freq_dir = np.dot(freq_dir, test_oddball_ctx['dred_comp'])

plt.figure()
plt.imshow(proj_freq_dir)
plt.xlabel('pcs')
plt.ylabel('freqs')


plt.figure()
plt.plot(freq_dir)

plt.figure()
plt.plot(proj_freq_dir)

# pcs
pc_plot = [[0, 1], [2, 3], [4, 5], [6, 7]]
for n_pc in range(len(pc_plot)):
    plt.figure()
    pc1 = pc_plot[n_pc]
    for n_freq in range(num_freqs):
        plt.plot([0, proj_freq_dir[n_freq,pc1[0]]], [0, proj_freq_dir[n_freq,pc1[1]]], color=colors_freq[n_freq,:])
    plt.xlabel(pc1[0])
    plt.ylabel(pc1[1])
    
# cells
pc_plot = [[0, 1], [2, 3], [4, 5], [6, 7]]
for n_pc in range(len(pc_plot)):
    plt.figure()
    pc1 = pc_plot[n_pc]
    for n_freq in range(num_freqs):
        plt.plot([0, freq_dir[n_freq,pc1[0]]], [0, freq_dir[n_freq,pc1[1]]], color=colors_freq[n_freq,:])
    plt.xlabel(pc1[0])
    plt.ylabel(pc1[1])


#%% plot control vectors mags of indiv vs trial ave

trials_cont_vec = f_analyze_cont_trial_vectors(test_cont_freq['rates4d_cut'], trials_cont_cut, red_dd_seq, params, base_time = [-.250, 0], on_time = [.2, .5])

f_plot_cont_vec_data(trials_cont_vec, red_dd_seq)


#%% gather deviant trials

freq_red_all = np.unique(red_dd_seq[0,:])
freqs_dev_all = np.unique(red_dd_seq[1,:])
num_freq_r = len(freq_red_all)
num_freq_d = len(freqs_dev_all)

trials_rd_dev = f_gather_dev_trials(test_oddball_ctx['rates4d_cut'], trials_oddball_ctx_cut, red_dd_seq)
# analyze deviant trials

trials_dev_vec = f_analyze_trial_vectors(trials_rd_dev, params)

f_plot_rd_vec_data(trials_dev_vec, ctx_tag = 'deviant')


#%% gather red trials
red_tr_idx = -3
trials_rd_red = f_gather_red_trials(test_oddball_ctx['rates4d_cut'], trials_oddball_freq_cut, trials_oddball_ctx_cut, red_dd_seq, red_idx = red_tr_idx)

trials_red_vec = f_analyze_trial_vectors(trials_rd_red, params)

f_plot_rd_vec_data(trials_red_vec, ctx_tag = 'redundant %d' % red_tr_idx)

#%%
# plt.close('all')

f_plot_ctx_vec_data(trials_cont_vec, trials_dev_vec, trials_red_vec)

#%%
# plt.close('all')

f_plot_ctx_vec_dir(trials_dev_vec['mean_vec_dir'], ctx_tag = 'deviant')

f_plot_ctx_vec_dir(trials_red_vec['mean_vec_dir'], ctx_tag = 'redundant')

#%%

rates_ob = test_oddball_ctx['rates4d_cut']
rates_cont = test_cont_freq['rates4d_cut']

trial_ave_rdc = f_get_rdc_trav(trials_oddball_ctx_cut, rates_ob, trials_cont_cut, rates_cont, params, ob_data1['red_dd_seq'])

trial_ave4d2, trial_data_sort, num_dd_trials = f_trial_ave_ctx_pad2(rates_ob, trials_oddball_ctx_cut, pre_dd = 0, post_dd = 3, max_trials=5, shuffle_trials=True)


trial_ave4d2.shape

trials_oddball_red_fwr, trials_oddball_red_rev = f_label_redundants(trials_oddball_ctx_cut)

freqs_all = np.unique(ob_data1['red_dd_seq'])
num_freqs = freqs_all.shape[0]

ctx_rdc_std_all = np.zeros((trial_len, 3, num_cells, num_freqs))

for n_freq in range(num_freqs):
    freq1 = freqs_all[n_freq]

    freq_idx = ob_data1['red_dd_seq'][1,:]  == freq1
    n_run = np.where(freq_idx)[0][0]
    

    dev_tr_idx = trials_oddball_ctx_cut[:,n_run] == 1
    
    num_tr1 = np.sum(dev_tr_idx)   
    
    rates_ob[:,dev_tr_idx, n_run,:]
    
    ctx_rdc_std_all[:,1,:,n_freq] = np.std(rates_ob[:,dev_tr_idx, n_run,:], axis=1)
    
    cont_run = 0
    cont_tr_idx = trials_cont_cut[:,cont_run] == freq1
    cont_resp_all = rates_cont[:,cont_tr_idx,cont_run,:]
    
    ctx_rdc_std_all[:,2,:,n_freq] = np.mean(cont_resp_all[:,:num_tr1,:], axis=1)
    
    freq_idx = ob_data1['red_dd_seq'][0,:]  == freq1
    n_run = np.where(freq_idx)[0][0]

   
    
    red_tr_idx = trials_oddball_red_rev[:,n_run] == -3
    
    ctx_rdc_std_all[:,0,:,n_freq] = np.mean(rates_ob[:,red_tr_idx, n_run,:], axis=1)

if 1:
    ctx_rdc_std_all2= ctx_rdc_std_all - np.mean(ctx_rdc_std_all[:5,:,:,:], axis=0)
else:
    ctx_rdc_std_all2= ctx_rdc_std_all
    
ctx_rdc_std_all3 = np.mean(np.mean(ctx_rdc_std_all2, axis=3), axis=2)

col_ctx = ['blue', 'red', 'black']
plt.figure()
for n_ctx in range(3):
    plt.plot(plot_t1, ctx_rdc_std_all3[:,n_ctx], color=col_ctx[n_ctx])


#%% generate mmn pca plot

plt.close('all')
plot_t1 = (np.arange(trial_len)-trial_len/4)*params['dt']

red_dd_seq = ob_data1['red_dd_seq']

num_trials5, num_runs5 = trials_oddball_ctx_cut.shape

num_cont_trials, num_cont_runs = trials_cont_cut.shape

trials_oddball_red_fwr, trials_oddball_red_rev = f_label_redundants(trials_oddball_ctx_cut)


freqs_all = np.unique(red_dd_seq)
num_freqs = freqs_all.shape[0]

trial_ave_rdc = np.zeros((trial_len, 3, num_freqs, num_cells))

for n_freq in range(num_freqs):
    freq1 = freqs_all[n_freq]
    
    red_run_idx = red_dd_seq[0,:] == freq1
    dev_run_idx = ob_data1['red_dd_seq'][1,:] == freq1
    
    
    red_resp_all = []
    dev_resp_all = []
    for n_run in range(num_runs5):
        if red_run_idx[n_run]:
            #red_tr_idx = trials_oddball_red_fwr[:,n_run] == 3
            red_tr_idx = trials_oddball_red_rev[:,n_run] == -3
            
            red_resp_all.append(test_oddball_ctx['rates4d_cut'][:,red_tr_idx, n_run,:])
        
        if dev_run_idx[n_run]:
            dev_tr_idx = trials_oddball_ctx_cut[:,n_run] == 1
            
            dev_resp_all.append(test_oddball_ctx['rates4d_cut'][:,dev_tr_idx, n_run,:])
    
    red_resp_all2 = np.concatenate(red_resp_all, axis=1)
    dev_resp_all2 = np.concatenate(dev_resp_all, axis=1)
    
    cont_run = 0
    cont_tr_idx = trials_cont_cut[:,cont_run] == freq1
    cont_resp_all = test_cont_freq['rates4d_cut'][:,cont_tr_idx,cont_run,:]
    
    red_tr_ave = np.mean(red_resp_all2, axis=1)
    
    dev_tr_ave = np.mean(dev_resp_all2, axis=1)
    
    cont_tr_ave = np.mean(cont_resp_all, axis=1)
  
    if 1:
        red_tr_ave3 = red_tr_ave - np.mean(red_tr_ave[:5,:], axis=0)
        dev_tr_ave3 = dev_tr_ave - np.mean(dev_tr_ave[:5,:], axis=0)
        cont_tr_ave3 = cont_tr_ave - np.mean(cont_tr_ave[:5,:], axis=0)
    else:
        red_tr_ave3 = red_tr_ave
        dev_tr_ave3 = dev_tr_ave
        cont_tr_ave3 = cont_tr_ave
    
    trial_ave_rdc[:,0,n_freq,:] = red_tr_ave3
    trial_ave_rdc[:,1,n_freq,:] = dev_tr_ave3
    trial_ave_rdc[:,2,n_freq,:] = cont_tr_ave3


trial_ave_rdc2 = np.reshape(trial_ave_rdc, (trial_len, 3, num_freqs*num_cells), order='F')

colors_ctx = ['blue', 'red', 'black']
plt.figure()
for n_ctx in range(1):
    plt.plot(trial_ave_rdc2[:,n_ctx,:], color=colors_ctx[n_ctx])
    

mmn_mean = np.mean(trial_ave_rdc2[:,:,:],axis=2)
mmn_sem = np.std(trial_ave_rdc2[:,:,:],axis=2)/np.sqrt(num_freqs*num_cells-1)

colors_ctx = ['blue', 'red', 'black']
plt.figure()
for n_ctx in range(3):
    plt.plot(plot_t1, mmn_mean[:,n_ctx], color=colors_ctx[n_ctx])
    plt.fill_between(plot_t1, mmn_mean[:,n_ctx]-mmn_sem[:,n_ctx], mmn_mean[:,n_ctx]+mmn_sem[:,n_ctx], color=colors_ctx[n_ctx], alpha=0.2)


#%%

colors_cells = cm.jet(np.linspace(0,1,num_cells))

trials_oddball_red_fwr, trials_oddball_red_rev = f_label_redundants(trials_oddball_ctx_cut)


for n_freq in range(5):
    freq1 = test_cont_stim[n_freq]
    
    red_run_idx = ob_data1['red_dd_seq'][0,:] == freq1
    red_run = np.where(red_run_idx)[0][0]
    
    #red_tr_idx = trials_oddball_red_fwr[:,red_run] == 3
    red_tr_idx = trials_oddball_red_rev[:,red_run] == -3
    
    dev_run_idx = ob_data1['red_dd_seq'][1,:] == freq1
    dev_run = np.where(dev_run_idx)[0][0]
    
    dev_tr_idx = trials_oddball_ctx_cut[:,dev_run] == 1
    
    cont_run = 0
    cont_tr_idx = trials_cont_cut[:,cont_run] == freq1
    
    
    
    rates_red = test_oddball_ctx['rates4d_cut'][:,red_tr_idx, red_run,:]
    rates_dev = test_oddball_ctx['rates4d_cut'][:,dev_tr_idx, dev_run,:]
    rates_cont = test_cont_freq['rates4d_cut'][:,cont_tr_idx,cont_run,:]
    
    
    base_red = np.mean(rates_red[:5,:,:], axis=0)[None,:,:]
    base_dev = np.mean(rates_dev[:5,:,:], axis=0)[None,:,:]
    base_cont = np.mean(rates_cont[:5,:,:], axis=0)[None,:,:]
    
    rates_redn = rates_red - base_red
    rates_devn = rates_dev - base_dev
    rates_contn = rates_cont - base_cont
    
    
    rates_redn_trav = np.mean(rates_redn, axis=1)
    rates_devn_trav = np.mean(rates_devn, axis=1)
    rates_contn_trav = np.mean(rates_contn, axis=1)
    
    
    range_pl = range(25)
    
    plt.figure()
    plt.subplot(1,3,1)
    for n_cell in range_pl:
        plt.plot(rates_redn[:,:,n_cell], color=colors_cells[n_cell,:])
    plt.title('redundant')
    plt.subplot(1,3,2)
    for n_cell in range_pl:
        plt.plot(rates_devn[:,:,n_cell], color=colors_cells[n_cell,:])
    plt.title('deviant')
    plt.subplot(1,3,3)
    for n_cell in range_pl:
        plt.plot(rates_contn[:,:,n_cell], color=colors_cells[n_cell,:])
    plt.title('control')
    plt.suptitle('freq %d' % freq1)
    
    plt.figure()
    plt.subplot(1,3,1)
    for n_cell in range_pl:
        plt.plot(rates_redn_trav[:,n_cell], color=colors_cells[n_cell,:])
    plt.title('redundant')
    plt.subplot(1,3,2)
    for n_cell in range_pl:
        plt.plot(rates_devn_trav[:,n_cell], color=colors_cells[n_cell,:])
    plt.title('deviant')
    plt.subplot(1,3,3)
    for n_cell in range_pl:
        plt.plot(rates_contn_trav[:,n_cell], color=colors_cells[n_cell,:])
    plt.title('control')
    plt.suptitle('freq %d' % freq1)
    
    
    plt.figure()
    plt.plot(np.mean(rates_redn_trav, axis=1), color='blue')
    plt.plot(np.mean(rates_devn_trav, axis=1), color='red')
    plt.plot(np.mean(rates_contn_trav, axis=1), color='black')
    plt.title('freq %d' % freq1)
    
    
    mmn_data = np.concatenate((rates_redn_trav, rates_devn_trav, rates_contn_trav), axis=0)
    
    proj_data_mmn, exp_var_mmn, dred_comp_mmn, dred_mean_mmn = f_run_dred(mmn_data, subtr_mean=dred_subtr_mean, method=dred_met)
    
    proj_data_mmn3d = np.reshape(proj_data_mmn, [20, 3, num_cells], order='F')
    
    
    plt.figure()
    plt.plot(proj_data_mmn3d[:,0,0], proj_data_mmn3d[:,0,1], color='blue')
    plt.plot(proj_data_mmn3d[:,1,0], proj_data_mmn3d[:,1,1], color='red')
    plt.plot(proj_data_mmn3d[:,2,0], proj_data_mmn3d[:,2,1], color='black')
    plt.title('freq %d' % freq1)
    
    
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(proj_data_mmn3d[:,0,0], proj_data_mmn3d[:,0,1], proj_data_mmn3d[:,0,2], color='blue')
    ax.plot(proj_data_mmn3d[:,1,0], proj_data_mmn3d[:,1,1], proj_data_mmn3d[:,1,2], color='red')
    ax.plot(proj_data_mmn3d[:,2,0], proj_data_mmn3d[:,2,1], proj_data_mmn3d[:,2,2], color='black')
    ax.view_init(elev=30., azim=-60, roll=0)  # 30, -60, 0
    plt.title('freq %d' % freq1)


#%%

rates_in = [test_cont_freq['rates4d_cut'], testf_cont_freq['rates4d_cut'], test0_cont_freq['rates4d_cut']]
legend_in = ['ob trained', 'freq trained', 'untrained']

dist_cont = f_plot_freq_space_distances_control(rates_in, trials_cont_cut, params, legend_in, base_correct = False, metric = 'euclidean')

#%%

rates_in = [test_oddball_ctx['rates4d_cut'], testf_oddball_ctx['rates4d_cut'], test0_oddball_ctx['rates4d_cut']]
legend_in = ['ob trained', 'freq trained', 'untrained']

dist_ctx = f_plot_freq_space_distances_oddball(rates_in, trials_oddball_ctx_cut, ob_data1['red_dd_seq'], params, legend_in, base_correct = False, metric = 'euclidean')

#%%
freq1 = np.sort(np.unique(ob_data1['red_dd_seq']))
num_net = len(dist_cont)
num_freq = freq1.shape[0]

dist_in = [dist_ctx[0], dist_ctx[1], dist_cont]
ctx_labels_in = ['Redundant', 'Deviant', 'Control'] 
net_labels_in = ['ob trained', 'freq trained', 'untrained']

num_ctx = len(dist_in)
num_net = len(dist_in[0])

for n_ctx in range(num_ctx):
    dist_mean = np.zeros((num_net, num_freq))
    dist_std = np.zeros((num_net, num_freq))
    for n_net in range(num_net):
        for n_freq in range(num_freq):
            dist_vals = np.diagonal(dist_in[n_ctx][n_net], offset=n_freq)
            dist_mean[n_net, n_freq] = np.mean(dist_vals)
            dist_std[n_net, n_freq] = np.std(dist_vals)
            
    plt.figure()
    for n_net in range(num_net):
        plt.plot(np.arange(num_freq), dist_mean[n_net,:])
    plt.legend(net_labels_in)
    for n_net in range(num_net):
        plt.fill_between(np.arange(num_freq), dist_mean[n_net,:]-dist_std[n_net,:], dist_mean[n_net,:]+dist_std[n_net,:], alpha=0.2)
    plt.xlabel('Freq difference')
    plt.ylabel('Euclidean distance')
    plt.title('%s responses' % ctx_labels_in[n_ctx])


#%%

freq1 = np.sort(np.unique(ob_data1['red_dd_seq']))
num_net = len(dist_cont)
num_freq = freq1.shape[0]

dist_in = [dist_ctx[0], dist_ctx[1], dist_cont]
ctx_labels_in = ['Redundant', 'Deviant', 'Control'] 
net_labels_in = ['ob trained', 'freq trained', 'untrained']


colors_ctx = ['blue', 'red', 'black']

num_ctx = len(dist_in)
num_net = len(dist_in[0])

for n_net in range(num_net):

    dist_mean = np.zeros((num_ctx, num_freq))
    dist_std = np.zeros((num_ctx, num_freq))
    for n_ctx in range(num_ctx):
        for n_freq in range(num_freq):
            dist_vals = np.diagonal(dist_in[n_ctx][n_net], offset=n_freq)
            dist_mean[n_ctx, n_freq] = np.mean(dist_vals)
            dist_std[n_ctx, n_freq] = np.std(dist_vals)
            
    plt.figure()
    for n_ctx in range(num_ctx):
        plt.plot(np.arange(num_freq), dist_mean[n_ctx,:], color=colors_ctx[n_ctx])
    plt.legend(ctx_labels_in)
    for n_ctx in range(num_ctx):
        plt.fill_between(np.arange(num_freq), dist_mean[n_ctx,:]-dist_std[n_ctx,:], dist_mean[n_ctx,:]+dist_std[n_ctx,:], color=colors_ctx[n_ctx], alpha=0.2)
    plt.xlabel('Freq difference')
    plt.ylabel('Euclidean distance')
    plt.title('%s responses' % net_labels_in[n_net])

 
#%% parameter selection for binwise decoders
# plt.close('all')

pre_tr1 = 1
post_tr1 = 5
num_tr_ave = pre_tr1 + post_tr1 + 1

plot_t1 = (np.arange(num_tr_ave*trial_len)-pre_tr1*trial_len-trial_len/4)*params['dt']

cond_bin = np.argmin(np.abs(0.25 - plot_t1))

#type1 = 'dd decoding'
dec_type1 = 'red'  # red dd
train_test_method = 'diag'          # full, diag, train_at_stim, test_at_stim

#%% oddball freq decoder, binwise single trial decoding of deviance or redundance freq during dd trial of oddball, with cv

max_trials = 2
shuffle_trials = False

trial_data_sort_dd1, dd_freqs, num_dd_trials1 = f_trial_sort_data_ctx_pad(test_oddball_ctx['rates4d_cut'], trials_oddball_ctx_cut, trials_oddball_freq_cut, pre_trials = pre_tr1, post_trials = post_tr1, max_trials=max_trials, shuffle_trials=shuffle_trials)
trial_data_sortf_dd1, _, _ = f_trial_sort_data_ctx_pad(testf_oddball_ctx['rates4d_cut'], trials_oddball_ctx_cut, trials_oddball_freq_cut, pre_trials = pre_tr1, post_trials = post_tr1, max_trials=max_trials, shuffle_trials=shuffle_trials)
trial_data_sort0_dd1, _, _ = f_trial_sort_data_ctx_pad(test0_oddball_ctx['rates4d_cut'], trials_oddball_ctx_cut, trials_oddball_freq_cut, pre_trials = pre_tr1, post_trials = post_tr1, max_trials=max_trials, shuffle_trials=shuffle_trials)

red_freqs = []
for n_run in range(num_runs):
    red_freqs.append(np.ones(num_dd_trials1[n_run], dtype=int) * ob_data1['red_dd_seq'][0,n_run])

dd_freqs2 = np.concatenate(dd_freqs, axis=0)
red_freqs2 = np.concatenate(red_freqs, axis=0)

trial_data_sort_dd2 = np.concatenate(trial_data_sort_dd1, axis=1)
trial_data_sortf_dd2 = np.concatenate(trial_data_sortf_dd1, axis=1)
trial_data_sort0_dd2 = np.concatenate(trial_data_sort0_dd1, axis=1)

# SVC does one vs one classif
# LinearSVC does one vs all
# SVR is for regression

X_all = [trial_data_sort_dd2, trial_data_sortf_dd2, trial_data_sort0_dd2, trial_data_sort_dd2]

if dec_type1 == 'dd':
    y_data = dd_freqs2
elif dec_type1 == 'red':
    y_data = red_freqs2

leg1 = ('ob trained', 'freq trained', 'untrained', 'ob trained shuff')
shuff_stim_type = [0, 0, 0, 1]

preform2_train_test = f_run_binwise_dec(X_all, y_data, shuff_stim_type, train_test_method=train_test_method, pca_var = 1, cond_bin=cond_bin)

if 0:
    preform2_train_test = np.load(path1 + '/RNN_data/decoder_data/' + '%s_%s_decoding_%s_%ddec_postdd%d.npy' % (fname_RNN_save, dec_type1, train_test_method, len(X_all), post_tr1))
    if len(preform2_train_test.shape) >3:
        preform2_train_test = np.mean(preform2_train_test, axis=2)

f_plot_binwise_dec(preform2_train_test, plot_t1, leg1, train_test_method=train_test_method, plt_start=-1, plot_end=5, plot_cont=0.25, title_tag='oddball %s freq single trial decoder' % (dec_type1))

if 0:
    np.save(path1 + '/RNN_data/decoder_data/' + '%s_%s_decoding_%s_%ddec_postdd%d' % (fname_RNN_save, dec_type1, train_test_method, len(X_all), post_tr1), preform2_train_test)

#%% oddball freq decoder, binwise trial ave, decoding of deviance or redundance freq during dd trial of oddball, with cv, finalized ver, single trial data, prelim decoder ob 
# plt.close('all')
#
trial_ave3d = f_trial_ave_ctx_pad(test_oddball_ctx['rates4d_cut'], trials_oddball_ctx_cut, pre_dd = pre_tr1, post_dd = post_tr1)
trial_ave3df= f_trial_ave_ctx_pad(testf_oddball_ctx['rates4d_cut'], trials_oddball_ctx_cut, pre_dd = pre_tr1, post_dd = post_tr1)
trial_ave3d0 = f_trial_ave_ctx_pad(test0_oddball_ctx['rates4d_cut'], trials_oddball_ctx_cut, pre_dd = pre_tr1, post_dd = post_tr1)

use_idx = ~np.isnan(np.mean(np.mean(trial_ave3d,axis=2), axis=0))

trial_ave3d_use = trial_ave3d[:,use_idx,:]
trial_ave3df_use = trial_ave3df[:,use_idx,:]
trial_ave3d0_use = trial_ave3d0[:,use_idx,:]

red_dd_trial_types = ob_data1['red_dd_seq'][:,use_idx]

if dec_type1 == 'dd':
    y_data = red_dd_trial_types[1,:]
elif dec_type1 == 'red':
    y_data = red_dd_trial_types[0,:]
    
# SVC does one vs one classif
# LinearSVC does one vs all
# SVR is for regression

X_all = [trial_ave3d_use, trial_ave3df_use, trial_ave3d0_use, trial_ave3d_use]
leg1 = ('ob trained', 'freq trained', 'untrained', 'ob shuff')
shuff_stim_type = [0, 0, 0, 1]

preform2_train_test = f_run_binwise_dec(X_all, y_data, shuff_stim_type, train_test_method=train_test_method, pca_var = 1, cond_bin=cond_bin)

f_plot_binwise_dec(preform2_train_test, plot_t1, leg1, train_test_method='diag', plt_start=-1, plot_end=5, plot_cont=0.25, title_tag='oddball %s freq trial ave decoder' % (dec_type1))

#%% control freq decoder, single trial data 

trial_data_sort_cont, trial_types_cont, trial_types_cont_pad = f_trial_sort_data_pad(test_cont_freq['rates4d_cut'], trials_cont_cut, pre_trials = pre_tr1, post_trials = post_tr1)
trial_data_sort_contf, _, _ = f_trial_sort_data_pad(testf_cont_freq['rates4d_cut'], trials_cont_cut, pre_trials = pre_tr1, post_trials = post_tr1)
trial_data_sort_cont0, _, _ = f_trial_sort_data_pad(test0_cont_freq['rates4d_cut'], trials_cont_cut, pre_trials = pre_tr1, post_trials = post_tr1)

n_run = 0

trial_data_sort_cont_use = trial_data_sort_cont[:,:,n_run,:]
trial_data_sort_contf_use = trial_data_sort_contf[:,:,n_run,:]
trial_data_sort_cont0_use = trial_data_sort_cont0[:,:,n_run,:]

y_data = trial_types_cont[:,n_run]

X_all = [trial_data_sort_cont_use, trial_data_sort_contf_use, trial_data_sort_cont0_use, trial_data_sort_cont_use]
leg1 = ('ob trained', 'freq trained', 'untrained', 'ob shuff')
shuff_stim_type = [0, 0, 0, 1]


preform2_train_test = f_run_binwise_dec(X_all, y_data, shuff_stim_type, train_test_method=train_test_method, pca_var = 1, cond_bin=cond_bin)

f_plot_binwise_dec(preform2_train_test, plot_t1, leg1, train_test_method='diag', plt_start=-1, plot_end=5, plot_cont=0.25, title_tag='control freq single trial decoder')

#%% one-shot context - analyze rnn preformance with decoder, specific decoding for dd and red and off

trial_len = round((params['stim_duration']+params['isi_duration'])/params['dt'])
trial_stim_on = np.zeros(trial_len, dtype=bool)
trial_stim_on[round(np.floor(params['isi_duration']/params['dt']/2)):(round(np.floor(params['isi_duration']/params['dt']/2))+round(params['stim_duration']/params['dt']))] = 1
plot_t1 = (np.arange(trial_len)-trial_len/4)*params['dt']

#
stim_loc = trials_oddball_ctx_cut
rates_in = [test_oddball_ctx['rates4d_cut'], testf_oddball_ctx['rates4d_cut'], test0_oddball_ctx['rates4d_cut'], test_oddball_ctx['rates4d_cut'], test_oddball_ctx['rates4d_cut']]

leg_all = ['ob trained', 'freq trained', 'untrained', 'ob stim shuff', 'ob bin shuff']
shuff_stim_type = [0, 0, 0, 1, 0]
shuff_bins = [0, 0, 0, 0, 1]

x_data, y_data = f_sample_trial_data_dec(rates_in, stim_loc, [1, 0])

perform1_final, perform1_binwise, perform1_y_is_cat = f_run_one_shot_dec(x_data, y_data, trial_stim_on, shuff_stim_type, shuff_bins, stim_on_train=False, num_cv=5, equalize_y_input=True)


f_plot_one_shot_dec_bycat(perform1_final, perform1_binwise, perform1_y_is_cat, plot_t1, leg_all, ['deviant', 'redundant'], ['pink', 'lightblue'])

#%% analyze rnn preformance with decoder, decoding for deviant vs not deviant

stim_loc = trials_oddball_ctx_cut
rates_in = [test_oddball_ctx['rates4d_cut'], testf_oddball_ctx['rates4d_cut'], test0_oddball_ctx['rates4d_cut'], test_oddball_ctx['rates4d_cut'], test_oddball_ctx['rates4d_cut']]

leg_all = ['ob trained', 'freq trained', 'untrained', 'ob stim shuff', 'ob bin shuff']
shuff_stim_type = [0, 0, 0, 1, 0]
shuff_bins = [0, 0, 0, 0, 1]


x_data, y_data = f_sample_trial_data_dec(rates_in, stim_loc, [1, 0])
y_data1 = (y_data==1).astype(int)

perform1_final, perform1_binwise, perform1_y_is_cat = f_run_one_shot_dec(x_data, y_data1, trial_stim_on, shuff_stim_type, shuff_bins, stim_on_train=True, num_cv=5, equalize_y_input=False)

f_plot_one_shot_dec_iscat(perform1_final, perform1_binwise, perform1_y_is_cat, plot_t1, leg_all, ['deviant', 'not deviant'], ['pink', 'gray'], cat_plot=1)

#%% decoding frequency identity from control inputs

colors10 = cm.jet(np.linspace(0,1, num_cont_stim))
colors10[:,3] = 0.3

n_run = 0

y_data = trials_cont_cut[:,n_run]
x_data = [test_cont_freq['rates4d_cut'][:,:,n_run,:], testf_cont_freq['rates4d_cut'][:,:,n_run,:], test0_cont_freq['rates4d_cut'][:,:,n_run,:], test_cont_freq['rates4d_cut'][:,:,n_run,:], test_cont_freq['rates4d_cut'][:,:,n_run,:]]
leg_all = ['ob trained', 'freq trained', 'untrained', 'ob stim shuff', 'ob bin shuff']
shuff_stim_type = [0, 0, 0, 1, 0]
shuff_bins = [0, 0, 0, 0, 1]


perform1_final, perform1_binwise, perform1_y_is_cat = f_run_one_shot_dec(x_data, y_data, trial_stim_on, shuff_stim_type, shuff_bins, stim_on_train=False, num_cv=5, equalize_y_input=True)

f_plot_one_shot_dec_bycat(perform1_final, perform1_binwise, perform1_y_is_cat, plot_t1, leg_all, [], colors10)


#%% try analyze binwise rnn preformance with decoder  (not sure about this)

num_cv = 5

trial_stim_on = np.zeros((trial_len), dtype=bool)
trial_stim_on[5:15] = 1

num_stim_on = np.sum(trial_stim_on)

t_bin_use = 10

plot_t1 = (np.arange(1*trial_len)-0*trial_len-trial_len/4)*params['dt']

stim_loc = trials_oddball_ctx_cut
rates_in = [test_oddball_ctx['rates4d_cut'], testf_oddball_ctx['rates4d_cut'], test0_oddball_ctx['rates4d_cut']]
leg_all = ['ob trained', 'freq trained', 'untrained']

num_dec = len(rates_in)

#rates4d_cut.shape
#trials_oddball_ctx_cut

dd_red_samp_rates2, y_data = f_sample_trial_data_dec(rates_in, stim_loc, [1, 0])

num_run_use = dd_red_samp_rates2[0].shape[1]


#y_temp = np.hstack((np.ones((num_stim_on)),np.zeros((num_stim_on)))).astype(int).reshape((num_stim_on*2,1))

y_data1 = (y_data==1).astype(int)

y_temp = np.array([1, 0]).reshape([2,1], order = 'F')

perform1_train_test = np.zeros((trial_len, num_cv, num_dec))
for n_dec in range(num_dec):
    cv_groups = f_make_cv_groups(num_run_use, num_cv)
    
    for n_cv in range(num_cv):
        test_idx = cv_groups[n_cv]
        train_idx = ~test_idx
        num_train = np.sum(train_idx)
        num_test = np.sum(test_idx)
        
        y_train = y_data1[train_idx]
        y_test = y_data1[test_idx]
        
        # y_train = np.repeat(y_temp, num_train, axis=1)
        # y_train2 = np.reshape(y_train, (2*num_train), order = 'F')
        
        # y_test = np.repeat(y_temp, num_test, axis=1)
        # y_test2 = np.reshape(y_test, (2*num_test), order = 'F')
        
        #temp_data = np.mean(dd_red_samp_rates2[trial_stim_on,:,:,:], axis=0)
        
        for n_t in range(trial_len):
        
            dd_red_samp_rates3 = dd_red_samp_rates2[n_dec]
            temp_data = dd_red_samp_rates3[n_t,:,:]
            
            train_data = temp_data[train_idx,:]
 
            test_data = temp_data[test_idx,:]

            svc = svm.SVC(kernel='linear', C=1, gamma='auto')
            svc.fit(train_data, y_train)
            
            test_pred1 = svc.predict(test_data)
            
            perform1_train_test[n_t, n_cv, n_dec] = np.sum(y_test == test_pred1)/test_pred1.shape[0]
    
perform1_final = np.mean(perform1_train_test, axis=1)

plt.figure()
for n_dec in range(num_dec):
    plt.plot(plot_t1, perform1_final[:,n_dec])
plt.legend(leg_all)
plt.title('binwise context decoding')

#%%
rates_in = test_const_ctx['dred_rates4d']
#rates_in = testf_const_ctx['dred_rates4d']
#rates_in = test0_const_ctx['dred_rates4d']

trial_len, num_trials, num_runs, num_cells = rates_in.shape

rates_in3d = np.reshape(rates_in, (trial_len*num_trials, num_runs, num_cells), order='F')

plot_trials = 50
plot_runs = 50
plot_pc2 = [1, 2]
plot_T = plot_trials*trial_len

plot_pc = [[1, 2], [3, 4]] 

for n_np in range(len(plot_pc)):
    plot_pc2 = plot_pc[n_np]
    
    plt.figure()
    plt.plot(rates_in3d[:plot_T, 0, plot_pc2[0]-1], rates_in3d[:plot_T, 0, plot_pc2[1]-1], color='black')
    for n_red in range(plot_runs):
        red_fr = n_red
        
        plt.plot(rates_in3d[:plot_T, red_fr, plot_pc2[0]-1], rates_in3d[:plot_T, red_fr, plot_pc2[1]-1], color=colors1[n_red,:])
    plt.xlabel('PC %d' % plot_pc2[0])
    plt.ylabel('PC %d' % plot_pc2[1])
    plt.title('const input trained')


#%% trial averages
# plt.close('all')

pre_dd1 = 4
post_dd1 = 4
num_tr_ave = pre_dd1 + post_dd1 + 1

#
trial_ave4d, trial_data_sort, num_dd_trials = f_trial_ave_ctx_pad2(test_oddball_ctx['rates4d_cut'], trials_oddball_ctx_cut, pre_dd = pre_dd1, post_dd = post_dd1)

trial_ave3d = np.reshape(trial_ave4d, (trial_len*num_tr_ave, num_runs, num_cells), order = 'F')

trial_ave2d = np.reshape(trial_ave3d, (trial_len*num_tr_ave*num_runs, num_cells), order = 'F')


proj_data_ta, exp_var_ta, _, _ = f_run_dred(trial_ave2d, subtr_mean=dred_subtr_mean, method=dred_met)

comp_out3d_ta = np.reshape(proj_data_ta, (trial_len*num_tr_ave, num_runs, num_cells), order = 'F')
comp_out4d_ta = np.reshape(proj_data_ta, (trial_len, num_tr_ave, num_runs, num_cells), order = 'F')


# isomap

num_comp = 4

embedding = Isomap(n_components=num_comp, metric='cosine')
X_transformed = embedding.fit_transform(trial_ave2d)
X_transformed3d = np.reshape(X_transformed, (trial_len*num_tr_ave, num_runs, num_comp), order = 'F')

n_run = 1

plt.figure()
plt.plot(trial_ave3d[:,n_run,:], color='gray')
plt.plot(np.mean(trial_ave3d[:,n_run,:], axis=1), color='black')

plt.figure()
for n_run in range(num_runs):
    plt.plot(np.mean(trial_ave3d[:,n_run,:], axis=1))
    
    
  
plt.figure()
plt.plot(exp_var_ta)


color_ctx = 0

f_plot_dred_pcs(comp_out3d_ta, [[0, 1]], ob_data1['red_dd_seq'], color_ctx, colors1, title_tag='svd')

f_plot_dred_pcs(X_transformed3d, [[0, 1]], ob_data1['red_dd_seq'], color_ctx, colors1, title_tag='isomap cosine')

#%% make trial ave data

trial_ave4d2, trial_data_sort, num_dd_trials = f_trial_ave_ctx_pad2(test_oddball_ctx['rates4d_cut'], trials_oddball_ctx_cut, pre_dd = pre_dd1, post_dd = post_dd1, max_trials=5, shuffle_trials=True)


num_dd_use = np.min(num_dd_trials).astype(int)

# trial_data_sort2 = trial_data_sort[:,:,:num_dd_use,:,:]


# # t(tr*num_tr), dd_trial, stim_type(run), cells
# trial_data_sort3 = np.reshape(trial_data_sort2, (trial_len*num_tr_ave, num_dd_use, num_batch, num_cells), order = 'F')


#%% do pca on trial averaged data
trial_data_sort3.shape
trial_ave3 = np.mean(trial_data_sort3, axis=1)

title_tag='trained RNN'

trial_ave3_2d = np.reshape(trial_ave3, (trial_len*num_tr_ave*num_runs, num_cells), order = 'F')


proj_data_ta, exp_var_ta, components_ta, mean_all_ta = f_run_dred(trial_ave3_2d, subtr_mean=dred_subtr_mean, method=dred_met)

# proj_data_test = np.dot(trial_ave3_2d, components_ta) # to get same thing

proj_data_ta3d = np.reshape(proj_data_ta, (trial_len*num_tr_ave, num_runs, num_cells), order = 'F')


for n_pcpl in range(len(plot_pc)):
    plot_pc2 = plot_pc[n_pcpl]
    plt.figure()

    plt.plot(proj_data_ta3d[:,:,plot_pc2[0]-1], proj_data_ta3d[:,:,plot_pc2[1]-1])

    plt.xlabel('PC%d' % plot_pc2[0]); 
    plt.ylabel('PC%d' % plot_pc2[1]) 
    plt.title('tria ave PCA components; %s' % title_tag);



proj_data_rates = np.dot(test_oddball_ctx['rates2d_cut'], components_ta)

proj_data_rates3d = np.reshape(proj_data_rates, (trial_len*num_trials_cut, num_runs, num_cells), order = 'F')

for n_pcpl in range(len(plot_pc)):
    plot_pc2 = plot_pc[n_pcpl]
    plt.figure()

    plt.plot(proj_data_rates3d[:,:,plot_pc2[0]-1], proj_data_rates3d[:,:,plot_pc2[1]-1])

    plt.xlabel('PC%d' % plot_pc2[0]); 
    plt.ylabel('PC%d' % plot_pc2[1]) 
    plt.title('tria ave proj PCA components; %s' % title_tag);



#%% dpca 

# for dpca

trial_data_sort_dpca = np.moveaxis(trial_data_sort3, [0, 1, 2, 3], [3, 0, 2, 1])

trial_ave_dpca = np.mean(trial_data_sort_dpca, axis=0)


# (n_samples,N,S,T)
dpca = dPCA(labels='st',regularizer='auto')

dpca.protect = ['t']


Z = dpca.fit_transform(trial_ave_dpca, trial_data_sort_dpca)


plt.figure(figsize=(16,7))
plt.subplot(131)

for s in range(num_dd_use):
    plt.plot(Z['t'][0,s])

plt.title('1st time component')
    
plt.subplot(132)

for s in range(num_dd_use):
    plt.plot(Z['s'][0,s])
    
plt.title('1st stimulus component')
    
plt.subplot(133)

for s in range(num_dd_use):
    plt.plot(Z['st'][0,s])
    
plt.title('1st mixing component')
plt.show()


plt.figure()
plt.plot(Z['s'][:,0,:].T, Z['s'][:,1,:].T)

plt.figure()
plt.plot(Z['st'][:,0,:].T, Z['st'][:,1,:].T)

plt.figure()
plt.plot(Z['t'][:,0,:].T, Z['t'][:,1,:].T)

tags_all = ['s', 'st', 't']
plt.figure()
for n_tag in range(3):
    tag = tags_all[n_tag]
    plt.plot(dpca.explained_variance_ratio_[tag])
plt.legend(tags_all)
plt.title('explained variance')


# D - from data to low d
# P - from low D to data

dpca.D['st'].shape
dpca.P['s'].shape


title_tag='trained RNN'



plt.figure()
plt.plot(dpca.D['st'][:,0], dpca.D['st'][:,1])

plot_data = Z

plot_pc = [[1, 2], [3, 4]]

for n_pcpl in range(len(plot_pc)):
    plot_pc2 = plot_pc[n_pcpl]
    plt.figure()
    for n_tag in range(3):
        plt.subplot(1,3,n_tag+1)
        tag = tags_all[n_tag]
        plt.plot(plot_data[tag][:,plot_pc2[0]-1,:].T, plot_data[tag][:,plot_pc2[1]-1,:].T)
        plt.title(tag)
        plt.xlabel('PC%d' % plot_pc2[0]); 
        plt.ylabel('PC%d' % plot_pc2[1]) 
    plt.suptitle('dPCA components; %s' % title_tag); 
    

proj_data_rates = np.dot(test_oddball_ctx['rates2d_cut'], components_ta)

trial_data_sort_dpca.shape

trial_data_sort_dpca2d = np.reshape(trial_data_sort_dpca, (10, 25*20*180))

proj_data_st = np.dot(trial_ave3_2d, dpca.D['t'])

proj_data_st3d = np.reshape(proj_data_st, (trial_len*num_tr_ave, num_runs, 10), order = 'F')


for n_pcpl in range(len(plot_pc)):
    plot_pc2 = plot_pc[n_pcpl]
    plt.figure()

    plt.plot(proj_data_st3d[:,:,plot_pc2[0]-1], proj_data_st3d[:,:,plot_pc2[1]-1])

    plt.xlabel('PC%d' % plot_pc2[0]); 
    plt.ylabel('PC%d' % plot_pc2[1]) 
    plt.title('dPCA st proj tria ave components; %s' % title_tag);


proj_data_rates_st = np.dot(test_oddball_ctx['rates2d_cut'], dpca.D['st'])

proj_data_rates_st3d = np.reshape(proj_data_rates_st, (trial_len*num_trials_cut, num_runs, 10), order = 'F')


for n_pcpl in range(len(plot_pc)):
    plot_pc2 = plot_pc[n_pcpl]
    plt.figure()

    plt.plot(proj_data_rates_st3d[:,:,plot_pc2[0]-1], proj_data_rates_st3d[:,:,plot_pc2[1]-1])

    plt.xlabel('PC%d' % plot_pc2[0]); 
    plt.ylabel('PC%d' % plot_pc2[1]) 
    plt.title('dPCA proj rates components; %s' % title_tag);

#%% sample rates space
#plt.close('all')

plot_trials = 100
plot_T = plot_trials*trial_len

n_tr = 0
stim_on = 1
plot_dd = 0
thresh_on = 1

plot_pc = [[1, 2], [3, 4], [5, 6]]

pc_init_type = 'mean_plus_last' # 'mean_pc', zero, mean_plus_last

vec_size_fac = 3

red_fr = ob_data1['red_dd_seq'][0, n_tr]
dd_fr = ob_data1['red_dd_seq'][1, n_tr]


if stim_on:
    st_frame = 10
    if plot_dd:
        plot_fr = dd_fr
        stim_on_tag = 'dd; stim on'
        qv_col = 'pink'
        fx_pt_col = 'red'
    else:
        plot_fr = red_fr
        stim_on_tag = 'red; stim on'
        qv_col = 'lightblue'
        fx_pt_col = 'blue'
else:
    plot_fr = dd_fr
    stim_on_tag = 'stim off'
    st_frame = 0
    qv_col = 'lightgray'
    fx_pt_col = 'black'


dred_rates_in = test_oddball_ctx['dred_rates4d']
trial_len, num_trials, num_runs, num_cells = dred_rates_in.shape
comp_plot = np.reshape(dred_rates_in, (trial_len*num_trials, num_runs, num_cells), order='F')

dred_rates_const_in = test_const_ctx['dred_rates4d']
trial_len, num_trials, num_runs, num_cells = dred_rates_const_in.shape
comp_conts_plot = np.reshape(dred_rates_const_in, (trial_len*num_trials, num_runs, num_cells), order='F')

dred_comp_use = test_oddball_ctx['dred_comp']
title_tag5 = 'trained, %s' %  (stim_on_tag)
rnn_use = rnn

# comp_plot = comp_out03d
# comp_conts_plot = comp_out0_const3d
# dred_comp_use = dred_comp0
# title_tag5 = 'untrained, stim off'
# rnn_use = rnn0

num_samp = 1000

mean_fix_pt = np.zeros((num_cells))

for n_pc in range(len(plot_pc)):
    plot_pc2 = np.asarray(plot_pc[n_pc])

    # input2 = trials_const_input[0,red_fr,:]
    # input_cut = ob_data1['input_test_oddball'][num_skip_trials*trial_len:,:,:]
    
    # plt.figure()
    # plt.imshow(trials_const_input[:,red_fr,:].T, aspect='auto')
    
    input0 = stim_templates['freq_input'][:,st_frame, plot_fr]
    input1 = torch.tensor(input0).float().to(params['device'])
    
    # rates_temp = rates3d_cut[:, n_tr,:]
    # proj_temp = np.dot(rates_temp, dred_comp_use)
    
    # plt.figure()
    # plt.plot(input0.flatten())
    # plt.plot(input2)
    
    # plt.figure()
    # plt.plot(rates_temp)
    
    plt.figure()
    
    pl1 = plt.plot(comp_plot[:plot_T, n_tr, plot_pc2[0]-1], comp_plot[:plot_T, n_tr, plot_pc2[1]-1], color='green')
    pl2 = plt.plot(comp_conts_plot[:plot_T, 0, plot_pc2[0]-1], comp_conts_plot[:plot_T, 0, plot_pc2[1]-1], color='darkgray')
    pl3 = plt.plot(comp_conts_plot[:plot_T, red_fr, plot_pc2[0]-1], comp_conts_plot[:plot_T, red_fr, plot_pc2[1]-1], color='darkgreen')
    
    plt.xlabel('PC %d' % plot_pc2[0])
    plt.ylabel('PC %d' % plot_pc2[1])
    plt.title('%s; trial %d; red freq %d; dd freq %d' % (title_tag5, n_tr, red_fr, dd_fr))
    
    
    # pc_min1 = np.min(comp_plot[:,n_tr,:], axis=0)
    # pc_max1 = np.max(comp_plot[:,n_tr,:], axis=0)
    
    # pc_min2 = np.min(comp_conts_plot[:,0,:], axis=0)
    # pc_max2 = np.max(comp_conts_plot[:,0,:], axis=0)
    
    # pc_min3 = np.min(comp_conts_plot[:,red_fr,:], axis=0)
    # pc_max3 = np.max(comp_conts_plot[:,red_fr,:], axis=0)
    
    # pc_min = np.min(np.vstack((pc_min1, pc_min2, pc_min3)), axis=0)
    # pc_max = np.max(np.vstack((pc_max1, pc_max2, pc_max3)), axis=0)
    
    
    pc_min1 = np.min(np.min(comp_plot[:,:,:], axis=0), axis=0)
    pc_max1 = np.max(np.max(comp_plot[:,:,:], axis=0), axis=0)

    pc_min3 = np.min(np.min(comp_conts_plot[:,:,:], axis=0), axis=0)
    pc_max3 = np.max(np.max(comp_conts_plot[:,:,:], axis=0), axis=0)

    pc_min = np.min(np.vstack((pc_min1, pc_min3)), axis=0)
    pc_max = np.max(np.vstack((pc_max1, pc_max3)), axis=0)
    
    pc_mean = np.mean(comp_plot[:,n_tr,:], axis=0)
    
    pc_scale = pc_max - pc_min
    pc_cent = np.mean(np.vstack((pc_min, pc_max)), axis=0)
    
    dred_inv = linalg.inv(dred_comp_use)
    
    # plt.figure()
    # plt.plot(rate_new2)
    
    # plt.figure()
    # plt.plot(rates_temp)
    
    #rate_new2 = np.zeros(rates_temp.shape)
    #rate_new2[0,:] = rates_temp[0,:]
    
    dist1 = np.ones((num_samp))*100
    
    
    if pc_init_type == 'zero':
        start_loc = np.zeros((num_samp, num_cells))
    elif pc_init_type == 'mean_pc':
        start_loc = np.ones((num_samp, num_cells)) * pc_mean.reshape((1, num_cells))
    elif pc_init_type == 'mean_plus_last':
        start_loc = np.ones((num_samp, num_cells)) * pc_mean.reshape((1, num_cells))
        start_loc[:,:plot_pc2[0]-1] = np.ones((num_samp, plot_pc2[0]-1)) * mean_fix_pt[:plot_pc2[0]-1].reshape((1, plot_pc2[0]-1))
        
    samp5 = np.random.uniform(low=0.0, high=1.0, size=(num_samp, 2))-0.5
    start_loc[:,plot_pc2-1] = np.ones((num_samp, 2)) * (samp5)*pc_scale[plot_pc2-1] + pc_cent[plot_pc2-1]
    
    out_all = np.zeros((num_samp, 2))
    
    for n_samp in range(num_samp):
        
        #start_loc = (np.random.uniform(low=0.0, high=1.0, size=25)-0.5)*pc_scale + pc_cent

        rates1 = np.dot(start_loc[n_samp,:], dred_inv)
        
        #rat111 = rnn.init_rate(50).to(params['device'])
        
        #rates11 = rates_temp[n_samp,:]
        
        rate_start = torch.tensor(rates1).float().to(params['device'])
        #rate_start = torch.tensor(rates_temp[n_samp,:]).float().to(params['device'])
        #input5 = torch.tensor(input_cut[n_samp+1,n_tr,:]).float().to(params['device'])
        input5 = torch.tensor(input1).float().to(params['device'])
        
        rates_out = rnn_use.recurrence(input5, rate_start)
        
        out_all[n_samp,:] = rnn_use.h2o_ctx(rate_start).detach().numpy()
        
        #rate_new2[n_samp+1,:] = rates_out.detach().numpy()
        
        #output, rates = rnn.forward_ctx(input1, rate_start)
        
        rates2 = rates_out.detach().numpy()
        
        dist1[n_samp] = np.mean((rates2 - rates1)**2)
        
        proj_data_const1 = np.dot(rate_start, dred_comp_use)
        proj_data_const2 = np.dot(rates2, dred_comp_use)
        
        #plt.plot([proj_data_const1[0], proj_data_const2[0]], [proj_data_const1[1], proj_data_const2[1]])
        
        plt.quiver(proj_data_const1[plot_pc2[0]-1], proj_data_const1[plot_pc2[1]-1], (proj_data_const2[plot_pc2[0]-1]-proj_data_const1[plot_pc2[0]-1]), (proj_data_const2[plot_pc2[1]-1]-proj_data_const1[plot_pc2[1]-1]), width =0.001, color=qv_col, scale = vec_size_fac)       

    low_thresh = np.percentile(dist1, 1)
    idx_low = dist1 < low_thresh
    
    start_low = start_loc[idx_low,:]
    
    mean_fix_pt[plot_pc2-1] = np.mean(start_low[:,plot_pc2-1], axis=0)
    
    # proj_data_dist1 = np.dot(start_low, dred_comp_use)
    
    idx_rd = np.argmax(out_all, axis=1).astype(bool)

    pl4 = plt.plot(start_low[:,plot_pc2[0]-1], start_low[:,plot_pc2[1]-1], 'o', color=fx_pt_col)
    
    plt.legend([pl1[0], pl2[0], pl3[0], pl4[0]], ['oddball', 'no stim', 'stim on', 'fix pt'])
    
    if thresh_on:
        plt.plot(start_loc[idx_rd,plot_pc2[0]-1], start_loc[idx_rd,plot_pc2[1]-1], '.', color='red')
        plt.plot(start_loc[~idx_rd,plot_pc2[0]-1], start_loc[~idx_rd,plot_pc2[1]-1], '.', color='blue')

#%%

st_frame = 10


rate_max = np.max(test_oddball_ctx['rates2d_cut'])
rate_min = np.min(test_oddball_ctx['rates2d_cut'])

num_samp = int(1e5)


plt.figure()

rates_samp_all = []

for n_red in range(1,50, 7):
    print('trial %d' % n_red)
    red_fr = n_red+1
    
    input0 = stim_templates['freq_input'][:,st_frame, red_fr]
    input1 = torch.tensor(input0).float().to(params['device'])
    
    dist1 = np.ones((num_samp))*100
    
    rate_samp_all = np.random.uniform(low=0.0, high=1.0, size=(num_samp, num_cells)) * (rate_max - rate_min) + rate_min
    
    #rate_samp_all2 = torch.tensor(rate_samp_all).float().to(params['device'])
    
    for n_samp in range(num_samp):
        
        rate_samp = rate_samp_all[n_samp, :]
        
        rate_samp2 = torch.tensor(rate_samp).float().to(params['device'])
        
        #rate_samp2 = rate_samp_all2[:,n_samp]
        
        rates_out = rnn.recurrence(input1, rate_samp2)
        
        rates_out2 = rates_out.detach().numpy()
        
        dist1[n_samp] = np.sqrt(np.mean((rates_out2 - rate_samp)**2))
    
    low_thresh = np.percentile(dist1, 0.005)
    
    idx_low_rate = dist1 < low_thresh
    
    rate_samp_low = rate_samp_all[idx_low_rate, :]
    
    rates_samp_all.append(rate_samp_low)
    
    proj_data_samp = np.dot(rate_samp_low, test_oddball_ctx['dred_comp'])

    
    plt.plot(proj_data_samp[:,0], proj_data_samp[:,1], 'o', color=colors1[n_red,:])
    
    
    # proj_data_samp2, exp_var_samp2, dred_comp_samp2, dred_mean_samp2 = f_run_dred(rate_samp_low, subtr_mean=dred_subtr_mean, method=dred_met)


rates_samp_all2 = np.vstack(rates_samp_all)

proj_data_samp2, exp_var_samp2, dred_comp_samp2, dred_mean_samp2 = f_run_dred(rates_samp_all2, subtr_mean=dred_subtr_mean, method=dred_met)

plt.figure()
plt.plot(proj_data_samp2[:,0], proj_data_samp2[:,1], 'o')
    


#%% plot vectors for 

rates_in = test_const_ctx['dred_rates4d']
rates_in = test0_const_ctx['dred_rates4d']

trial_len, num_trials, num_runs, num_cells = rates_in.shape
comp_plot = np.reshape(rates_in, (trial_len*num_trials, num_runs, num_cells), order='F')

plot_trials = 100
plot_T = plot_trials*trial_len

st_frame = 10

plot_pc = [[1, 2], [3, 4]] 

pc_min1 = np.min(np.min(comp_plot[:,:,:], axis=0), axis=0)
pc_max1 = np.max(np.max(comp_plot[:,:,:], axis=0), axis=0)

pc_min3 = np.min(np.min(comp_conts_plot[:,:,:], axis=0), axis=0)
pc_max3 = np.min(np.max(comp_conts_plot[:,:,:], axis=0), axis=0)

pc_min = np.min(np.vstack((pc_min1, pc_min3)), axis=0)
pc_max = np.max(np.vstack((pc_max1, pc_max3)), axis=0)

pc_mean = np.mean(np.mean(comp_plot[:,:,:], axis=0), axis=0)

pc_scale = pc_max - pc_min
pc_cent = np.mean(np.vstack((pc_min, pc_max)), axis=0)

dred_inv = linalg.inv(dred_comp_use)

for n_pc in range(len(plot_pc)):
    plot_pc2 = np.asarray(plot_pc[n_pc])
    
    
    for n_red in range(1,50, 7):
        red_fr = n_red+1
        
        input0 = stim_templates['freq_input'][:,st_frame, red_fr]
        input1 = torch.tensor(input0).float().to(params['device'])
        
        plt.figure()
        
        for n_samp in range(1000):
            
            #start_loc = (np.random.uniform(low=0.0, high=1.0, size=25)-0.5)*pc_scale + pc_cent
            
            #start_loc = np.zeros((25))
            start_loc = pc_mean.copy()
            start_loc[plot_pc2-1] = (np.random.uniform(low=0.0, high=1.0, size=2)-0.5)*pc_scale[plot_pc2-1] + pc_cent[plot_pc2-1]
            
            rates1 = np.dot(start_loc, dred_inv)
            
            #rat111 = rnn.init_rate(50).to(params['device'])
            
            #rates11 = rates_temp[n_samp,:]
            
            rate_start = torch.tensor(rates1).float().to(params['device'])
            #rate_start = torch.tensor(rates_temp[n_samp,:]).float().to(params['device'])
            #input5 = torch.tensor(input_cut[n_samp+1,n_tr,:]).float().to(params['device'])
            input5 = torch.tensor(input1).float().to(params['device'])
            
            rates_out = rnn.recurrence(input5, rate_start)
            
            #rate_new2[n_samp+1,:] = rates_out.detach().numpy()
            
            #output, rates = rnn.forward_ctx(input1, rate_start)
            
            rates2 = rates_out.detach().numpy()
            
            proj_data_const1 = np.dot(rate_start, dred_comp_use)
            proj_data_const2 = np.dot(rates2, dred_comp_use)
            
            #plt.plot([proj_data_const1[0], proj_data_const2[0]], [proj_data_const1[1], proj_data_const2[1]])
            plt.quiver(proj_data_const1[plot_pc2[0]-1], proj_data_const1[plot_pc2[1]-1], proj_data_const2[plot_pc2[0]-1]-proj_data_const1[plot_pc2[0]-1], proj_data_const2[plot_pc2[1]-1]-proj_data_const1[plot_pc2[1]-1], width =0.001, color='gray')       
        plt.title('red fr %d' % red_fr)
        plt.xlabel('PC %d' % plot_pc2[0])
        plt.ylabel('PC %d' % plot_pc2[1])
        

#%% plot speed of population trajectory
n_run = 2

rates = test_oddball_ctx['rates']
rates0 = test_oddball_ctx['rates']

f_plot_traj_speed(rates, ob_data1, n_run, start_idx=1000, title_tag= 'trained RNN; run %d' % n_run)

f_plot_traj_speed(rates0, ob_data1, n_run, start_idx=1000, title_tag= 'untrained RNN; run %d' % n_run)

#%%
f_plot_rates_only(test_oddball_ctx, 'ctx', num_plot_batches = 2, num_plot_cells = 20, preprocess = True, norm_std_fac = 6, start_from = 200*trial_len, plot_extra = 0)

f_plot_rates_only(test0_oddball_ctx, 'ctx', num_plot_batches = 2, num_plot_cells = 20, preprocess = True, norm_std_fac = 6, start_from = 200*trial_len, plot_extra = 0)



#%%  analyze distances const dd/red


choose_idx_method = 'center'  # center or sample
variab_tr_idx1 = 0;   # 1 = dd 0 = red
plot_tr_idx1 = 0

f_plot_resp_distances(test_oddball_ctx['rates4d_cut'], trials_oddball_ctx_cut, ob_data1, params, choose_idx = choose_idx_method, variab_tr_idx = variab_tr_idx1, plot_tr_idx = plot_tr_idx1, title_tag='trained RNN')

f_plot_resp_distances(test0_oddball_ctx['rates4d_cut'], trials_oddball_ctx_cut, ob_data1, params, choose_idx = choose_idx_method, variab_tr_idx = variab_tr_idx1, plot_tr_idx = plot_tr_idx1, title_tag='untrained RNN')


#%% trial average context
# plt.close('all')

f_plot_mmn(test_oddball_ctx['rates4d_cut'], trials_oddball_ctx_cut, params, title_tag='trained RNN')

f_plot_mmn(test0_oddball_ctx['rates4d_cut'], trials_oddball_ctx_cut, params, title_tag='untrained RNN')

#%% spont analysis 
# plt.close('all')

if 0:
    #spont inputs
    norm_method = 0
    start_val = 0
    
    
    
    input_shape = (trial_len*params['test_trials_in_sample'], params['test_batch_size'], params['input_size'])
    input_spont1 = np.random.normal(0,params['input_noise_std'], input_shape)
    
    test_spont = f_RNN_test_spont(rnn, input_spont1, params)
    
    f_plot_rates_only(test_spont, 'spont', num_plot_batches = 1, num_plot_cells = 25, preprocess = True, norm_std_fac = 6, start_from = 1000, plot_extra = 0)

    rates_spont = test_spont['rates']
    rates_spont_cut = rates_spont[start_val:,:,:]
    
    rates_spont3 = rates_spont
    #rates_spont3 = rates_spont_cut
    
    means1 = np.mean(rates_spont3, axis=0)
    stds1 = np.std(rates_spont3, axis=0)
    
    if not norm_method:
        rates_spont3n = rates_spont3
    elif norm_method == 1:
        rates_spont3n = rates_spont3 - means1
    elif norm_method == 2:
        stds2 = stds1.copy()
        stds2[stds1 == 0] = 1
    
        rates_spont3n = rates_spont3 - means1
        rates_spont3n = rates_spont3n/stds2
    
    T, num_bouts, num_cells = rates_spont3n.shape
    
    rates_spont3n2d = np.reshape(rates_spont3n, (T*num_bouts, num_cells), order = 'F')
    
    pca = PCA();
    pca.fit(rates_spont3n2d)
    
    proj_data = pca.fit_transform(rates_spont3n2d)
    
    comp_out3d = np.reshape(proj_data, (T, num_bouts, num_cells), order = 'F')
    
    plt.figure()
    #plt.subplot(1,2,1);
    plt.plot(pca.explained_variance_ratio_, 'o-')
    plt.ylabel('fraction')
    plt.title('Explained Variance'); plt.xlabel('component')
    
    plt.figure()
    #plt.subplot(1,2,2);
    for n_bt in range(num_bouts): #num_bouts
        plt.plot(comp_out3d[:, n_bt, 0], comp_out3d[:, n_bt, 1])
    plt.title('PCA components'); plt.xlabel('PC1'); plt.ylabel('PC2')
    
    idx1 = np.linspace(0, T-trial_len, round(T/trial_len)).astype(int)
    
    idx2 = idx1[0:25]
    
    n_bt  = 0
    
    plt.figure()
    plt.plot(comp_out3d[:T, n_bt, 0], comp_out3d[:T, n_bt, 1])
    plt.plot(comp_out3d[0, n_bt, 0], comp_out3d[0, n_bt, 1], '*')
    plt.title('PCA components; bout %d' % n_bt); plt.xlabel('PC1'); plt.ylabel('PC2')
    
    plot_T = 800
    idx3 = np.linspace(0, plot_T-trial_len, round(plot_T/trial_len)).astype(int)
    
    plt.figure()
    plt.plot(comp_out3d[:plot_T, n_bt, 0], comp_out3d[:plot_T, n_bt, 1])
    plt.plot(comp_out3d[idx3, n_bt, 0], comp_out3d[idx3, n_bt, 1], 'o')
    plt.plot(comp_out3d[0, n_bt, 0], comp_out3d[0, n_bt, 1], '*')
    plt.title('PCA components; bout %d' % n_bt); plt.xlabel('PC1'); plt.ylabel('PC2')
      
#%% dpca stuff

N,T,S = 100,250,6

# noise-level and number of trials in each condition
noise, n_samples = 0.2, 10

# build two latent factors
zt = (np.arange(T)/float(T))
zs = (np.arange(S)/float(S))

# build trial-by trial data
trialR = noise*np.random.randn(n_samples,N,S,T)
trialR += np.random.randn(N)[None,:,None,None]*zt[None,None,None,:]
trialR += np.random.randn(N)[None,:,None,None]*zs[None,None,:,None]

# trial-average data
R = np.mean(trialR,0)

# center data
R -= np.mean(R.reshape((N,-1)),1)[:,None,None]

#%%

# (n_samples,N,S,T)
dpca = dPCA(labels='st',regularizer='auto')


dpca.protect = ['t']

Z = dpca.fit_transform(R,trialR)

#%%

plt.figure(figsize=(16,7))
plt.subplot(131)

for s in range(S):
    plt.plot(time,Z['t'][0,s])

plt.title('1st time component')
    
plt.subplot(132)

for s in range(S):
    plt.plot(time,Z['s'][0,s])
    
plt.title('1st stimulus component')
    
plt.subplot(133)

for s in range(S):
    plt.plot(time,Z['st'][0,s])
    
plt.title('1st mixing component')
plt.show()


#%% working

class dred_torch(nn.Module):
    def __init__(self, data_in, k=2) -> None:
        super(dred_torch, self).__init__()
        num_row, num_col = data_in.shape
        
        self.num_col = num_col
        self.num_row = num_row
        self.k = k
        
        self.data = torch.tensor(data_in)
          
    def fit(self):
        
        L = nn.parameter.Parameter(torch.randn((self.num_row, self.k))).float()
        R = nn.parameter.Parameter(torch.randn((self.k, self.num_col))).float()
        
        optimizer = torch.optim.AdamW([L, R], lr=0.01)
        #optimizer = torch.optim.SGD(self.parameters(), lr=0.05)
        
        data = self.data
        
        num_it = round(2e3)
      
        tot_loss = torch.zeros((num_it))
        loss1_all = torch.zeros((num_it))
        loss2_all = torch.zeros((num_it))
        
        for n_it in range(num_it):
        
            optimizer.zero_grad()
        
            loss1 = (torch.sum(torch.abs((data - torch.matmul(L, R)))**1)/(self.num_col*self.num_row))
            loss2 = (torch.sum(torch.abs((torch.matmul(R, R.T) - torch.eye(self.k))))/(self.num_col**2))*1e2
            
            loss = loss1 # + loss2
            
            loss.backward()
            optimizer.step()
        
            tot_loss[n_it] = loss.item()
            loss1_all[n_it] = loss1.item()
            loss2_all[n_it] = loss2.item()
            
            print('iter %d; loss_tot = %.2f; loss1 = %.2f; loss2 = %.2f' % (n_it, loss.item(), loss1.item(), loss2.item()))
        
        self.tot_loss = tot_loss
        self.loss1_all = loss1_all
        self.loss2_all = loss2_all
        self.L_final = L.detach()
        self.R_final = R.detach()
        

#%%

dred_tr = dred_torch(test_oddball_ctx['rates2d_cut'], k=3)


dred_tr.fit()

tot_loss = dred_tr.tot_loss.numpy()


plt.figure()
plt.plot(tot_loss)

L = dred_tr.L_final.detach().numpy()
R = dred_tr.R_final.detach().numpy()


plt.figure()
plt.plot(L[:,1], L[:,2])



plt.figure()
plt.plot(proj_data[:,1], proj_data[:,2])


np.dot(R, R.T)

np.dot(L.T, L)


np.dot(proj_data[:,:3].T, proj_data[:,:3])


plt.figure()
plt.plot(L)

plt.plot(proj_data[:,:3])


#%% old stuff
#%% gen test data
# test control trials
#trials_test_cont = f_gen_cont_seq(params['num_freq_stim'], params['test_trials_in_sample'], params['test_batch_size'], 1)

if 'test_num_freq_stim' not in params.keys():
    params['test_num_freq_stim'] = 10

test_cont_stim = np.round(np.linspace(1, params['num_freq_stim'], params['test_num_freq_stim'])).astype(int)
trials_test_cont_idx = f_gen_cont_seq(params['test_num_freq_stim'], params['test_trials_in_sample'], params['test_batch_size'], 1)-1
trials_test_cont = test_cont_stim[trials_test_cont_idx]

input_test_cont, output_test_cont = f_gen_input_output_from_seq(trials_test_cont, stim_templates['freq_input'], stim_templates['freq_output'], params)

#%%
# plt.close('all')
f_plot_examle_inputs(input_test_cont, output_test_cont, params, 1)

#%% test
test_cont_freq = f_RNN_test(rnn, loss_freq, input_test_cont, output_test_cont, params, paradigm='freq')

#%%
#dev_stim = (np.array([3, 6])/10*params['num_freq_stim']).astype(int)
#red_stim = (np.array([3, 6])/10*params['num_freq_stim']).astype(int)

if 'test_num_freq_stim' not in params.keys():
    params['test_num_freq_stim'] = 10

dev_stim = ((np.arange(params['test_num_freq_stim'])+1)/params['test_num_freq_stim']*params['num_freq_stim']).astype(int)
red_stim = ((np.arange(params['test_num_freq_stim'])+1)/params['test_num_freq_stim']*params['num_freq_stim']).astype(int)


# test oddball trials
trials_test_oddball_freq, trials_test_oddball_ctx, _ = f_gen_oddball_seq(dev_stim, red_stim, params['test_trials_in_sample'], params['dd_frac'], params['num_ctx'], params['test_batch_size'], can_be_same = False)
#trials_test_oddball_freq, trials_test_oddball_ctx = f_gen_oddball_seq([5], params['test_oddball_stim'], params['test_trials_in_sample'], params['dd_frac'], params['num_ctx'], params['test_batch_size'], can_be_same = True)

input_test_oddball, output_test_oddball_freq = f_gen_input_output_from_seq(trials_test_oddball_freq, stim_templates['freq_input'], stim_templates['freq_output'], params)
_, output_test_oddball_ctx = f_gen_input_output_from_seq(trials_test_oddball_ctx, stim_templates['freq_input'], stim_templates['ctx_output'], params)

#%%
# plt.close('all')
f_plot_examle_inputs(input_test_oddball, output_test_oddball_ctx, params, 5)

f_plot_examle_inputs(input_test_oddball, output_test_oddball_freq, params, 5)

#%%
test_oddball_freq = f_RNN_test(rnn, loss_freq, input_test_oddball, output_test_oddball_freq, params, paradigm='freq')

test_oddball_ctx = f_RNN_test(rnn, loss_ctx, input_test_oddball, output_test_oddball_ctx, params, paradigm='ctx')

#%%
# plt.close('all')
f_plot_train_test_loss(train_out, test_cont_freq, test_oddball_ctx, name_tag1, name_tag2)

#%% test long controls

input_test_cont2 = input_test_cont.reshape((8000*100, 1, 50), order = 'F')
output_test_cont2 = output_test_cont.reshape((8000*100, 1, 11), order = 'F')

test_cont_freq2 = f_RNN_test(rnn, loss_freq, input_test_cont2, output_test_cont2, params, paradigm='freq')

#%% test oddball

input_test_oddball2 = input_test_oddball.reshape((8000*100, 1, 50), order = 'F')
output_test_oddball_freq2 = output_test_oddball_freq.reshape((8000*100, 1, 11), order = 'F')
output_test_oddball_ctx2 = output_test_oddball_ctx.reshape((8000*100, 1, 3), order = 'F')

test_oddball_freq2 = f_RNN_test(rnn, loss_freq, input_test_oddball2, output_test_oddball_freq2, params, paradigm='freq')

test_oddball_ctx2 = f_RNN_test(rnn, loss_ctx, input_test_oddball2, output_test_oddball_ctx2, params, paradigm='ctx')


#%% test oddball

#train_oddball = f_RNN_test(rnn, loss_ctx, input_train_oddball_freq, output_train_oddball_ctx, params)


#%% add loss of final pass to train data

T, batch_size, num_neurons = train_out['rates'].shape

output2 = torch.tensor(train_out['output'])
target2 =  torch.tensor(train_out['target_idx'])
train_out['lossT'] = np.zeros((T, batch_size))
for n_t in range(T):
    for n_bt2 in range(batch_size):
        train_out['lossT'][n_t, n_bt2] = loss_ctx(output2[n_t, n_bt2, :], target2[n_t, n_bt2].long()).item()

# plot train data

f_plot_rates2(train_out, 'train', num_plot_batches = 5)

#%%

f_plot_rates2(test_cont_freq, 'test_cont', num_plot_batches = 5)

f_plot_rates2(test_oddball_freq, 'test_oddball_freq', num_plot_batches = 5)

f_plot_rates2(test_oddball_ctx, 'test_oddball_ctx', num_plot_batches = 5)

#%%
#f_plot_rates(test_cont_freq, input_test_cont, output_test_cont, 'test cont')

#f_plot_rates(test_oddball, input_test_oddball, output_test_oddball, 'test oddball')

#%%

#f_plot_rates(test_oddball_ctx, input_test_oddball, output_test_oddball_freq, 'test oddball')

#f_plot_rates_ctx(test_oddball_ctx, input_test_oddball2, output_test_oddball_ctx2, 'test oddball')


#%% end of usefull stuff



#%% analyzing controls -- put in control data 

if 0:
    
    rates_mean = np.mean(test_oddball_ctx['rates2d_cut'], axis=0)
    
    #rates3n2dn = rates2d_cut
    rates_in = test_oddball_ctx['rates2d_cut'] - rates_mean;
    
    
    if 0:
        pca = PCA();
        pca.fit(rates_in)
        proj_data = pca.fit_transform(rates_in)
        #V2 = pca.components_
        #US = pca.fit_transform(rates_in)
        exp_var = pca.explained_variance_ratio_
    else:
        U, S, V = linalg.svd(rates_in, full_matrices=False)
        #data_back = np.dot(U * S, V)
        #US = U*S
        proj_data = U*S
        Ssq = S*S
        exp_var = Ssq / np.sum(Ssq)
    
    
    comp_out3d = np.reshape(proj_data, (trial_len*num_trials_cut, num_bouts, num_cells), order = 'F')
    
    
    plt.figure()
    #plt.subplot(1,2,1);
    plt.plot(exp_var, 'o-')
    plt.ylabel('fraction')
    plt.title('Explained Variance'); plt.xlabel('component')
    
    
    plot_patches = range(30)#[0, 1, 5]
    
    plot_T = 500; #800
    
    plot_pc = [[1, 2], [3, 4], [5, 6]]
    for n_pcpl in range(len(plot_pc)):
        plot_pc2 = plot_pc[n_pcpl]
        plt.figure()
        #plt.subplot(1,2,2);
        for n_bt in plot_patches: #num_bouts
            plt.plot(comp_out3d[:plot_T, n_bt, plot_pc2[0]-1], comp_out3d[:plot_T, n_bt, plot_pc2[1]-1])
        plt.title('PCA components'); plt.xlabel('PC%d' % plot_pc2[0]); plt.ylabel('PC%d' % plot_pc2[1])
    
    
    
    idx1 = np.linspace(0, T-trial_len, round(T/trial_len)).astype(int)
    
    idx2 = idx1[0:25]
    
    n_bt  = 0
    
    plt.figure()
    plt.plot(comp_out3d[:T, n_bt, 0], comp_out3d[:T, n_bt, 1])
    plt.plot(comp_out3d[0, n_bt, 0], comp_out3d[0, n_bt, 1], '*')
    plt.title('PCA components; bout %d' % n_bt); plt.xlabel('PC1'); plt.ylabel('PC2')
    
    plot_T = 800
    idx3 = np.linspace(0, plot_T-trial_len, round(plot_T/trial_len)).astype(int)
    
    plt.figure()
    plt.plot(comp_out3d[:plot_T, n_bt, 0], comp_out3d[:plot_T, n_bt, 1])
    plt.plot(comp_out3d[idx3, n_bt, 0], comp_out3d[idx3, n_bt, 1], 'o')
    plt.plot(comp_out3d[0, n_bt, 0], comp_out3d[0, n_bt, 1], '*')
    plt.title('PCA components; bout %d' % n_bt); plt.xlabel('PC1'); plt.ylabel('PC2')


#%% analyze control history
if 0:
    # plt.close('all')
    
    rates = test_cont_freq['rates']
    
    num_t, num_run, num_cells = rates.shape
    num_tr, num_run = trials_test_cont.shape # (400, 100)
    
    trial_len = round((params['stim_duration'] + params['isi_duration']) / params['dt'])
    
    rates4d = np.reshape(rates, (num_tr, trial_len, num_run, num_cells))
    
    
    throw_tr = 10
    
    rates4d_cut = rates4d[throw_tr:,:,:,:]
    num_tr2 = num_tr - throw_tr
    
    
    trials_test_cont_cut = trials_test_cont[throw_tr:,:]
    
    
    trial_ave_cont = np.zeros((10, trial_len, num_run, num_cells))
    
    
    for n_run in range(num_run):
        for n_tr in range(10):
            idx1 = trials_test_cont_cut[:,n_run] == (n_tr+1)
            trial_ave_cont[n_tr,:,n_run,:] = np.mean(rates4d_cut[idx1,:,n_run,:], axis=0)
    
    trial_ave_cont2 = np.mean(trial_ave_cont, axis=2)
    
    trial_ave_cont_2d = np.reshape(trial_ave_cont2, (10*trial_len, num_cells), order='F')
    
    
    plt.figure()
    plt.plot(trial_ave_cont2[0,:,:])
    
    
    plt.figure()
    plt.plot(trial_ave_cont_2d[0,:])
    
    
    input1 = trial_ave_cont_2d
    if 0:
        pca = PCA();
        pca.fit(input1)
        proj_data = pca.fit_transform(input1)
        #V2 = pca.components_
        #US = pca.fit_transform(rates3n2dn)
        exp_var = pca.explained_variance_ratio_
    else:
        U, S, V = linalg.svd(input1, full_matrices=False)
        #data_back = np.dot(U * S, V)
        #US = U*S
        proj_data = U*S
        Ssq = S*S
        exp_var = Ssq / np.sum(Ssq)
    
    
    comp_out3d = np.reshape(proj_data, (10, trial_len, num_cells), order = 'F')
    
    
    
    plt.figure()
    #plt.subplot(1,2,1);
    plt.plot(exp_var, 'o-')
    plt.ylabel('fraction')
    plt.title('Explained Variance'); plt.xlabel('component')
    
    plot_pc = [[1, 2], [3, 4], [5, 6]]
    for n_pcpl in range(len(plot_pc)):
        plt.figure()
        for n_tr in range(10):
            plot_pc2 = plot_pc[n_pcpl]
            plt.plot(comp_out3d[n_tr, :, plot_pc2[0]-1], comp_out3d[n_tr, :, plot_pc2[1]-1])
        plt.title('PCA components'); plt.xlabel('PC%d' % plot_pc2[0]); plt.ylabel('PC%d' % plot_pc2[1])
        
    
    
    plot_patches = range(30)#[0, 1, 5]
    
    plot_T = 500; #800
    
    plot_pc = [[1, 2], [3, 4], [5, 6]]
    for n_pcpl in range(len(plot_pc)):
        plot_pc2 = plot_pc[n_pcpl]
        plt.figure()
        #plt.subplot(1,2,2);
        for n_bt in plot_patches: #num_bouts
            plt.plot(comp_out3d[:plot_T, n_bt, plot_pc2[0]-1], comp_out3d[:plot_T, n_bt, plot_pc2[1]-1])
        plt.title('PCA components'); plt.xlabel('PC%d' % plot_pc2[0]); plt.ylabel('PC%d' % plot_pc2[1])
    
    
    
    variab_tr_idx = 0;   # 0 = dd 1 = red
    plot_tr_idx = 0;
    
    
    
    trial_ave_rd = np.zeros((2, trial_len, num_run, num_cells))
    
    for n_run in range(num_run):
        idx1 = trials_oddball_ctx_cut[:,n_run] == 1
        trial_ave_rd[0,:,n_run,:] = np.mean(rates4d_cut[idx1,:,n_run,:], axis=0)
        
        idx1 = trials_oddball_ctx_cut[:,n_run] == 2
        trial_ave_rd[1,:,n_run,:] = np.mean(rates4d_cut[idx1,:,n_run,:], axis=0)
    
    tr_mmn_rd = np.zeros((num_run, 2))
    for n_run in range(num_run):
        uq = np.unique(trials_test_oddball_freq[:, n_run])
        counts_uq = np.zeros((2))
        if len(uq)>1:
            for n_uq in range(2):
                counts_uq[n_uq] = np.sum(trials_test_oddball_freq[:, n_run] == uq[n_uq])
            if counts_uq[0] > counts_uq[1]:
                tr_mmn_rd[n_run, 0] = uq[0]
                tr_mmn_rd[n_run, 1] = uq[1]
            else:
                tr_mmn_rd[n_run, 0] = uq[1]
                tr_mmn_rd[n_run, 1] = uq[0]
        else:
            tr_mmn_rd[n_run, 0] = uq[0]
            tr_mmn_rd[n_run, 1] = uq[0]
                
    
    cur_tr = 5
    idx_cur = tr_mmn_rd[:,variab_tr_idx] == cur_tr
    base_resp = np.mean(trial_ave_rd[plot_tr_idx,:,idx_cur,:], axis=0)
    base_resp1d = np.reshape(base_resp, (trial_len*num_cells))
    
    
    dist_all = np.zeros((10))
    dist_all_cos = np.zeros((10))
    
    for n_tr in range(10):
        idx1 = tr_mmn_rd[:,variab_tr_idx] == (n_tr+1)
        temp1 = np.mean(trial_ave_rd[plot_tr_idx,:,idx1,:], axis=0)
        temp1_1d = np.reshape(temp1, (trial_len*num_cells))
        
        dist_all[n_tr] = pdist([base_resp1d,temp1_1d], metric='euclidean')
        
        dist_all_cos[n_tr] = pdist([base_resp1d,temp1_1d], metric='cosine')
    
    plt.figure()
    plt.plot(np.arange(10)+1, dist_all)
    plt.ylabel('euclidean dist')
    if variab_tr_idx:
        plt.title('const red, var dd; ref tr = %d' % cur_tr)
        plt.xlabel('dd stim')
    else:
        plt.title('const dd, var red; ref tr = %d' % cur_tr)
        plt.xlabel('red stim')
    
    plt.figure()
    plt.plot(np.arange(10)+1, dist_all_cos)
    plt.ylabel('cosine dist')
    plt.xlabel('red stim')
    if variab_tr_idx:
        plt.title('const red, var dd; ref tr = %d' % cur_tr)
        plt.xlabel('dd stim')
    else:
        plt.title('const dd, var red; ref tr = %d' % cur_tr)
        plt.xlabel('red stim')


#%%

f_plot_rates_only(test_oddball_ctx2, 'ctx', num_plot_batches = 1, num_plot_cells = 20, preprocess = True, norm_std_fac = 6, start_from = 1000, plot_extra = 0)

rates = test_oddball_ctx2['rates'].reshape((8000, 100, 25), order = 'F')

start_val = 0


rates2 = rates[start_val:,:,:]


#rates3n = rates
rates3n = rates2

T, num_bouts, num_cells = rates3n.shape

rates3n2d = np.reshape(rates3n, (T*num_bouts, num_cells), order = 'F')


rates_mean = np.mean(rates3n2d, axis=0)

#rates3n2dn = rates3n2d
rates3n2dn = rates3n2d - rates_mean;



# pca = PCA();
# pca.fit(rates3n2dn)
# proj_data = pca.fit_transform(rates3n2d)
# #V2 = pca.components_
# #US = pca.fit_transform(rates3n2dn)


U, S, V = linalg.svd(rates3n2dn, full_matrices=False)
#data_back = np.dot(U * S, V)
#US = U*S
proj_data = U*S

comp_out3d = np.reshape(proj_data, (T, num_bouts, num_cells), order = 'F')



plt.figure()
#plt.subplot(1,2,1);
plt.plot(pca.explained_variance_ratio_, 'o-')
plt.ylabel('fraction')
plt.title('Explained Variance'); plt.xlabel('component')


plot_patches = range(10)#[0, 1, 5]

plot_T = 800; #800

plot_pc = [[1, 2], [3, 4], [5, 6]]
for n_pcpl in range(len(plot_pc)):
    plot_pc2 = plot_pc[n_pcpl]
    plt.figure()
    #plt.subplot(1,2,2);
    for n_bt in plot_patches: #num_bouts
        plt.plot(comp_out3d[:plot_T, n_bt, plot_pc2[0]-1], comp_out3d[:plot_T, n_bt, plot_pc2[1]-1])
    plt.title('PCA components'); plt.xlabel('PC%d' % plot_pc2[0]); plt.ylabel('PC%d' % plot_pc2[1])



idx1 = np.linspace(0, T-trial_len, round(T/trial_len)).astype(int)

idx2 = idx1[0:25]

n_bt  = 0

plt.figure()
plt.plot(comp_out3d[:T, n_bt, 0], comp_out3d[:T, n_bt, 1])
plt.plot(comp_out3d[0, n_bt, 0], comp_out3d[0, n_bt, 1], '*')
plt.title('PCA components; bout %d' % n_bt); plt.xlabel('PC1'); plt.ylabel('PC2')

plot_T = 800
idx3 = np.linspace(0, plot_T-trial_len, round(plot_T/trial_len)).astype(int)

plt.figure()
plt.plot(comp_out3d[:plot_T, n_bt, 0], comp_out3d[:plot_T, n_bt, 1])
plt.plot(comp_out3d[idx3, n_bt, 0], comp_out3d[idx3, n_bt, 1], 'o')
plt.plot(comp_out3d[0, n_bt, 0], comp_out3d[0, n_bt, 1], '*')
plt.title('PCA components; bout %d' % n_bt); plt.xlabel('PC1'); plt.ylabel('PC2')

#%%
f_plot_rates_only(test_oddball_ctx2, 'ctx', num_plot_batches = 1, num_plot_cells = 20, preprocess = True, norm_std_fac = 6, start_from = 1000, plot_extra = 0)


rates5 = test_oddball_freq2['rates'].reshape((8000, 100, 25), order = 'F')
rates6 = test_cont_freq2['rates'].reshape((8000, 100, 25), order = 'F')

rates = np.concatenate((rates5, rates6), axis = 1)


start_val = 0


rates2 = rates[start_val:,:,:]


#rates3n = rates
rates3n = rates2

T, num_bouts, num_cells = rates3n.shape

rates3n2d = np.reshape(rates3n, (T*num_bouts, num_cells), order = 'F')


rates_mean = np.mean(rates3n2d, axis=0)

#rates3n2dn = rates3n2d
rates3n2dn = rates3n2d - rates_mean;



# pca = PCA();
# pca.fit(rates3n2dn)
# proj_data = pca.fit_transform(rates3n2d)
# #V2 = pca.components_
# #US = pca.fit_transform(rates3n2dn)


U, S, V = linalg.svd(rates3n2dn, full_matrices=False)
#data_back = np.dot(U * S, V)
#US = U*S
proj_data = U*S

comp_out3d = np.reshape(proj_data, (T, num_bouts, num_cells), order = 'F')



plt.figure()
#plt.subplot(1,2,1);
plt.plot(pca.explained_variance_ratio_, 'o-')
plt.ylabel('fraction')
plt.title('Explained Variance'); plt.xlabel('component')


plot_patches = [1, 2, 3, 4, 5, 101, 102, 103]# range(5)#[0, 1, 5]

plot_T = 800; #800

plot_pc = [[1, 2], [3, 4], [5, 6]]
for n_pcpl in range(len(plot_pc)):
    plot_pc2 = plot_pc[n_pcpl]
    plt.figure()
    #plt.subplot(1,2,2);
    for n_bt in plot_patches: #num_bouts
        plt.plot(comp_out3d[:plot_T, n_bt, plot_pc2[0]-1], comp_out3d[:plot_T, n_bt, plot_pc2[1]-1])
    plt.title('PCA components'); plt.xlabel('PC%d' % plot_pc2[0]); plt.ylabel('PC%d' % plot_pc2[1])



idx1 = np.linspace(0, T-trial_len, round(T/trial_len)).astype(int)

idx2 = idx1[0:25]

n_bt  = 0

plt.figure()
plt.plot(comp_out3d[:T, n_bt, 0], comp_out3d[:T, n_bt, 1])
plt.plot(comp_out3d[0, n_bt, 0], comp_out3d[0, n_bt, 1], '*')
plt.title('PCA components; bout %d' % n_bt); plt.xlabel('PC1'); plt.ylabel('PC2')

plot_T = 800
idx3 = np.linspace(0, plot_T-trial_len, round(plot_T/trial_len)).astype(int)

plt.figure()
plt.plot(comp_out3d[:plot_T, n_bt, 0], comp_out3d[:plot_T, n_bt, 1])
plt.plot(comp_out3d[idx3, n_bt, 0], comp_out3d[idx3, n_bt, 1], 'o')
plt.plot(comp_out3d[0, n_bt, 0], comp_out3d[0, n_bt, 1], '*')
plt.title('PCA components; bout %d' % n_bt); plt.xlabel('PC1'); plt.ylabel('PC2')

#%%
# plt.close('all')

w_in = np.asarray(rnn.i2h.weight.data)

w_r = np.asarray(rnn.h2h.weight.data)

w_o = np.asarray(rnn.h2o.weight.data)


plt.figure()
plt.imshow(w_in.T, aspect='auto')
plt.colorbar()
plt.title('Input')
plt.xlabel('W recurrent')
plt.ylabel('W input')

if 0:
    print('Saving RNN %s' % fname_RNN_save)
    plt.savefig(path1 + '/RNN_data/' + fname_RNN_save + 'win_fig.png', dpi=1200)

plt.figure()
plt.imshow(w_r)
plt.colorbar()
plt.title('Recurrents')
plt.xlabel('W recurrent')
plt.ylabel('W recurrent')

if 0:
    print('Saving RNN %s' % fname_RNN_save)
    plt.savefig(path1 + '/RNN_data/' + fname_RNN_save + 'wr_fig.png', dpi=1200)

plt.figure()
plt.imshow(w_o, aspect='auto')
plt.colorbar()
plt.title('Output')
plt.xlabel('W recurrent')
plt.ylabel('W output')

if 0:
    print('Saving RNN %s' % fname_RNN_save)
    plt.savefig(path1 + '/RNN_data/' + fname_RNN_save + 'wout_fig.png', dpi=1200)


#%%

idx1 = np.argmax(w_in, axis = 1)
idx2 = np.argsort(idx1)


plt.figure()
plt.imshow(w_in[idx2,:].T, aspect='auto')
plt.colorbar()
plt.title('Input')
plt.xlabel('W recurrent')
plt.ylabel('W input')

if 0:
    print('Saving RNN %s' % fname_RNN_save)
    plt.savefig(path1 + '/RNN_data/' + fname_RNN_save + 'winsort_fig.png', dpi=1200)


plt.figure()
plt.plot(w_in[:20,:].T)



#%%
#flat_dist_met = pdist(rates_all, metric='cosine');
#cs = 1- squareform(flat_dist_met);


# res_linkage = linkage(w_r, method='average')

# N = len(w_r)
# res_ord = seriation(res_linkage,N, N + N -2)

# plt.figure()
# plt.imshow(w_r[res_ord,:][:,res_ord])
# plt.colorbar()
# plt.title('Recurrents')
# plt.xlabel('W recurrent sorted')
# plt.ylabel('W recurrent sorted')

# if 0:
#     print('Saving RNN %s' % fname_RNN_save)
#     plt.savefig(path1 + '/RNN_data/' + fname_RNN_save + 'wrsort_fig.png', dpi=1200)


# #%%

# cs_ord = 1- squareform(pdist(rates_all[res_ord], metric='cosine'));

# plt.figure()
# plt.imshow(cs_ord)
# plt.title('cosine similarity sorted')



#%%

trial_ave_win = [-5,15]     # relative to stim onset time

#trial_resp_win = [5,10]     # relative to trial ave win
trial_resp_win = [5,15]     # relative to trial ave win


test_data = test_cont_freq


output_calc = test_data['target'][:,:,1:]
rates_calc = test_data['rates']
num_cells = params['hidden_size'];


T, num_batch, num_stim = output_calc.shape

num_t = trial_ave_win[1] - trial_ave_win[0]
colors1 = plt.cm.jet(np.linspace(0, 1, num_stim))
plot_t = np.asarray(range(trial_ave_win[0], trial_ave_win[1]))


stim_times = np.diff(output_calc, axis=0, prepend=0)
stim_times2 = np.greater(stim_times, 0)
on_times_all = []
num_trials_all = np.zeros((num_batch, num_stim), dtype=int)
for n_bt in range(num_batch):
    on_times2 = []
    for n_st in range(num_stim):
        on_times = np.where(stim_times2[:,n_bt,n_st])[0]
        on_times2.append(on_times)
        num_trials_all[n_bt, n_st] = len(on_times)
    on_times_all.append(on_times2)
num_trials_all2 = np.sum(num_trials_all, axis=0)

trial_all_all = []
for n_stim in range(num_stim):
    trial_all_batch = []
    for n_bt in range(num_batch):
        trial_all_cell = [] 
        for n_cell in range(num_cells):
            cell_trace = rates_calc[:,n_bt,n_cell]
            on_times = on_times_all[n_bt][n_stim]
            num_tr = num_trials_all[n_bt, n_stim]
            trial_all2 = np.zeros((num_tr, num_t))
            
            for n_tr in range(num_tr):
                trial_all2[n_tr, :] = cell_trace[(on_times[n_tr] + trial_ave_win[0]):(on_times[n_tr] + trial_ave_win[1])]
            
            trial_all_cell.append(trial_all2)
        trial_all_batch.append(trial_all_cell)
    trial_all_all.append(trial_all_batch)
        

trial_all_all2 = []

for n_stim in range(num_stim):
    temp_data = trial_all_all[n_stim]
    
    temp_data2 = np.concatenate(temp_data, axis=1)  
    trial_all_all2.append(temp_data2)
    
trial_resp_null = np.concatenate(trial_all_all2, axis=1) 
    

trial_ave_all = np.zeros((num_cells, num_stim, num_t))
trial_std_all = np.zeros((num_cells, num_stim, num_t))
trial_resp_mean_all = np.zeros((num_cells, num_stim))
trial_resp_std_all = np.zeros((num_cells, num_stim))

trial_resp_null = []

for n_stim in range(num_stim):
    
    atemp_data = trial_all_all2[n_stim]
    
    trial_resp_null_cell = []
    
    for n_cell in range(num_cells):
        
        num_tr1 = num_trials_all2[n_stim]
        
        atemp_data2 = atemp_data[n_cell,:,:]
        base = np.mean(atemp_data2[:,:-trial_ave_win[0]])
        #base = np.mean(atemp_data2[:,:-trial_ave_win[0]], axis=1).reshape((num_tr1,1))
        
        atemp_data3 = atemp_data2 - base 
        
        trial_ave2 = np.mean(atemp_data3, axis=0)
        trial_std2 = np.std(atemp_data3, axis=0)
        trial_resp = np.mean(atemp_data3[:,trial_resp_win[0]:trial_resp_win[1]], axis=1)
        
        
        trial_resp_null_cell.append(trial_resp)

        trial_resp_mean2 = np.mean(trial_resp)
        trial_resp_std2 = np.std(trial_resp)
        
        trial_ave_all[n_cell, n_stim, :] = trial_ave2
        trial_std_all[n_cell, n_stim, :] = trial_std2
        
        trial_resp_mean_all[n_cell, n_stim] = trial_resp_mean2
        trial_resp_std_all[n_cell, n_stim] = trial_resp_std2

    trial_resp_null.append(trial_resp_null_cell)

trial_resp_null2 = np.concatenate(trial_resp_null, axis=1)

cell_resp_null_mean = np.mean(trial_resp_null2, axis=1)
cell_resp_null_std = np.std(trial_resp_null2, axis=1)



num_trials_mean = round(np.mean(num_trials_all2))
trial_resp_z_all = (trial_resp_mean_all - cell_resp_null_mean.reshape((num_cells,1)))/(cell_resp_null_std.reshape((num_cells,1))/np.sqrt(num_trials_mean-1))


trial_max_idx = np.argmax(trial_resp_z_all, axis=1)
idx1_sort = trial_max_idx.argsort()
trial_resp_z_all_sort = trial_resp_z_all[idx1_sort,:]

max_resp = np.max(trial_resp_z_all, axis=1)
idx1_sort = (-max_resp).argsort()
trial_resp_z_all_sort_mag = trial_resp_z_all[idx1_sort,:]


if 0:
    plt.figure()
    plt.imshow(trial_resp_mean_all, aspect="auto")
    
    plt.figure()
    plt.imshow(trial_resp_z_all_sort, aspect="auto")
    
    
    
    plt.figure()
    plt.imshow(trial_resp_z_all_sort_mag, aspect="auto")
    
    
    
    
    plt.figure()
    plt.imshow(trial_resp_z_all_sort_mag, aspect="auto")
    
    
    np.mean(max_resp>3)
    
    
    
    n_cell = 0
    
    stim_x = np.arange(num_stim)+1
    
    
    resp_tr = trial_resp_mean_all[n_cell,:] - cell_resp_null_mean[n_cell]
    resp_tr_err = trial_resp_std_all[n_cell,:]/np.sqrt(num_trials_mean-1)
    
    mean_tr = np.zeros((num_stim))
    mean_tr_err = np.ones((num_stim))*cell_resp_null_std[n_cell]/np.sqrt(num_trials_mean-1)
    
    plt.figure()
    #plt.plot(trial_resp_mean_all[n_cell,:] - cell_resp_null_mean[n_cell])
    plt.errorbar(stim_x, resp_tr, yerr=resp_tr_err)
    plt.errorbar(stim_x, mean_tr, yerr=mean_tr_err)
    
    
    trial_resp_z = (trial_resp_mean_all[n_cell,:] - cell_resp_null_mean[n_cell])/(cell_resp_null_std[n_cell]/np.sqrt(num_trials_mean-1))
    
    
    
    
    plt.figure()
    plt.plot(trial_resp_z_all_sort_mag[0])
    
    
    
    
    
    plt.figure()
    plt.plot(trial_ave_all[3,:,:].T)
    
    trial_resp_all = np.mean(trial_ave_all[:,:,trial_resp_win[0]:trial_resp_win[1]], axis=2)
    
    
    trial_max_idx = np.argmax(trial_resp_all, axis=1)
    
    idx1_sort = trial_max_idx.argsort()
    
    trial_resp_all_sort = trial_resp_all[idx1_sort,:]
    
    plt.figure()
    plt.imshow(trial_resp_all_sort, aspect="auto")
    
    plt.figure()
    for n_st in range(num_stim):
        pop_ave = trial_resp_all[trial_max_idx == n_st, :].mean(axis=0)
        x_lab = np.arange(num_stim) - n_st
    
        plt.plot(x_lab, pop_ave)



#%% analyze tuning of oddball


# plt.close('all')

num_cells = params['hidden_size'];

test_data_ob_freq = test_oddball_freq
test_data_ob_ctx = test_oddball_ctx

output_freq = test_data_ob_freq['target'][:,:,1:]
output_ctx = test_data_ob_ctx['target'][:,:,1:]
rates_calc = test_data_ob_ctx['rates']



T, num_batch, num_stim = output_freq.shape
num_ctx = output_ctx.shape[2]

num_t = trial_ave_win[1] - trial_ave_win[0]
colors1 = plt.cm.jet(np.linspace(0, 1, num_stim))
plot_t = np.asarray(range(trial_ave_win[0], trial_ave_win[1]))


stim_times_freq = np.diff(output_freq, axis=0, prepend=0)
stim_times_freq2 = np.greater(stim_times_freq, 0)

stim_times_ctx = np.diff(output_ctx, axis=0, prepend=0)
stim_times_ctx2 = np.greater(stim_times_ctx, 0)
on_times_all_ctx = []
num_trials_all_ctx = np.zeros((num_batch, num_ctx), dtype=int)
stim_type_rd = np.zeros((num_batch, num_ctx), dtype=int)
for n_bt in range(num_batch):
    on_times2 = []
    for n_st in range(num_ctx):
        on_times = np.where(stim_times_ctx2[:,n_bt,n_st])[0]
        on_times2.append(on_times)
        num_trials_all_ctx[n_bt, n_st] = len(on_times)
    
    on_times_all_ctx.append(on_times2)
    
    stim_times_freq3 = stim_times_freq2[:,n_bt,:]
    num_trials5 = np.sum(stim_times_freq3, axis=0)
    
    stim_type_rd[n_bt,:] = (-num_trials5).argsort()[:2]



trial_all_ctx = []
for n_ctx in range(num_ctx):
    trial_all_stim = []
    for n_stim in range(num_stim):
        trial_all_batch = []
        for n_bt in range(num_batch):
            
            stim1 = stim_type_rd[n_bt, n_ctx]
            trial_all_cell = []
            
            if stim1 == n_stim:
                for n_cell in range(num_cells):
                    cell_trace = rates_calc[:,n_bt,n_cell]
                    on_times = on_times_all_ctx[n_bt][n_ctx]
                    num_tr = num_trials_all_ctx[n_bt, n_ctx]
                    trial_all2 = np.zeros((num_tr, num_t))
                    
                    for n_tr in range(num_tr):
                        trial_all2[n_tr, :] = cell_trace[(on_times[n_tr] + trial_ave_win[0]):(on_times[n_tr] + trial_ave_win[1])]
                    
                    trial_all_cell.append(trial_all2) 
            trial_all_batch.append(trial_all_cell)
        trial_all_stim.append(trial_all_batch)
    trial_all_ctx.append(trial_all_stim)
        


trial_ave_ctx_crd = np.zeros((3, num_stim, num_cells, num_t))

for n_st in range(num_stim):
    
    
    temp_data_fr = trial_all_all[n_st]
    temp_data_fr3 = np.concatenate(temp_data_fr, axis=1)
    
    trial_ave_fr5 = np.mean(temp_data_fr3, axis=1)
    
    trial_ave_ctx_crd[0,:,:] = trial_ave_fr5
    
    for n_ctx in range(num_ctx):
        temp_data = trial_all_ctx[n_ctx][n_st][:]
        temp_data2 = []
        for n_bt in range(num_batch):
            if len(temp_data[n_bt]):
                temp_data2.append(temp_data[n_bt])
        
        
        # cells, trials, T
        temp_data3 = np.concatenate(temp_data2, axis=1)
        
        trial_ave5 = np.mean(temp_data3, axis=1)
        trial_ave_ctx_crd[n_ctx+1, n_st,:,:] = trial_ave5
        

plot_t = np.arange(num_t)+trial_ave_win[0]

pop_ave = np.mean(np.mean(trial_ave_ctx_crd, axis=1), axis=1)

pop_base = np.mean(pop_ave[:,:-trial_ave_win[0]],axis=1).reshape((3,1))

pop_ave_n = pop_ave - pop_base;

plt.figure()
plt.plot(plot_t, pop_ave_n[0,:], color='black')
plt.plot(plot_t, pop_ave_n[1,:], color='blue') 
plt.plot(plot_t, pop_ave_n[2,:], color='red')
plt.title('ave across all stim')
plt.ylim([-0.013, 0.023])


# trial_ave_all = np.zeros((num_cells, num_stim, num_t))
# trial_std_all = np.zeros((num_cells, num_stim, num_t))
# trial_resp_mean_all = np.zeros((num_cells, num_stim))
# trial_resp_std_all = np.zeros((num_cells, num_stim))
# cell_resp_null_mean = np.zeros((num_cells))
# cell_resp_null_std = np.zeros((num_cells))

    
        
#         trial_all3 = np.concatenate(trial_all_batch, axis=0)    

                 
        
#         trial_ave2 = np.mean(trial_all3, axis=0)
#         base = np.mean(trial_ave2[:-trial_ave_win[0]])
        
#         trial_all4 = trial_all3 - base   
        
#         trial_all_stim.append(trial_all4)
        
#         trial_std2 = np.std(trial_all4, axis=0)
        
#         trial_resp = np.mean(trial_all4[:,trial_resp_win[0]:trial_resp_win[1]], axis=1)
        
#         trial_resp_null.append(trial_resp)
        
#         trial_resp_mean2 = np.mean(trial_resp)
#         trial_resp_std2 = np.std(trial_resp)
        
#         trial_ave_all[n_cell, n_stim, :] = trial_ave2 - base
#         trial_std_all[n_cell, n_stim, :] = trial_std2
#         trial_resp_mean_all[n_cell, n_stim] = trial_resp_mean2
#         trial_resp_std_all[n_cell, n_stim] = trial_resp_std2
        
#     cell_null = np.concatenate(trial_resp_null, axis=0)   
        
#     cell_resp_null_mean[n_cell] = np.mean(cell_null)
#     cell_resp_null_std[n_cell] = np.std(cell_null)
  
#     trial_all_all.append(trial_all_stim)




# plt.figure()
# plt.imshow(output_calc[:,0,:].T, aspect='auto')

# plt.figure()
# plt.plot(stim_times2[:,0,:])

# plt.figure()
# plt.plot(test_data['rates'][:,0,0])
# plt.plot(test_data_ctx['rates'][:,0,0])





# #%%
# pca = PCA();
# pca.fit(rates_all[:,:,0])

# #%%
# plt.figure()
# plt.subplot(1,2,1);
# plt.plot(pca.explained_variance_, 'o')
# plt.title('Explained Variance'); plt.xlabel('component')

# plt.subplot(1,2,2);
# plt.plot(pca.components_[0,:], pca.components_[1,:])
# plt.title('PCA components'); plt.xlabel('PC1'); plt.ylabel('PC2')

# #%%
# flat_dist_met = pdist(rates_all[:,:,0], metric='cosine');
# cs = 1- squareform(flat_dist_met);
# res_linkage = linkage(flat_dist_met, method='average')

# N = len(cs)
# res_ord = seriation(res_linkage,N, N + N -2)
    
# #%%

# cs_ord = 1- squareform(pdist(rates_all[res_ord,:,0], metric='cosine'));

# plt.figure()
# plt.imshow(cs_ord)
# plt.title('cosine similarity sorted')

# #%% cell tuning

# #%% save data

# data_save = {"rates_all": rates_all, "loss_all_smooth": loss_all_smooth,
#              "input_sig": np.asarray(input_sig.data), "target": np.asarray(target.data),
#              "output": outputs_all, "g": g, "dt": dt, "tau": tau, "hidden_size": hidden_size,
#              'ti': ti,
#              'h2h_weight': np.asarray(rnn.h2h.weight.data), 'train_RNN': train_RNN,
#              'i2h_weight': np.asarray(rnn.i2h.weight.data),
#              'h2o_weight': np.asarray(rnn.h2o.weight.data),
#              'fname_input': fname_input}

# #save_fname = 'rnn_out_8_25_21_1_complex_g_tau10_5cycles.mat'

# save_fname = 'rnn_out_12_31_21_10tones_200reps_notrain.mat'

# savemat(fpath+ save_fname, data_save)



#%%

# dim = 256;

# radius = 4
# pat1 = np.zeros((radius*2+1,radius*2+1));

# for n_m in range(radius*2+1):
#     for n_n in range(radius*2+1):
#         if np.sqrt((radius-n_m)**2 + (radius-n_n)**2)<radius:
#             pat1[n_m,n_n] = 1;
        

# plt.figure()
# plt.imshow(pat1)

# coords = np.round(np.random.uniform(low=0.0, high=(dim-1), size=(hidden_size,2)))

# frame1 = np.zeros((dim,dim, hidden_size))

# for n_frame in range(hidden_size):
#     temp_frame = frame1[:,:,n_frame]
#     temp_frame[int(coords[n_frame,0]), int(coords[n_frame,1])] = 1;
#     temp_frame2 = signal.convolve2d(temp_frame, pat1, mode='same')
#     frame1[:,:,n_frame] = temp_frame2

# plt.figure()
# plt.imshow(frame1[:,:,1])



#%%% make movie if want

# rates_all2 = rates_all - np.min(rates_all)
# rates_all2 = rates_all2/np.max(rates_all2)

# frame2 = frame1.reshape(256*256,250)

# frame2 = np.dot(frame2, rates_all2).T

# frame2 = frame2.reshape(10000,256,256)

# skimage.io.imsave('test2.tif', frame2)






#%%

# plot_cells = np.sort(sample(range(hidden_size), num_plots));

# spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[4, 1])

# plt.figure()
# ax1 = plt.subplot(spec[0])
# for n_plt in range(num_plots):  
#     shift = n_plt*2.5    
#     ax1.plot(rates_all[plot_cells[n_plt],output_mat[1,:],-1]+shift)
# plt.title('example cells')
# plt.subplot(spec[1], sharex=ax1)
# plt.plot(output_mat[1,output_mat[1,:]]) # , aspect=100
# plt.title('target')

#%%


    

#%% testing

# print(np.std(np.asarray(i2h.weight.data).flatten()))
# print(np.std(np.asarray(h2h.weight.data).flatten()))

# print(np.mean(np.asarray(i2h(input_sig[:,n_t]).data).flatten()))
# print(np.std(np.asarray(i2h(input_sig[:,n_t]).data).flatten()))

# x1 = rate.data;
# x1 = i2h(input_sig[:,n_t]).data;
# print(np.mean(np.asarray(x1).flatten()))
# print(np.std(np.asarray(x1).flatten()))

# for n_cyc in range(num_cycles):
    
#     print('cycle ' + str(n_cyc+1) + ' of ' + str(num_cycles))
    
#     for n_t in range(T-1):
        
#         if train_RNN:
#             optimizer.zero_grad()
        
#         output, rate_new = rnn.forward(input_sig[:,n_t], rate)
        
#         rates_all[:,n_t+1,n_cyc] = rate_new.detach().numpy()[0,:];
        
#         rate = rate_new.detach();
    
#         outputs_all[:,n_t+1,n_cyc] = output.detach().numpy()[0,:];
        
#         target2 = torch.argmax(target[:,n_t]) * torch.ones(1) # torch.tensor()
#         loss2 = loss(output, target2.long())
        
#         if train_RNN:
#             loss2.backward() # retain_graph=True
#             optimizer.step()
            
#         loss_all[n_t,n_cyc] = loss2.item()
#         loss_all_all.append(loss2.item())
#         iteration1.append(iteration1[-1]+1);

# print('Done')




#%% analyze tuning of controls (old version)


# plt.close('all')

# trial_len = stim_templates['freq_input'].shape[1]


# trial_ave_win = [-5,15]     # relative to stim onset time

# #trial_resp_win = [5,10]     # relative to trial ave win
# trial_resp_win = [5,15]     # relative to trial ave win


# test_data = test_cont_freq


# output_calc = test_data['target'][:,:,1:]
# rates_calc = test_data['rates']
# num_cells = params['hidden_size'];


# T, num_batch, num_stim = output_calc.shape

# num_t = trial_ave_win[1] - trial_ave_win[0]
# colors1 = plt.cm.jet(np.linspace(0, 1, num_stim))
# plot_t = np.asarray(range(trial_ave_win[0], trial_ave_win[1]))


# stim_times = np.diff(output_calc, axis=0, prepend=0)
# stim_times2 = np.greater(stim_times, 0)
# on_times_all = []
# num_trials_all = np.zeros((num_batch, num_stim), dtype=int)
# for n_bt in range(num_batch):
#     on_times2 = []
#     for n_st in range(num_stim):
#         on_times = np.where(stim_times2[:,n_bt,n_st])[0]
#         on_times2.append(on_times)
#         num_trials_all[n_bt, n_st] = len(on_times)
#     on_times_all.append(on_times2)


# trial_all_all = []
# trial_ave_all = np.zeros((num_cells, num_stim, num_t))
# trial_std_all = np.zeros((num_cells, num_stim, num_t))
# trial_resp_mean_all = np.zeros((num_cells, num_stim))
# trial_resp_std_all = np.zeros((num_cells, num_stim))
# cell_resp_null_mean = np.zeros((num_cells))
# cell_resp_null_std = np.zeros((num_cells))
# for n_cell in range(num_cells):
#     trial_all_stim = []
    
#     trial_resp_null = []
    
#     for n_stim in range(num_stim):
        
#         trial_all_batch = []
    
#         for n_bt in range(num_batch):

#             cell_trace = rates_calc[:,n_bt,n_cell]

#             on_times = on_times_all[n_bt][n_stim]
            
#             num_tr = num_trials_all[n_bt, n_stim]
            
#             trial_all2 = np.zeros((num_tr, num_t))
            
#             for n_tr in range(num_tr):
#                 trial_all2[n_tr, :] = cell_trace[(on_times[n_tr] + trial_ave_win[0]):(on_times[n_tr] + trial_ave_win[1])]
            
#             trial_all_batch.append(trial_all2)
            
#         trial_all3 = np.concatenate(trial_all_batch, axis=0)    

#         trial_ave2 = np.mean(trial_all3, axis=0)
#         base = np.mean(trial_ave2[:-trial_ave_win[0]])
        
#         trial_all4 = trial_all3 - base   
        
#         trial_all_stim.append(trial_all3)
#         #trial_all_stim.append(trial_all4)
        
#         trial_std2 = np.std(trial_all4, axis=0)
        
#         trial_resp = np.mean(trial_all4[:,trial_resp_win[0]:trial_resp_win[1]], axis=1)
        
#         trial_resp_null.append(trial_resp)
        
#         trial_resp_mean2 = np.mean(trial_resp)
#         trial_resp_std2 = np.std(trial_resp)
        
#         trial_ave_all[n_cell, n_stim, :] = trial_ave2 - base
#         trial_std_all[n_cell, n_stim, :] = trial_std2
#         trial_resp_mean_all[n_cell, n_stim] = trial_resp_mean2
#         trial_resp_std_all[n_cell, n_stim] = trial_resp_std2
        
#     cell_null = np.concatenate(trial_resp_null, axis=0)   
        
#     cell_resp_null_mean[n_cell] = np.mean(cell_null)
#     cell_resp_null_std[n_cell] = np.std(cell_null)
  
#     trial_all_all.append(trial_all_stim)



# #trial_resp_mean_all
# #trial_resp_std_all
# #cell_resp_null_mean
# #cell_resp_null_std



# num_trials_all2 = np.sum(num_trials_all, axis=0)



# num_trials_mean = round(np.mean(num_trials_all2))
# trial_resp_z_all = (trial_resp_mean_all - cell_resp_null_mean.reshape((num_cells,1)))/(cell_resp_null_std.reshape((num_cells,1))/np.sqrt(num_trials_mean-1))


# trial_max_idx = np.argmax(trial_resp_z_all, axis=1)
# idx1_sort = trial_max_idx.argsort()
# trial_resp_z_all_sort = trial_resp_z_all[idx1_sort,:]

# max_resp = np.max(trial_resp_z_all, axis=1)
# idx1_sort = (-max_resp).argsort()
# trial_resp_z_all_sort_mag = trial_resp_z_all[idx1_sort,:]



# plt.figure()
# plt.imshow(trial_resp_mean_all, aspect="auto")

# plt.figure()
# plt.imshow(trial_resp_z_all_sort, aspect="auto")



# plt.figure()
# plt.imshow(trial_resp_z_all_sort_mag, aspect="auto")


# np.mean(max_resp>3)



# n_cell = 0

# stim_x = np.arange(num_stim)+1


# resp_tr = trial_resp_mean_all[n_cell,:] - cell_resp_null_mean[n_cell]
# resp_tr_err = trial_resp_std_all[n_cell,:]/np.sqrt(num_trials_mean-1)

# mean_tr = np.zeros((num_stim))
# mean_tr_err = np.ones((num_stim))*cell_resp_null_std[n_cell]/np.sqrt(num_trials_mean-1)

# plt.figure()
# #plt.plot(trial_resp_mean_all[n_cell,:] - cell_resp_null_mean[n_cell])
# plt.errorbar(stim_x, resp_tr, yerr=resp_tr_err)
# plt.errorbar(stim_x, mean_tr, yerr=mean_tr_err)


# trial_resp_z = (trial_resp_mean_all[n_cell,:] - cell_resp_null_mean[n_cell])/(cell_resp_null_std[n_cell]/np.sqrt(num_trials_mean-1))




# plt.figure()
# plt.plot(trial_resp_z_all_sort_mag[0])





# plt.figure()
# plt.plot(trial_ave_all[3,:,:].T)


# trial_resp_all = np.mean(trial_ave_all[:,:,trial_resp_win[0]:trial_resp_win[1]], axis=2)


# trial_max_idx = np.argmax(trial_resp_all, axis=1)

# idx1_sort = trial_max_idx.argsort()

# trial_resp_all_sort = trial_resp_all[idx1_sort,:]

# plt.figure()
# plt.imshow(trial_resp_all_sort, aspect="auto")

# plt.figure()
# for n_st in range(num_stim):
#     pop_ave = trial_resp_all[trial_max_idx == n_st, :].mean(axis=0)
#     x_lab = np.arange(num_stim) - n_st

#     plt.plot(x_lab, pop_ave)






