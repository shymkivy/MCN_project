# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:08:02 2024

@author: ys2605
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:01:34 2023

@author: ys2605
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
sys.path.append(path1 + 'RNN_scripts');


from f_analysis import f_plot_rates2, f_plot_rates_only # seriation, 
from f_RNN import f_RNN_test, f_RNN_test_spont, f_gen_ob_dset, f_gen_cont_dset #, f_trial_ave_pad, f_gen_equal_freq_space
from f_RNN_process import f_trial_ave_ctx_pad, f_trial_ave_ctx_pad2, f_trial_sort_data_pad, f_trial_sort_data_ctx_pad, f_label_redundants, f_get_rdc_trav, f_gather_dev_trials, f_gather_red_trials, f_analyze_trial_vectors, f_analyze_cont_trial_vectors, f_get_trace_tau # , f_euc_dist, f_cos_sim
from f_RNN_dred import f_run_dred, f_run_dred_wrap, f_proj_onto_dred
from f_RNN_plots import f_plot_dred_rates, f_plot_dred_rates2, f_plot_dred_rates3, f_plot_dred_rates3d, f_plot_traj_speed, f_plot_resp_distances, f_plot_mmn, f_plot_mmn2, f_plot_mmn_dist, f_plot_mmn_freq, f_plot_dred_pcs, f_plot_rnn_weights2, f_plot_run_dist, f_plot_cont_vec_data, f_plot_rd_vec_data, f_plot_ctx_vec_data, f_plot_ctx_vec_dir, f_plot_cat_data # , f_plot_shadederrorbar
from f_RNN_chaotic import RNN_chaotic
from f_RNN_utils import f_gen_stim_output_templates, f_gen_cont_seq, f_gen_oddball_seq, f_gen_input_output_from_seq, f_plot_examle_inputs, f_plot_train_loss, f_plot_train_test_loss, f_gen_name_tag, f_cut_reshape_rates_wrap, f_plot_exp_var, f_plot_freq_space_distances_control, f_plot_freq_space_distances_oddball # , f_reshape_rates
from f_RNN_decoder import f_make_cv_groups, f_sample_trial_data_dec, f_run_binwise_dec, f_plot_binwise_dec, f_run_one_shot_dec, f_plot_one_shot_dec_bycat, f_plot_one_shot_dec_bycat2, f_plot_one_shot_dec_iscat

import time
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import gridspec
#from matplotlib import colors
import matplotlib.cm as cm
#from random import sample, random
import torch
import torch.nn as nn

from sklearn.decomposition import PCA
#from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import Isomap
#from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform, cdist #
from scipy.signal import correlate
#from scipy.sparse import diags
#from scipy import signal
from scipy import linalg
#from scipy.io import loadmat, savemat
#import skimage.io

from datetime import datetime



#%%

# rnn_flist = ['oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_20trials_50stim_100batch_1e-03lr_2023_9_11_14h_19m_RNN',
#              'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_2023_10_4_17h_16m_RNN',
#              'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_2023_10_5_10h_54m_RNN',
#              'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_2023_10_6_17h_3m_RNN']

rnn_flist = [
             'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise0_2023_9_11_14h_19m_ext_2024_3_6_16h_56m_RNN',
             #'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_2023_9_11_14h_19m_RNN',
             'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise0_2023_10_4_17h_16m_ext_2024_3_7_12h_9m_RNN',
             #'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_2023_10_4_17h_16m_RNN',
             'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise0_2023_10_5_10h_54m_ext_2024_3_8_11h_57m_RNN',
             #'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_2023_10_5_10h_54m_RNN',
             'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_2023_10_6_17h_3m_RNN',
             'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2023_12_19_16h_20m_RNN',
             'oddball2_1ctx_80000trainsamp_250neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_1_5_13h_15m_RNN', # 250 neurons
             #'oddball2_1ctx_120000trainsamp_250neurons_ReLU_100tau_10dt_20trials_50stim_100batch_4e-05lr_noise1_2024_1_22_12h_49m_RNN', # not long enough? 250 neurons dt 10 cant use different dt
             #'oddball2_1ctx_120000trainsamp_100neurons_ReLU_100tau_10dt_20trials_50stim_100batch_4e-05lr_noise1_2024_2_1_11h_45m_RNN', # not long enough? 100 neurons # dt 10
             #'oddball2_1ctx_180000trainsamp_100neurons_ReLU_100tau_10dt_20trials_50stim_100batch_4e-05lr_noise1_2024_2_4_20h_33m_RNN', # not long enough? 100 neurons # dt 10
             'oddball2_1ctx_140000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_2_19_10h_15m_RNN',
             'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_2_24_13h_23m_RNN', #didn't reach low enough loss
             'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_2_28_11h_32m_RNN', #didn't reach low enough loss
             'oddball2_1ctx_250000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_3_3_13h_54m_RNN', #didn't reach low enough loss
             'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-03lr_noise1_2024_3_6_16h_3m_RNN',
             'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_3_7_12h_8m_RNN',
             'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_2e-03lr_noise1_2024_3_8_11h_57m_RNN',
             'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_2e-03lr_noise1_2024_3_10_4h_38m_RNN',
             'oddball2_1ctx_200000trainsamp_50neurons_ReLU_500tau_50dt_20trials_50stim_100batch_2e-03lr_noise1_2024_3_10_21h_27m_RNN',
             'oddball2_1ctx_200000trainsamp_100neurons_ReLU_500tau_50dt_20trials_50stim_100batch_2e-03lr_noise1_2024_3_11_14h_58m_RNN',
             'oddball2_1ctx_200000trainsamp_75neurons_ReLU_500tau_50dt_20trials_50stim_100batch_2e-03lr_noise1_2024_3_11_16h_47m_RNN',
             ]

rnnf_flist = ['freq2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise0_2023_12_20_0h_34m_RNN',
              'freq2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2023_12_20_0h_34m_RNN',
              'freq2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_1_4_13h_14m_RNN', # not long enough?
              #'freq2_1ctx_80000trainsamp_250neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-04lr_noise1_2024_1_10_11h_28m_RNN', # bit spiky
              #'freq2_1ctx_120000trainsamp_250neurons_ReLU_500tau_50dt_20trials_50stim_100batch_4e-05lr_noise1_2024_1_11_11h_33m_RNN', # bit spiky
              'freq2_1ctx_160000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_2_20_19h_9m_RNN', # bit spiky
              'freq2_1ctx_250000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_2_22_16h_20m_RNN',
              'freq2_1ctx_300000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_2_27_13h_17m_RNN',
              'freq2_1ctx_250000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_3_1_10h_1m_RNN',
              'freq2_1ctx_300000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_3_4_19h_25m_RNN',
              'freq2_1ctx_300000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_3_5_0h_28m_RNN',
              ]

num_rnn = len(rnn_flist)
num_rnnf = len(rnnf_flist)

num_rnn0 = num_rnn

#%% plot loss all

loss_all = []
params_ob_all = []
for n_rnn in range(num_rnn):
    train_out = np.load(path1 + '/RNN_data/' + rnn_flist[n_rnn][:-4] + '_train_out.npy', allow_pickle=True).item()
    loss_all.append(train_out['loss'])
    
    params = np.load(path1 + '/RNN_data/' + rnn_flist[n_rnn][:-4] + '_params.npy', allow_pickle=True).item()
    if 'train_add_noise' not in params.keys():
        params['train_add_noise'] = 0
    params_ob_all.append(params)
    
lossf_all = []
paramsf_all = []
for n_rnn in range(num_rnnf):
    train_out = np.load(path1 + '/RNN_data/' + rnnf_flist[n_rnn][:-4] + '_train_out.npy', allow_pickle=True).item()
    lossf_all.append(train_out['loss'])
    
    params = np.load(path1 + '/RNN_data/' + rnnf_flist[n_rnn][:-4] + '_params.npy', allow_pickle=True).item()
    if 'train_add_noise' not in params.keys():
        params['train_add_noise'] = 0
    paramsf_all.append(params)


#%%
colors1 = cm.jet(np.linspace(0,1,num_rnn))
colors2 = cm.jet(np.linspace(0,1,num_rnnf))

#%%
# plt.close('all')
sm_bin = 100#round(1/params['dt'])*50;
kernel = np.ones(sm_bin)/sm_bin

plt.figure()
leg1 = []
for n_rnn in range(len(loss_all)):
    loss1 = np.asarray(loss_all[n_rnn])
    
    if sm_bin:
        loss1_sm = np.convolve(loss1, kernel, mode='valid')
    else:
        loss1_sm = loss1

    loss_x_sm = np.arange(len(loss1_sm))+sm_bin/2 #/(trial_len)
    plt.semilogy(loss_x_sm, loss1_sm, color=colors1[n_rnn])
    
    leg1.append('RNN%d, n=%d, lr=%.1e' % (n_rnn+1, params_ob_all[n_rnn]['hidden_size'], params_ob_all[n_rnn]['learning_rate']))
    
plt.legend(leg1)
plt.title('oddball trained RNNs, sm %d' % sm_bin)


#%%
sm_bin = 0#round(1/params['dt'])*50;
kernel = np.ones(sm_bin)/sm_bin

plt.figure()
leg1 = []
for n_rnn in range(len(lossf_all)):
    loss1 = np.asarray(lossf_all[n_rnn])
    
    if sm_bin:
        loss1_sm = np.convolve(loss1, kernel, mode='valid')
    else:
        loss1_sm = loss1
    
    loss_x_sm = np.arange(len(loss1_sm))+sm_bin/2 #/(trial_len)
    plt.semilogy(loss_x_sm, loss1_sm, color=colors2[n_rnn])
    
    leg1.append('RNN%d, n=%d, lr=%.1e' % (n_rnn+1, paramsf_all[n_rnn]['hidden_size'], paramsf_all[n_rnn]['learning_rate']))
    
plt.legend(leg1)
plt.title('freq trained RNNs; sm %d' % sm_bin)

#%%
stim_templates = f_gen_stim_output_templates(params)
trial_len = round((params['stim_duration'] + params['isi_duration'])/params['dt'])

#%% create test inputs
dred_subtr_mean = 0
dred_met = 2
num_skip_trials = 90

num_prepend_zeros = 100


num_dev_stim = 20       # 20
num_red_stim = 20       # 20
num_cont_stim = 50


num_ob_runs = 20
num_ob_trials = 1000

num_cont_runs = 20
num_cont_trials = 1000

num_const_stim = 50


# oddball data
params_ob = params.copy()
params_ob['dd_frac'] = 0.02


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

trials_oddball_red_fwr, trials_oddball_red_rev = f_label_redundants(ob_data1['trials_oddball_ctx3'])

trials_oddball_red_fwr_cut = trials_oddball_red_fwr[num_skip_trials:,:]
trials_oddball_red_rev_cut = trials_oddball_red_rev[num_skip_trials:,:]

#%% load all params and rnns

rnn_leg = ['ob trained', 'freq trained', 'untrained']

rnn_all = []
params_all = []

# oddball rnn
rnn_ob_all = []
params_ob_all = []
for n_rnn in range(num_rnn):
    params = np.load(path1 + '/RNN_data/' + rnn_flist[n_rnn][:-4] + '_params.npy', allow_pickle=True).item()
    
    if 'train_add_noise' not in params.keys():
        params['train_add_noise'] = 0
    
    rnn = RNN_chaotic(params['input_size'], params['hidden_size'], params['num_freq_stim'] + 1, params['num_ctx'] + 1, params['dt']/params['tau'], params['train_add_noise'], activation=params['activation']).to(params['device'])
    rnn.init_weights(params['g'])
    rnn.load_state_dict(torch.load(path1 + '/RNN_data/' + rnn_flist[n_rnn]))
    rnn.cpu()
    
    rnn_ob_all.append(rnn)
    params_ob_all.append(params)

rnn_all.append(rnn_ob_all)
params_all.append(params_ob_all)

# freq rnn
rnnf_all = []
paramsf_all = []
for n_rnn in range(num_rnnf):
    params = np.load(path1 + '/RNN_data/' + rnnf_flist[n_rnn][:-4] + '_params.npy', allow_pickle=True).item()
    
    if 'train_add_noise' not in params.keys():
        params['train_add_noise'] = 0
    
    rnn = RNN_chaotic(params['input_size'], params['hidden_size'], params['num_freq_stim'] + 1, params['num_ctx'] + 1, params['dt']/params['tau'], params['train_add_noise'], activation=params['activation']).to(params['device'])
    rnn.init_weights(params['g'])
    rnn.load_state_dict(torch.load(path1 + '/RNN_data/' + rnnf_flist[n_rnn]))
    rnn.cpu()
    
    rnnf_all.append(rnn)
    paramsf_all.append(params)
    
rnn_all.append(rnnf_all)
params_all.append(paramsf_all)

# untrained rnn
params0 = params_ob_all[0]
rnn0_all = []
params0_all = []
for n_rnn in range(num_rnn0):
    rnn0 = RNN_chaotic(params['input_size'], params['hidden_size'], params['num_freq_stim'] + 1, params['num_ctx'] + 1, params['dt']/params['tau'], params['train_add_noise'], activation=params['activation']).to('cpu')
    rnn0.init_weights(params['g'])
    
    rnn0_all.append(rnn0)
    params0_all.append(params0)
    
rnn_all.append(rnn0_all)
params_all.append(params0_all)

loss_freq = nn.CrossEntropyLoss().cpu()
loss_ctx = nn.CrossEntropyLoss(weight = torch.tensor(params['train_loss_weights']).to('cpu'))

#%%
    
test_ob_all = []
test_cont_all = []
for n_tr in range(3):
    test_ob2 = []
    test_cont2 = []
    for n_rnn in range(len(rnn_all[n_tr])):
        print('test gr %s, rnn %d of %d' % (n_tr+1, n_rnn+1, len(rnn_all[n_tr])))
        test_oddball_ctx = f_RNN_test(rnn_all[n_tr][n_rnn], loss_ctx, ob_data1['input_oddball'], ob_data1['target_oddball_ctx'], paradigm='ctx')
        test_ob2.append(test_oddball_ctx)
        test_cont_freq = f_RNN_test(rnn_all[n_tr][n_rnn], loss_freq, cont_data['input_control'], cont_data['target_control'], paradigm='freq')
        test_cont2.append(test_cont_freq)
    test_ob_all.append(test_ob2)
    test_cont_all.append(test_cont2)

#%%



