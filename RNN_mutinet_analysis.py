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
             # 'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_2023_10_5_10h_54m_RNN',
             'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_2023_10_6_17h_3m_RNN',
             'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2023_12_19_16h_20m_RNN', # *****
             # # 'oddball2_1ctx_80000trainsamp_250neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_1_5_13h_15m_RNN', # 250 neurons
             # # 'oddball2_1ctx_120000trainsamp_250neurons_ReLU_100tau_10dt_20trials_50stim_100batch_4e-05lr_noise1_2024_1_22_12h_49m_RNN', # not long enough? 250 neurons dt 10 cant use different dt
             # # 'oddball2_1ctx_120000trainsamp_100neurons_ReLU_100tau_10dt_20trials_50stim_100batch_4e-05lr_noise1_2024_2_1_11h_45m_RNN', # not long enough? 100 neurons # dt 10
             # # 'oddball2_1ctx_180000trainsamp_100neurons_ReLU_100tau_10dt_20trials_50stim_100batch_4e-05lr_noise1_2024_2_4_20h_33m_RNN', # not long enough? 100 neurons # dt 10
             # 'oddball2_1ctx_140000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_2_19_10h_15m_RNN',
             # 'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_2_24_13h_23m_RNN',
             # 'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_2_28_11h_32m_RNN',
             #'oddball2_1ctx_250000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_3_3_13h_54m_RNN', # weird
             #'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-03lr_noise1_2024_3_6_16h_3m_RNN',
             #'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_3_7_12h_8m_RNN',
             #'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_2e-03lr_noise1_2024_3_8_11h_57m_RNN',
             'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_2e-03lr_noise1_2024_3_10_4h_38m_RNN',
             ]

rnnf_flist = ['freq2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise0_2023_12_20_0h_34m_RNN',
              'freq2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2023_12_20_0h_34m_RNN',
              # 'freq2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_1_4_13h_14m_RNN', # not long enough?
              # #'freq2_1ctx_80000trainsamp_250neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-04lr_noise1_2024_1_10_11h_28m_RNN', # bit spiky
              # #'freq2_1ctx_120000trainsamp_250neurons_ReLU_500tau_50dt_20trials_50stim_100batch_4e-05lr_noise1_2024_1_11_11h_33m_RNN', # bit spiky
              # 'freq2_1ctx_160000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_2_20_19h_9m_RNN', # bit spiky
              # 'freq2_1ctx_250000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_2_22_16h_20m_RNN',
              # 'freq2_1ctx_300000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_2_27_13h_17m_RNN',
              # 'freq2_1ctx_250000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_3_1_10h_1m_RNN',
              # 'freq2_1ctx_300000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_3_4_19h_25m_RNN',
              'freq2_1ctx_300000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_3_5_0h_28m_RNN',
              ]

num_rnn = len(rnn_flist)
num_rnnf = len(rnnf_flist)

num_rnn0 = num_rnn


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


num_ob_runs = 400
num_ob_trials = 100

num_cont_runs = 50
num_cont_trials = 100

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

# input_test = ob_data1['input_oddball']
# target_test = ob_data1['target_oddball_ctx']
# rnn = rnn_all[n_tr][n_rnn]
#%%

for n_tr in range(3):
    for n_rnn in range(len(rnn_all[n_tr])):
        numnans = np.sum(np.isnan(test_ob_all[n_tr][n_rnn]['rates']))
        if numnans:
            print('oddball %s, rnn %d has %d nans' % (rnn_leg[n_tr], n_rnn, numnans))
        numnans = np.sum(np.isnan(test_cont_all[n_tr][n_rnn]['rates']))
        if numnans:
            print('control %s, rnn %d has %d nans' % (rnn_leg[n_tr], n_rnn, numnans))

#%%
net_idx = []
for n_tr in range(3):
    for n_rnn in range(len(rnn_all[n_tr])):
        net_idx.append(n_tr)
        
        f_cut_reshape_rates_wrap(test_ob_all[n_tr][n_rnn], params_all[n_tr][n_rnn], num_skip_trials = num_skip_trials)
        f_run_dred_wrap(test_ob_all[n_tr][n_rnn], subtr_mean=dred_subtr_mean, method=dred_met)
        
        f_cut_reshape_rates_wrap(test_cont_all[n_tr][n_rnn], params_all[n_tr][n_rnn], num_skip_trials = num_skip_trials)
        f_run_dred_wrap(test_cont_all[n_tr][n_rnn], subtr_mean=dred_subtr_mean, method=dred_met)
    
net_idx = np.array(net_idx)

# rates3d_in = test_ob_all[n_tr][n_rnn]['rates']
# params = params_all[n_tr][n_rnn]
#%% plot mmn population response
# plt.close('all')
base_sub = True
split_pos_neg = False

for n_tr in range(3):
    for n_rnn in range(len(test_ob_all[n_tr])):
        #f_plot_mmn(test_ob_all[n_tr][n_rnn]['rates4d_cut'], trials_oddball_ctx_cut, params_all[n_tr][n_rnn], title_tag='ob trained RNN')
        f_plot_mmn2(trials_oddball_ctx_cut, test_ob_all[n_tr][n_rnn]['rates4d_cut'], trials_cont_cut, test_cont_all[n_tr][n_rnn]['rates4d_cut'], params_all[n_tr][n_rnn], red_dd_seq, title_tag='%s RNN%d' % (rnn_leg[n_tr], n_rnn+1), baseline_subtract=base_sub, split_pos_cells=split_pos_neg)
        #f_plot_traj_speed(test_ob_all[n_rnn]['rates'], ob_data1, n_run, start_idx=1000, title_tag= 'trained RNN %d; run %d' % (n_rnn, n_run))

# rates4d_cut = test_ob_all[n_tr][n_rnn]['rates4d_cut']
# rates_cont_freq4d_cut = test_cont_all[n_tr][n_rnn]['rates4d_cut']
#%% MMN pool
base_sub = True

rdc_all = []

for n_gr in range(3):
    for n_rnn in range(len(test_ob_all[n_gr])):
        rates4d_cut = test_ob_all[n_gr][n_rnn]['rates4d_cut']
        trial_len, num_tr, num_runs, num_cells = rates4d_cut.shape
        
        plot_t1 = (np.arange(trial_len)-trial_len/4)*params_all[n_gr][n_rnn]['dt']
        
        rdc_all.append(f_get_rdc_trav(trials_oddball_ctx_cut, rates4d_cut, trials_cont_cut, test_cont_all[n_gr][n_rnn]['rates4d_cut'], params, red_dd_seq, baseline_subtract=base_sub))


rdc_all = np.array(rdc_all)

_, _, _, num_freqs, num_cells = rdc_all.shape

for n_net in range(3):
    rdc_all2 = rdc_all[net_idx==n_net]
    rdc_all3 = np.moveaxis(rdc_all2, 0, -1)
    
    num_cells2 = num_cells*np.sum(net_idx==n_net)
    rdc_all4 = np.reshape(rdc_all3, (trial_len, 3, num_freqs*num_cells2), order='F')
    

    mmn_mean = np.mean(rdc_all4, axis=2)
    mmn_sem = np.std(rdc_all4, axis=2)/np.sqrt(num_freqs*num_cells2-1)
    
    colors_ctx = ['blue', 'red', 'black']
    if num_cells:
        plt.figure()
        for n_ctx in range(3):
            plt.plot(plot_t1, mmn_mean[:,n_ctx], color=colors_ctx[n_ctx])
            plt.fill_between(plot_t1, mmn_mean[:,n_ctx]-mmn_sem[:,n_ctx], mmn_mean[:,n_ctx]+mmn_sem[:,n_ctx], color=colors_ctx[n_ctx], alpha=0.2)
        plt.title('%s RNN population trial ave; %d cells' % (rnn_leg[n_net], num_cells2))

#%% compute MMN response magnitude

base_sub = True
plot_t1 = (np.arange(trial_len)-trial_len/4)*params['dt']
on_time = (plot_t1>0.2)*(plot_t1<0.5)

dr_ratio = []
mmn_mag = []
mmn_magn = []

for n_tr in range(3):
    for n_rnn in range(len(test_ob_all[n_tr])):
        trial_ave_rdc = f_get_rdc_trav(trials_oddball_ctx_cut, test_ob_all[n_tr][n_rnn]['rates4d_cut'], trials_cont_cut, test_cont_all[n_tr][n_rnn]['rates4d_cut'], params_all[n_tr][n_rnn], red_dd_seq, baseline_subtract=base_sub)
        
        rdc_mag = np.mean(np.mean(np.mean(trial_ave_rdc[on_time,:,:,:], axis=0), axis=2), axis=1)
        dr_ratio.append(np.abs(rdc_mag[1]/rdc_mag[0]))
        mmn_mag.append((rdc_mag[1] - rdc_mag[0]))
        mmn_magn.append((rdc_mag[1] - rdc_mag[0])/np.max(trial_ave_rdc))


dr_ratio = np.array(dr_ratio)
mmn_mag = np.array(mmn_mag)
mmn_magn = np.array(mmn_magn)

#%%

trial_len = round((params['stim_duration']+params['isi_duration'])/params['dt'])
trial_stim_on = np.zeros(trial_len, dtype=bool)
trial_stim_on[round(np.floor(params['isi_duration']/params['dt']/2)):(round(np.floor(params['isi_duration']/params['dt']/2))+round(params['stim_duration']/params['dt']))] = 1
plot_t1 = (np.arange(trial_len)-trial_len/4)*params['dt']

stim_loc = trials_oddball_ctx_cut


rates_in = []
shuff_stim_type = []
shuff_bins = []
leg_all = []
for n_tr in range(3):
    for n_rnn in range(len(test_ob_all[n_tr])):
        rates_in.append(test_ob_all[n_tr][n_rnn]['rates4d_cut'])
        shuff_stim_type.append(0)
        shuff_bins.append(0)
        leg_all.append('%s, rnn%d' % (rnn_leg[n_tr], n_rnn))
        #params_all[n_tr][n_rnn]
        

# rates_in = [test_oddball_ctx['rates4d_cut'], testf_oddball_ctx['rates4d_cut'], test0_oddball_ctx['rates4d_cut'], test_oddball_ctx['rates4d_cut'], test_oddball_ctx['rates4d_cut']]
# leg_all = ['ob trained', 'freq trained', 'untrained', 'ob stim shuff', 'ob bin shuff']
# shuff_stim_type = [0, 0, 0, 1, 0]
# shuff_bins = [0, 0, 0, 0, 1]


x_data, y_data = f_sample_trial_data_dec(rates_in, stim_loc, [1, 0])

perform1_final, perform1_binwise, perform1_y_is_cat = f_run_one_shot_dec(x_data, y_data, trial_stim_on, shuff_stim_type, shuff_bins, stim_on_train=False, num_cv=5, equalize_y_input=True)

f_plot_one_shot_dec_bycat(perform1_final, perform1_binwise, perform1_y_is_cat, plot_t1, leg_all, ['deviant', 'redundant'], ['pink', 'lightblue'])

f_plot_one_shot_dec_bycat2(perform1_final, perform1_binwise, perform1_y_is_cat, plot_t1, net_idx, rnn_leg, ['deviant', 'redundant'], ['pink', 'lightblue'])


perform_final = perform1_final
perform_binwise = perform1_binwise
perform_y_is_cat = perform1_y_is_cat
trial_labels = ['deviant', 'redundant']
trial_colors = ['pink', 'lightblue']
#%%
# plt.close('all')
plt.figure()
for n_net in range(3):
    plt.plot(dr_ratio[net_idx==n_net], perform1_final[net_idx==n_net], '.')
plt.title('D/R ratio vs ob performance')
plt.xlabel('dev/red ratio')
plt.ylabel('Oddball performance')
plt.legend(rnn_leg)

plt.figure()
for n_net in range(3):
    plt.plot(mmn_mag[net_idx==n_net], perform1_final[net_idx==n_net], '.')
plt.title('MMN vs ob performance')
plt.xlabel('MMN magnitude')
plt.ylabel('Oddball performance')
plt.legend(rnn_leg)

plt.figure()
for n_net in range(3):
    plt.plot(mmn_magn[net_idx==n_net], perform1_final[net_idx==n_net], '.')
plt.title('Normalized MMN vs ob performance')
plt.xlabel('Normalized MMN')
plt.ylabel('Oddball performance')
plt.legend(rnn_leg)

#%%

# if 'red_count' not in ob_data1.keys():
#     ctx_data = ob_data1['trials_oddball_ctx']
#     num_trials, num_runs = ctx_data.shape
#     red_count = np.zeros((num_trials, num_runs))
#     for n_run in range(num_runs):
#         red_count1 = 1
#         for n_tr in range(num_trials):
#             if not ctx_data[n_tr,n_run]:
#                 red_count[n_tr,n_run] = red_count1
#                 red_count1 += 1
#             else:
#                 red_count1 = 1
        
#     ob_data1['red_count'] = red_count

#%%
colors1 = cm.jet(np.linspace(0,1,params['num_freq_stim']))
if 0:
    plt.figure()
    plt.imshow(colors1[:,:3].reshape((50,1,3)), aspect=.2)
    plt.ylabel('color map')
    plt.xticks([])

#%%    

cell_w_out = []
cell_w_in = []

for n_tr in range(3):
    for n_rnn in range(len(rnn_all[n_tr])):
        temp_rnn = rnn_all[n_tr][n_rnn]
        Wr = temp_rnn.h2h.weight.detach().numpy()
        
        num_cells = Wr.shape[0]
        
        w_out_mean = np.mean(Wr, axis=0)
        w_in_mean = np.mean(Wr, axis=1)
        
        cell_w_out.append(w_out_mean)
        cell_w_in.append(w_in_mean)

cell_w_out = np.array(cell_w_out)
cell_w_in = np.array(cell_w_in)


f_plot_cat_data(cell_w_out, net_idx, rnn_leg, title_tag = 'mean Wr out')

f_plot_cat_data(cell_w_in, net_idx, rnn_leg, title_tag = 'mean Wr in')

    
#%% 

# num_corr_pts = 50
# corr_len = 300
# corr_neur_len = 100

# sm_bin = 10#round(1/params['dt'])*50;
# #trial_len = out_temp_all.shape[1]
# kernel = np.ones(sm_bin)/sm_bin


tau_net_all = []
tau_cell_has_data = []
tau_cell_all = []

for n_gr in range(3):
    for n_rnn in range(len(rnn_all[n_gr])):

        rates4d = test_ob_all[n_gr][n_rnn]['rates4d_cut']
        
        trial_len, num_tr, num_runs, num_cells = rates4d.shape
        
        
        tau_net = np.zeros(num_runs)
        tau_cell = np.zeros((num_runs, num_cells))
        has_data_neur = np.zeros((num_runs, num_cells), dtype=bool)
        
        
        temp_ave4d, trial_data_sort, num_dd_trials = f_trial_ave_ctx_pad2(rates4d, trials_oddball_ctx_cut, pre_dd = 1, post_dd = 16, limit_1_dd=False, max_trials=999, shuffle_trials=False)
        
        temp_ave3d = np.reshape(temp_ave4d, (20*temp_ave4d.shape[1],100,25), order='F')
        
        plt.figure()
        plt.plot(np.mean(temp_ave3d[:,3,:], axis=1))
        
        plt.figure()
        plt.plot(temp_ave3d[:,3,:])
        
        for n_run in range(num_runs):
            
            #plt.figure(); plt.plot(np.reshape(rates4d[:,:,n_run,:], (trial_len*num_tr, num_cells)))
            
            base_vec = np.mean(temp_ave4d[:,0,n_run,:], axis=0)
            
            resp_vec = np.reshape(temp_ave4d[:,:,n_run,:], (trial_len*temp_ave4d.shape[1], num_cells), order='F')
            
            tr_dist = cdist(np.reshape(base_vec, (1,num_cells)), resp_vec, 'euclidean')[0]
            
            tau_net[n_run] = f_get_trace_tau(tr_dist, sm_bin = 10)*params_all[n_gr][n_rnn]['dt']
            
            # last_red = np.where(trials_oddball_red_rev_cut[:,n_run] == -1)[0]
            # trial_tau = np.zeros(len(last_red))
            # for n_tr in range(len(last_red)):
            #     base_vec = np.mean(rates4d[:,last_red[n_tr],n_run,:], axis=0)
            #     resp_vec = rates4d[:,last_red[n_tr]:last_red[n_tr]+10,n_run,:]
            #     tr_dist = cdist(np.reshape(base_vec, (1,num_cells)), np.reshape(rates4d[:,last_red[n_tr]:last_red[n_tr]+10,n_run,:], (trial_len*resp_vec.shape[1],num_cells)), 'euclidean')[0]
            #     # plt.figure(); plt.plot(tr_dist)
            #     trial_tau[n_tr] = f_get_trace_tau(tr_dist, sm_bin = 10)*params['dt']
                
            # trial_ave1 = np.mean(np.mean(rates4d[:,last_red,n_run,:], axis=1), axis=0)
            
            # dist1 = cdist(np.reshape(trial_ave1, (1,num_cells)), np.reshape(rates4d[:,:,n_run,:], (trial_len*num_tr,num_cells)), 'euclidean')[0]
            
            # plt.figure(); plt.plot(tr_dist)
            
            # tau_corr = f_get_trace_tau(tr_dist, sm_bin = 10)*params['dt']
            
            #np.mean(trial_tau)
            
            for n_nr in range(num_cells):
                neur = np.reshape(temp_ave4d[:,:,n_run,n_nr], (trial_len*temp_ave4d.shape[1]))
                # plt.figure(); plt.plot(neur)
                
                if np.sum(neur) > 0.1:
                    
                    tau_neur = f_get_trace_tau(neur, sm_bin = 10)*params['dt']

                    tau_cell[n_run, n_nr] = tau_neur
                    has_data_neur[n_run, n_nr] = True

        tau_net_all.append(tau_net)
        tau_cell_has_data.append(has_data_neur)
        tau_cell_all.append(tau_cell)


tau_net_all = np.array(tau_net_all)
tau_cell_all = np.array(tau_cell_all)
tau_cell_has_data = np.array(tau_cell_has_data)

f_plot_cat_data(tau_net_all, net_idx, rnn_leg, title_tag = 'Tau network')

f_plot_cat_data(tau_cell_all, net_idx, rnn_leg, title_tag = 'Tau neurons', cell_idx=tau_cell_has_data)


#%%



#%% plot control vectors mags of indiv vs trial ave
red_tr_idx = -3
trials_cont_vec_all = []
trials_dev_all = []
trials_red_all = []
for n_gr in range(3):
    for n_rnn in range(len(rnn_all[n_gr])):
        trials_cont_vec = f_analyze_cont_trial_vectors(test_cont_all[n_gr][n_rnn]['rates4d_cut'],  trials_cont_cut, red_dd_seq, params_all[n_gr][n_rnn], base_time = [-.250, 0], on_time = [.2, .5])
        
        trials_cont_vec_all.append(trials_cont_vec)
        
        trials_rd_dev = f_gather_dev_trials(test_ob_all[n_gr][n_rnn]['rates4d_cut'], trials_oddball_ctx_cut, red_dd_seq)
        trials_dev_vec = f_analyze_trial_vectors(trials_rd_dev, params_all[n_gr][n_rnn])
        trials_dev_all.append(trials_dev_vec)
        
        trials_rd_red = f_gather_red_trials(test_ob_all[n_gr][n_rnn]['rates4d_cut'], trials_oddball_freq_cut, trials_oddball_ctx_cut, red_dd_seq, red_idx = red_tr_idx)
        trials_red_vec = f_analyze_trial_vectors(trials_rd_red, params_all[n_gr][n_rnn])
        trials_red_all.append(trials_red_vec)
        
        
for n_rnn in range(10):
        plt.figure()
        plt.imshow(trials_cont_vec_all[n_rnn]['base_dist_mean'])
        plt.colorbar()

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


mean_indiv_mag_dev_all = []
mean_indiv_mag_red_all = []

for n_rnn in range(len(trials_dev_all)):
    mean_indiv_mag_dev_all.append(trials_dev_all[n_rnn]['mean_indiv_mag'])
    mean_indiv_mag_red_all.append(trials_red_all[n_rnn]['mean_indiv_mag'])


mean_indiv_mag_dev_all = np.array(mean_indiv_mag_dev_all)
mean_indiv_mag_red_all = np.array(mean_indiv_mag_red_all)
    
for n_net in range(3):
    
    plt.figure()
    plt.imshow(np.mean(mean_indiv_mag_dev_all[net_idx==n_net], axis=0))
    plt.colorbar()
    plt.title('%s RNN dev trials' % rnn_leg[n_net])
    plt.ylabel('deviant freq')
    plt.xlabel('redundant freq')
    
    plt.figure()
    plt.imshow(np.mean(mean_indiv_mag_red_all[net_idx==n_net], axis=0))
    plt.colorbar()
    plt.title('%s RNN red trials' % rnn_leg[n_net])
    plt.ylabel('deviant freq')
    plt.xlabel('redundant freq')
    
    
    plt.figure()
    plt.imshow(np.mean(mean_indiv_mag_dev_all[net_idx==n_net], axis=0)/np.mean(mean_indiv_mag_red_all[net_idx==n_net], axis=0))
    plt.colorbar()
    plt.title('%s RNN dev-red ratio' % rnn_leg[n_net])
    plt.ylabel('deviant freq')
    plt.xlabel('redundant freq')
    

#%% speed comparisons

speeds_dr = np.zeros((num_rnn, num_run, 2))

for n_rnn in range(num_rnn):
    test_data = test_ob_all[n_rnn]
    
    rates = test_data['rates_cut']
    
    
    for n_run in range(num_run):
        print('rnn %d; run %d' % (n_rnn, n_run))
        
        dist1 = squareform(pdist(rates[:,n_run,:], metric='euclidean'))
        dist2 = np.hstack((0,np.diag(dist1, 1)))
        
        dist23d = np.reshape(dist2, (trial_len, num_trials2), order = 'F')
        
        
        red_count_cut = ob_data1['red_count'][num_skip_trials:,:]
        dd_trial = np.where(red_count_cut[:,n_run] == 0)[0]
        last_red = dd_trial-1
        
        mean_dd_speed = np.mean(dist23d[5:15,dd_trial])
        mean_red_speed = np.mean(dist23d[5:15,last_red])
        
        speeds_dr[n_rnn, n_run, 0] = mean_dd_speed
        speeds_dr[n_rnn, n_run, 1] = mean_red_speed
 

speeds0_dr = np.zeros((num_rnn0, num_run, 2))

for n_rnn in range(num_rnn0):
    test_data = test0_ob_all[n_rnn]
    
    rates = test_data['rates_cut']

    for n_run in range(num_run):
        
        print('rnn %d; run %d' % (n_rnn, n_run))
        
        dist1 = squareform(pdist(rates[:,n_run,:], metric='euclidean'))
        dist2 = np.hstack((0,np.diag(dist1, 1)))
        
        dist23d = np.reshape(dist2, (trial_len, num_trials2), order = 'F')
        
        
        red_count_cut = ob_data1['red_count'][num_skip_trials:,:]
        dd_trial = np.where(red_count_cut[:,n_run] == 0)[0]
        last_red = dd_trial-1
        
        mean_dd_speed = np.mean(dist23d[5:15,dd_trial])
        mean_red_speed = np.mean(dist23d[5:15,last_red])
        
        speeds0_dr[n_rnn, n_run, 0] = mean_dd_speed
        speeds0_dr[n_rnn, n_run, 1] = mean_red_speed


#%%


plt.figure()
for n_rnn in range(num_rnn):
    norm_f = speeds_dr[n_rnn,:,0] + speeds_dr[n_rnn,:,1]
    plt.plot(speeds_dr[n_rnn,:,0], speeds_dr[n_rnn,:,1], '.')
plt.xlabel('dd mean speed')
plt.ylabel('red mean speed')
plt.xlim((np.min(speeds_dr[n_rnn,:,:])*0.95, np.max(speeds_dr[n_rnn,:,:])*1.05))
plt.ylim((np.min(speeds_dr[n_rnn,:,:])*0.95, np.max(speeds_dr[n_rnn,:,:])*1.05))
plt.title('trained')

plt.figure()
for n_rnn in range(num_rnn0):
    norm_f = speeds0_dr[n_rnn,:,0] + speeds0_dr[n_rnn,:,1]
    plt.plot(speeds0_dr[n_rnn,:,0], speeds0_dr[n_rnn,:,1], '.')
plt.xlabel('dd mean speed')
plt.ylabel('red mean speed')
plt.xlim((np.min(speeds0_dr[n_rnn,:,:])*0.95, np.max(speeds0_dr[n_rnn,:,:])*1.05))
plt.ylim((np.min(speeds0_dr[n_rnn,:,:])*0.95, np.max(speeds0_dr[n_rnn,:,:])*1.05))
plt.title('untrained')




