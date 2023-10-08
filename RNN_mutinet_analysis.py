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
from f_RNN import f_RNN_trial_ctx_train2, f_RNN_trial_freq_train2, f_RNN_test, f_RNN_test_spont, f_gen_dset, f_trial_ave_ctx_pad, f_run_dred, f_plot_dred_rates, f_plot_traj_speed, f_plot_resp_distances, f_plot_mmn, f_plot_dred_pcs, f_plot_rnn_weights
from f_RNN_chaotic import RNN_chaotic
from f_RNN_utils import f_gen_stim_output_templates, f_gen_cont_seq, f_gen_oddball_seq, f_gen_input_output_from_seq, f_plot_examle_inputs, f_plot_train_loss, f_plot_train_test_loss


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
from scipy.spatial.distance import pdist #, cdist, squareform
#from scipy.sparse import diags
#from scipy import signal
from scipy import linalg
#from scipy.io import loadmat, savemat
#import skimage.io

from datetime import datetime



#%%

rnn_flist = ['oddball2_1ctx_80000trainsamp_25neurons_ReLU_0.50tau_20trials_50stim_100batch_0.0010lr_2023_9_11_14h_19m_RNN',
             'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_0.0010lr_2023_10_4_17h_16m_RNN',
             'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_0.0010lr_2023_10_5_10h_54m_RNN',
             'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_0.0010lr_2023_10_6_17h_3m_RNN']

num_rnn = len(rnn_flist)

num_rnn0 = 6


dparams = {}
dparams['num_trials'] = 1000
dparams['num_batch'] = 20
dparams['num_dev_stim'] = 5
dparams['num_red_stim'] = 50


#%%

params_all = []
rnn_all = []
for n_rnn in range(num_rnn):
    params = np.load(path1 + '/RNN_data/' + rnn_flist[n_rnn][:-4] + '_params.npy', allow_pickle=True).item()

    rnn = RNN_chaotic(params['input_size'], params['hidden_size'], params['num_freq_stim'] + 1, params['num_ctx'] + 1, params['dt']/params['tau'], activation=params['activation']).to(params['device'])
    rnn.init_weights(params['g'])
    rnn.load_state_dict(torch.load(path1 + '/RNN_data/' + rnn_flist[n_rnn]))
    rnn.cpu()
    
    params_all.append(params)
    rnn_all.append(rnn)

params = params_all[0]

rnn0_all = []
for n_rnn in range(num_rnn0):
    rnn0 = RNN_chaotic(params['input_size'], params['hidden_size'], params['num_freq_stim'] + 1, params['num_ctx'] + 1, params['dt']/params['tau'], activation=params['activation']).to('cpu')
    rnn0.init_weights(params['g'])
    
    rnn0_all.append(rnn0)


loss_freq = nn.CrossEntropyLoss().cpu()
loss_ctx = nn.CrossEntropyLoss(weight = torch.tensor(params['train_loss_weights']).to('cpu'))


#%%

params2 = params.copy()
params3 = params.copy()

params2['dd_frac'] = 0.02

params3['isi_duration'] = 0
params3['stim_duration'] = 1
params3['dd_frac'] = 0


num_skip_trials = 50


trial_len = round((params['stim_duration'] + params['isi_duration']) / params['dt'])

num_run = dparams['num_batch']
num_trials = dparams['num_trials']
num_trials2 = num_trials - num_skip_trials


#%%

stim_templates = f_gen_stim_output_templates(params)

ob_data1 = f_gen_dset(dparams, params2, stim_templates, stim_sample='equal')

stim_templates_mono = f_gen_stim_output_templates(params3)

# make long constant sounds
red_stim_const = np.round(np.hstack((0,np.linspace(0,params['num_freq_stim']+1, dparams['num_red_stim']+2)[1:-1]))).astype(int)

trials_const_freq = (np.ones((dparams['num_trials'], red_stim_const.shape[0]))* red_stim_const.reshape((1,red_stim_const.shape[0]))).astype(int)
trials_const_ctx = np.zeros((dparams['num_trials'], red_stim_const.shape[0]), dtype=int)

trials_const_input, trials_const_output_freq = f_gen_input_output_from_seq(trials_const_freq, stim_templates_mono['freq_input'], stim_templates_mono['freq_output'], params2)
_, trials_const_output_ctx = f_gen_input_output_from_seq(trials_const_ctx, stim_templates_mono['freq_input'], stim_templates_mono['ctx_output'], params2)


#%%
colors1 = cm.jet(np.linspace(0,1,params['num_freq_stim']))
if 0:
    plt.figure()
    plt.imshow(colors1[:,:3].reshape((50,1,3)), aspect=.2)
    plt.ylabel('color map')
    plt.xticks([])
    
    
#%%

test_ob_all = []
for n_rnn in range(num_rnn):
    print('test rnn %d of %d' % (n_rnn+1, num_rnn))

    test_oddball_ctx = f_RNN_test(rnn_all[n_rnn], loss_ctx, ob_data1['input_test_oddball'], ob_data1['output_test_oddball_ctx'], params, paradigm='ctx')

    test_ob_all.append(test_oddball_ctx)
    
    
test0_ob_all = []
for n_rnn in range(num_rnn0):
    print('test rnn0 %d of %d' % (n_rnn+1, num_rnn0))
    
    test_oddball_ctx = f_RNN_test(rnn0_all[n_rnn], loss_ctx, ob_data1['input_test_oddball'], ob_data1['output_test_oddball_ctx'], params, paradigm='ctx')
    
    test0_ob_all.append(test_oddball_ctx)


#%%

num_t, num_batch, num_cells = test_ob_all[0]['rates'].shape

for n_rnn in range(num_rnn):

    test_oddball_ctx = test_ob_all[n_rnn]
    
    rates = test_oddball_ctx['rates'] #(8000, 100, 25)
    
    rates4d = np.reshape(rates, (trial_len, num_trials, dparams['num_batch'], num_cells), order = 'F')
    rates4d_cut = rates4d[:,num_skip_trials:,:,:]
    
    rates_cut = np.reshape(rates4d_cut, (trial_len*num_trials2, num_run, num_cells), order = 'F')
    
    num_t2, _, _ = rates_cut.shape
    
    rates2d_cut = np.reshape(rates_cut, (num_t2*num_run, num_cells), order = 'F')
    
    test_oddball_ctx['rates4d_cut'] = rates4d_cut
    test_oddball_ctx['rates_cut'] = rates_cut
    test_oddball_ctx['rates2d_cut'] = rates2d_cut
    
    
for n_rnn in range(num_rnn0):

    test_oddball_ctx = test0_ob_all[n_rnn]
    
    rates = test_oddball_ctx['rates'] #(8000, 100, 25)
    
    rates4d = np.reshape(rates, (trial_len, num_trials, dparams['num_batch'], num_cells), order = 'F')
    rates4d_cut = rates4d[:,num_skip_trials:,:,:]
    
    rates_cut = np.reshape(rates4d_cut, (trial_len*num_trials2, num_run, num_cells), order = 'F')
    
    num_t2, _, _ = rates_cut.shape
    
    rates2d_cut = np.reshape(rates_cut, (num_t2*num_run, num_cells), order = 'F')
    
    test_oddball_ctx['rates4d_cut'] = rates4d_cut
    test_oddball_ctx['rates_cut'] = rates_cut
    test_oddball_ctx['rates2d_cut'] = rates2d_cut

    
#%%
# plt.close('all')
n_run = 2

for n_rnn in range(num_rnn):
    
    f_plot_traj_speed(test_ob_all[n_rnn]['rates'], ob_data1, n_run, start_idx=1000, title_tag= 'trained RNN %d; run %d' % (n_rnn, n_run))
    
    
for n_rnn in range(num_rnn0):
    
    f_plot_traj_speed(test0_ob_all[n_rnn]['rates'], ob_data1, n_run, start_idx=1000, title_tag= 'trained RNN %d; run %d' % (n_rnn, n_run))
    

#%%
    
    
    
    
    
    
    
    
    
    
    
