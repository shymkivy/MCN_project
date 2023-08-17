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
sys.path.append(path1 + 'RNN_scripts');

from f_analysis import *
from f_RNN import *
from f_RNN_chaotic import *
from f_RNN_utils import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import colors
import matplotlib.cm as cm
from random import sample, random
import math

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy import signal
from scipy import linalg
from scipy.io import loadmat, savemat
import skimage.io

from datetime import datetime

#%% params
load_RNN = 1
plot_deets = 1

#%% input params

params = {'train_type':                     'oddball2',     #   oddball2, freq2  standard, linear, oddball, freq_oddball,
          'device':                         'cpu',         # 'cpu', 'cuda'
          
          'stim_duration':                  0.5,
          'isi_duration':                   0.5,
          'num_freq_stim':                  50,
          'num_ctx':                        1,
          'oddball_stim':                   np.arange(50)+1, # np.arange(10)+1, #[3, 6], #np.arange(10)+1,
          'dd_frac':                        0.1,
          'dt':                             0.05,
          
          'train_batch_size':               100,
          'train_trials_in_sample':         20,
          'train_num_samples_freq':         1000,
          'train_num_samples_ctx':          80000,
          'train_loss_weights':              [0.05, 0.95], # isi, red, dd [1e-5, 1e-5, 1] [0.05, 0.05, 0.9], [0.05, 0.95]  [1/.5, 1/.45, 1/0.05]

          'train_repeats_per_samp':         1,
          'train_reinit_rate':              0,
          
          'test_batch_size':                100,
          'test_trials_in_sample':          400,
          'test_oddball_stim':              np.arange(10)+1,        #[3, 5, 7],
          'test_num_freq_stim':             10,
          
          'input_size':                     50,
          'hidden_size':                    25,            # number of RNN neurons
          'g':                              1,  # 1            # recurrent connection strength 
          'tau':                            0.5,
          'learning_rate':                  0.001,           # 0.005
          'activation':                     'ReLU',             # ReLU tanh
          'normalize_input':                False,
          
          'stim_t_std':                     3,              # 3 or 0
          'input_noise_std':                1/100,
          
          'plot_deets':                     0,
          }

now1 = datetime.now()

save_tag = ''

name_tag1 = '%s%s_%dctx_%dtrainsamp_%dneurons_%s_%dtrials_%dstim' % (save_tag, params['train_type'], params['num_ctx'],
            params['train_num_samples_ctx'], params['hidden_size'], params['activation'], params['train_trials_in_sample'], params['num_freq_stim'])

name_tag2 = '%dbatch_%.4flr_%d_%d_%d_%dh_%dm' % (params['train_batch_size'], params['learning_rate'],
             now1.year, now1.month, now1.day, now1.hour, now1.minute)

name_tag  = name_tag1 + '_' + name_tag2

#%%

#fname_RNN_load = 'test_20k_std3'
#fname_RNN_load = '50k_20stim_std3';
#fname_RNN_load = 'oddball2_60000trainsamp_25neurons_ReLU_20trials_50stim_200batch_0.0010lr_2023_8_4_17h_41m_RNN'
#fname_RNN_load = 'oddball2_80000trainsamp_25neurons_ReLU_20trials_50stim_100batch_0.0010lr_2023_8_5_13h_59m_RNN'
#fname_RNN_load = 'oddball2_1ctx_20000trainsamp_25neurons_ReLU_20trials_50stim_100batch_0.0010lr_2023_8_14_13h_42m_RNN'
fname_RNN_load = 'oddball2_2ctx_80000trainsamp_25neurons_ReLU_20trials_50stim_100batch_0.0010lr_2023_8_15_13h_23m_RNN'

#fname_RNN_save = 'test_50k_std4'
#fname_RNN_save = '50k_20stim_std3'
fname_RNN_save = name_tag


#%%
if load_RNN:
    params = np.load(path1 + '/RNN_data/' + fname_RNN_load[:-4] + '_params.npy', allow_pickle=True).item()

#%% generate train data

#plt.close('all')

# generate stim templates


stim_templates = {}
stim_templates['freq_input'], stim_templates['freq_output'] = f_gen_stim_output_templates(params)

stim_templates['ctx_output'] = stim_templates['freq_output'][:params['num_ctx']+1,:,:params['num_ctx']+1]
    
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

rnn = RNN_chaotic(params['input_size'], params['hidden_size'], output_size, output_size_ctx, alpha, activation=params['activation']).to(params['device'])
rnn.init_weights(params['g'])

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

train_out = {}     # initialize outputs, so they are saved when process breaks

#%%
if load_RNN:
    print('Loading RNN %s' % fname_RNN_load)
    rnn.load_state_dict(torch.load(path1 + '/RNN_data/' + fname_RNN_load))

#%%
if not load_RNN:
    if params['train_type'] == 'standard':
        train_out = f_RNN_trial_train(rnn, loss, input_train_cont, output_train_cont, params)
    elif params['train_type'] == 'freq_oddball':
        train_out = f_RNN_trial_freq_ctx_train(rnn, loss, loss_ctx, input_train_oddball, output_train_oddball_freq, output_train_oddball_ctx, params)
    elif params['train_type'] == 'oddball':
        train_out = f_RNN_trial_ctx_train(rnn, loss_ctx, input_train_oddball, output_train_oddball_ctx, params)
        
        #train_cont = f_RNN_trial_ctx_train(rnn, loss, input_train_oddball_freq, output_train_oddball_freq, output_train_oddball_ctx, params)
    
    elif params['train_type'] == 'freq2':
        
        f_RNN_trial_freq_train2(rnn, loss_freq, stim_templates, params, train_out)
    
    elif params['train_type'] == 'oddball2':
        
        f_RNN_trial_ctx_train2(rnn, loss_ctx, stim_templates, params, train_out)
        
    
    else:
        train_out = f_RNN_linear_train(rnn, loss, input_train_cont, output_train_cont, params)
        
else:
    print('running without training')
    #train_out_cont = f_RNN_test(rnn, loss, input_train_cont, output_train_cont, params)
    
#%%
if load_RNN:
    train_out = np.load(path1 + '/RNN_data/' + fname_RNN_load[:-4] + '_train_out.npy', allow_pickle=True).item()

#%%
#plt.close('all')
figs = f_plot_train_loss(train_out, name_tag1, name_tag2)
    
# f_plot_rates(rates_all[:,:, 1], 10)
#%%
if not load_RNN:
    print('Saving RNN %s' % fname_RNN_save)
    torch.save(rnn.state_dict(), path1 + '/RNN_data/' + fname_RNN_save  + '_RNN')
    np.save(path1 + '/RNN_data/' + fname_RNN_save + '_params.npy', params) 
    np.save(path1 + '/RNN_data/' + fname_RNN_save + '_train_out.npy', train_out) 
    
    for key1 in figs.keys():
        figs[key1].savefig(path1 + '/RNN_data/' + fname_RNN_save + '_' + key1 + '.png', dpi=1200)


#%%

params['device'] = 'cpu';
rnn.cpu()

loss_freq.cpu()
loss_ctx.cpu()


#%% gen test data
# test control trials
#trials_test_cont = f_gen_cont_seq(params['num_freq_stim'], params['test_trials_in_sample'], params['test_batch_size'], 1)

if 'test_num_freq_stim' not in params.keys():
    params['test_num_freq_stim'] = 10

test_cont_stim = np.round(np.linspace(1, params['num_freq_stim'], params['test_num_freq_stim'])).astype(int)
trials_test_cont_idx = f_gen_cont_seq(params['test_num_freq_stim'], params['test_trials_in_sample'], params['test_batch_size'], 1)-1
trials_test_cont = test_cont_stim[trials_test_cont_idx]

input_test_cont, output_test_cont = f_gen_input_output_from_seq(trials_test_cont, stim_templates['freq_input'], stim_templates['freq_output'], params)

#%% test
test_cont_freq = f_RNN_test(rnn, loss_freq, input_test_cont, output_test_cont, params, paradigm='freq')

#%%
# plt.close('all')
f_plot_examle_inputs(input_test_cont, output_test_cont, params, 1)


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
test_oddball_freq = f_RNN_test(rnn, loss_freq, input_test_oddball, output_test_oddball_freq, params, paradigm='freq')

test_oddball_ctx = f_RNN_test(rnn, loss_ctx, input_test_oddball, output_test_oddball_ctx, params, paradigm='ctx')

#%%
# plt.close('all')
f_plot_examle_inputs(input_test_oddball, output_test_oddball_ctx, params, 5)


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

#%% create colormap jet 

colors1 = cm.jet(np.linspace(0,1,params['num_freq_stim']))
if 0:
    plt.figure()
    plt.imshow(colors1[:,:3].reshape((50,1,3)), aspect=.2)
    plt.ylabel('color map')
    plt.xticks([])

#%% spont analysis 
# plt.close('all')
if 0:
    #spont inputs

    input_shape = (trial_len*params['test_trials_in_sample'], params['test_batch_size'], params['input_size'])
    input_spont1 = np.random.normal(0,params['input_noise_std'], input_shape)
    
    test_spont = f_RNN_test_spont(rnn, input_spont1, params)
    
    f_plot_rates_only(test_spont, 'spont', num_plot_batches = 1, num_plot_cells = 25, preprocess = True, norm_std_fac = 6, start_from = 1000, plot_extra = 0)
    
    norm_method = 0
    
    start_val = 2000
    
    rates = test_spont['rates']
    
    rates2 = rates[start_val:,:,:]
    
    rates3 = rates
    #rates3 = rates2
    
    means1 = np.mean(rates3, axis=0)
    stds1 = np.std(rates3, axis=0)
    
    if not norm_method:
        rates3n = rates3
    elif norm_method == 1:
        rates3n = rates3 - means1
    elif norm_method == 2:
        stds2 = stds1.copy()
        stds2[stds1 == 0] = 1
    
        rates3n = rates3 - means1
        rates3n = rates3n/stds2
    
    T, num_bouts, num_cells = rates3n.shape
    
    rates3n2d = np.reshape(rates3n, (T*num_bouts, num_cells))
    
    pca = PCA();
    pca.fit(rates3n2d)
    
    proj_data = pca.fit_transform(rates3n2d)
    
    comp_out3d = np.reshape(proj_data, (T, num_bouts, num_cells))
    
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


#%% analyze rates during oddball  
# plt.close('all')

if 0:
    # make stim
    num_skip_trials = 200
    
    num_trials = 400
    num_batch = 20
    num_stim_use = 10
    
    num_runs_plot = 5
    plot_trials = 30; #800
    color_ctx = 0;  # 0 = red; 1 = dd
    mark_red = 0
    mark_dd = 1
    
    #dev_stim = np.asarray([20]).astype(int)
    #dev_stim = np.asarray([round(params['num_freq_stim']/2)]).astype(int)
    dev_stim = ((np.arange(num_stim_use)+1)/num_stim_use*params['num_freq_stim']).astype(int)
    #red_stim = np.asarray([24]).astype(int)
    #red_stim = np.asarray([round(params['num_freq_stim']/2)]).astype(int)
    red_stim = ((np.arange(num_stim_use)+1)/num_stim_use*params['num_freq_stim']).astype(int)
    
    # test oddball trials
    trials_test_oddball_freq, trials_test_oddball_ctx, red_dd_seq = f_gen_oddball_seq(dev_stim, red_stim, num_trials, params['dd_frac'], params['num_ctx'], num_batch, can_be_same = False)
    #trials_test_oddball_freq, trials_test_oddball_ctx = f_gen_oddball_seq([5], params['test_oddball_stim'], params['test_trials_in_sample'], params['dd_frac'], params['num_ctx'], params['test_batch_size'], can_be_same = True)

    input_test_oddball, output_test_oddball_freq = f_gen_input_output_from_seq(trials_test_oddball_freq, stim_templates['freq_input'], stim_templates['freq_output'], params)
    _, output_test_oddball_ctx = f_gen_input_output_from_seq(trials_test_oddball_ctx, stim_templates['freq_input'], stim_templates['ctx_output'], params)

    # run test data
    #test_oddball_freq = f_RNN_test(rnn, loss_freq, input_test_oddball, output_test_oddball_freq, params, paradigm='freq')
    test_oddball_ctx = f_RNN_test(rnn, loss_ctx, input_test_oddball, output_test_oddball_ctx, params, paradigm='ctx')
    
    #f_plot_rates2(test_oddball_ctx, 'test_oddball_ctx', num_plot_batches = 5)
    
    #
    
    rates = test_oddball_ctx['rates']
    
    T, num_run, num_cells = rates.shape
    
    num_trials2 = num_trials - num_skip_trials
    
    rates4d = np.reshape(rates, (trial_len, num_trials, num_batch, num_cells), order = 'F')
    
    rates4d2 = rates4d[:,num_skip_trials:,:,:]

    rates2 = np.reshape(rates4d2, (trial_len*num_trials2, num_batch, num_cells), order = 'F')
    
    trials_test_oddball_ctx2 = trials_test_oddball_ctx[num_skip_trials:,:]
    
    # f_plot_rates_only(test_oddball_ctx, 'ctx', num_plot_batches = 1, num_plot_cells = 20, preprocess = True, norm_std_fac = 6, start_from = num_stim_use*trial_len, plot_extra = 0)
    
    rates3n = rates2
    
    T2, _, _ = rates3n.shape
    
    rates3n2d = np.reshape(rates3n, (T2*num_batch, num_cells), order = 'F')
    
    # subtract mean 
    if 1:
        rates_mean = np.mean(rates3n2d, axis=0)
        rates3n2dn = rates3n2d - rates_mean;
    else:
        rates3n2dn = rates3n2d
    
    
    if 0:
        pca = PCA();
        pca.fit(rates3n2dn)
        proj_data = pca.fit_transform(rates3n2d)
        #V2 = pca.components_
        #US = pca.fit_transform(rates3n2dn)
        exp_var = pca.explained_variance_ratio_
    else:
        U, S, V = linalg.svd(rates3n2dn, full_matrices=False)
        #data_back = np.dot(U * S, V)
        #US = U*S
        proj_data = U*S
        Ssq = S*S
        exp_var = Ssq / np.sum(Ssq)
    
    
    comp_out3d = np.reshape(proj_data, (T2, num_batch, num_cells), order = 'F')
    comp_out4d = np.reshape(proj_data, (trial_len, num_trials2, num_batch, num_cells), order = 'F')
    
    
    plt.figure()
    #plt.subplot(1,2,1);
    plt.plot(exp_var, 'o-')
    plt.ylabel('fraction')
    plt.title('Explained Variance'); plt.xlabel('component')
    
    
    plot_patches = range(num_runs_plot)#[0, 1, 5]
    
    
    plot_T = plot_trials*trial_len
    
    plot_pc = [[1, 2], [3, 4], [5, 6], [7, 8]]
    for n_pcpl in range(len(plot_pc)):
        plot_pc2 = plot_pc[n_pcpl]
        plt.figure()
        #plt.subplot(1,2,2);
        for n_bt in plot_patches: #num_bouts
            temp_ob_tr = trials_test_oddball_ctx2[:,n_bt]
            
            if params['num_ctx'] == 1:
                dd_idx = temp_ob_tr == 1
            elif params['num_ctx'] == 2:
                red_idx = temp_ob_tr == 1
                dd_idx = temp_ob_tr == 2
            
            temp_comp4d = comp_out4d[:,:plot_trials,n_bt,:]
            
            plt.plot(comp_out3d[:plot_T, n_bt, plot_pc2[0]-1], comp_out3d[:plot_T, n_bt, plot_pc2[1]-1], color=colors1[red_dd_seq[color_ctx,n_bt]-1,:])
            
            if mark_red:
                if params['num_ctx'] == 2:
                    plt.plot(temp_comp4d[4:15,:,plot_pc2[0]-1][:,red_idx[:plot_trials]], temp_comp4d[4:15,:,plot_pc2[1]-1][:,red_idx[:plot_trials]], '.b')
                    plt.plot(temp_comp4d[4,:,plot_pc2[0]-1][red_idx[:plot_trials]], temp_comp4d[4,:,plot_pc2[1]-1][red_idx[:plot_trials]], 'ob')
            
            if mark_dd: 
                plt.plot(temp_comp4d[4:15,:,plot_pc2[0]-1][:,dd_idx[:plot_trials]], temp_comp4d[4:15,:,plot_pc2[1]-1][:,dd_idx[:plot_trials]], '.r')
                plt.plot(temp_comp4d[4,:,plot_pc2[0]-1][dd_idx[:plot_trials]], temp_comp4d[4,:,plot_pc2[1]-1][dd_idx[:plot_trials]], 'or')
            

            
        plt.title('PCA components'); plt.xlabel('PC%d' % plot_pc2[0]); plt.ylabel('PC%d' % plot_pc2[1])
        
    
    if 0:
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


#%% analyze distances const dd/red
# plt.close('all')

if 0:
    num_skip_trials = 100
    
    variab_tr_idx = 1;   # 1 = dd 0 = red
    plot_tr_idx = 1;
    
    num_trials = 400
    num_batch = 200
    num_stim_use = 20
    
    
    #dev_stim = (np.arange(0,num_stim_use)/num_stim_use*params['num_freq_stim']).astype(int)
    #red_stim = np.asarray([24]).astype(int)
    
    #dev_stim = np.asarray([20]).astype(int)
    #dev_stim = np.asarray([round(params['num_freq_stim']/2)]).astype(int)
    dev_stim = ((np.arange(num_stim_use)+1)/num_stim_use*params['num_freq_stim']).astype(int)
    red_stim = np.asarray([24]).astype(int)
    #red_stim = np.asarray([round(params['num_freq_stim']/2)]).astype(int)
    #red_stim = ((np.arange(num_stim_use)+1)/num_stim_use*params['num_freq_stim']).astype(int)
    
    var_seq = dev_stim

    # test oddball trials
    trials_test_oddball_freq, trials_test_oddball_ctx, red_dd_seq = f_gen_oddball_seq(dev_stim, red_stim, num_trials, params['dd_frac'], params['num_ctx'], num_batch, can_be_same = False)
    #trials_test_oddball_freq, trials_test_oddball_ctx = f_gen_oddball_seq([5], params['test_oddball_stim'], params['test_trials_in_sample'], params['dd_frac'], params['num_ctx'], params['test_batch_size'], can_be_same = True)

    input_test_oddball, output_test_oddball_freq = f_gen_input_output_from_seq(trials_test_oddball_freq, stim_templates['freq_input'], stim_templates['freq_output'], params)
    _, output_test_oddball_ctx = f_gen_input_output_from_seq(trials_test_oddball_ctx, stim_templates['freq_input'], stim_templates['ctx_output'], params)

    # run test data
    #test_oddball_freq = f_RNN_test(rnn, loss_freq, input_test_oddball, output_test_oddball_freq, params, paradigm='freq')
    test_oddball_ctx = f_RNN_test(rnn, loss_ctx, input_test_oddball, output_test_oddball_ctx, params, paradigm='ctx')
    
    
    rates = test_oddball_ctx['rates'] #(8000, 100, 25)
    
    num_t, num_run, num_cells = rates.shape
   
    rates4d = np.reshape(rates, (num_trials, trial_len, num_batch, num_cells), order = 'C')
    
    
    rates4d_cut = rates4d[num_skip_trials:,:,:,:]
    num_tr2 = num_trials - num_skip_trials
    
    trials_test_oddball_ctx_cut = trials_test_oddball_ctx[num_skip_trials:,:]
    
    
    trial_ave_rd = np.zeros((2, trial_len, num_batch, num_cells))
    
    for n_run in range(num_batch):
        idx1 = trials_test_oddball_ctx_cut[:,n_run] == 1
        trial_ave_rd[0,:,n_run,:] = np.mean(rates4d_cut[idx1,:,n_run,:], axis=0)
        
        idx1 = trials_test_oddball_ctx_cut[:,n_run] == 2
        trial_ave_rd[1,:,n_run,:] = np.mean(rates4d_cut[idx1,:,n_run,:], axis=0)
    


    cur_tr = dev_stim[round(len(dev_stim)/2)]
    idx_cur = red_dd_seq[variab_tr_idx,:,0] == cur_tr
    base_resp = np.mean(trial_ave_rd[plot_tr_idx,:,idx_cur,:], axis=0)
    base_resp1d = np.reshape(base_resp, (trial_len*num_cells), order='F')
    
    num_var = len(var_seq)
    
    dist_all = np.zeros((num_var))
    dist_all_cos = np.zeros((num_var))
    
    for n_tr in range(num_var):
        idx1 = red_dd_seq[variab_tr_idx,:,0] == var_seq[n_tr]
        temp1 = np.mean(trial_ave_rd[plot_tr_idx,:,idx1,:], axis=0)
        temp1_1d = np.reshape(temp1, (trial_len*num_cells), order='F')
        
        dist_all[n_tr] = pdist([base_resp1d,temp1_1d], metric='euclidean')
        dist_all_cos[n_tr] = pdist([base_resp1d,temp1_1d], metric='cosine')
    
    
    
    plt.figure()
    plt.plot(var_seq, dist_all)
    plt.ylabel('euclidean dist')
    if variab_tr_idx:
        plt.title('const red, var dd; ref tr = %d' % cur_tr)
        plt.xlabel('dd stim')
    else:
        plt.title('const dd, var red; ref tr = %d' % cur_tr)
        plt.xlabel('red stim')
    
    plt.figure()
    plt.plot(var_seq, dist_all_cos)
    plt.ylabel('cosine dist')
    plt.xlabel('red stim')
    if variab_tr_idx:
        plt.title('const red, var dd; ref tr = %d' % cur_tr)
        plt.xlabel('dd stim')
    else:
        plt.title('const dd, var red; ref tr = %d' % cur_tr)
        plt.xlabel('red stim')


#%% analyzing controls

if 0:
    
    #f_plot_rates_only(test_oddball_ctx, 'ctx', num_plot_batches = 1, num_plot_cells = 20, preprocess = True, norm_std_fac = 6, start_from = 1000, plot_extra = 0)
    
    rates = test_cont_freq['rates']
    
    start_val = 2000
    
    trial_len = (params['stim_duration'] + params['isi_duration']) / params['dt']
    
    rates2 = rates[start_val:,:,:]
    
    
    #rates3n = rates
    rates3n = rates2
    
    T, num_bouts, num_cells = rates3n.shape
    
    rates3n2d = np.reshape(rates3n, (T*num_bouts, num_cells), order = 'F')
    
    
    rates_mean = np.mean(rates3n2d, axis=0)
    
    #rates3n2dn = rates3n2d
    rates3n2dn = rates3n2d - rates_mean;
    
    
    if 0:
        pca = PCA();
        pca.fit(rates3n2dn)
        proj_data = pca.fit_transform(rates3n2d)
        #V2 = pca.components_
        #US = pca.fit_transform(rates3n2dn)
        exp_var = pca.explained_variance_ratio_
    else:
        U, S, V = linalg.svd(rates3n2dn, full_matrices=False)
        #data_back = np.dot(U * S, V)
        #US = U*S
        proj_data = U*S
        Ssq = S*S
        exp_var = Ssq / np.sum(Ssq)
    
    
    comp_out3d = np.reshape(proj_data, (T, num_bouts, num_cells), order = 'F')
    
    
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
    
    
    
    
    trials_test_oddball_ctx_cut = trials_test_oddball_ctx[10:,:]
    
    
    trial_ave_rd = np.zeros((2, trial_len, num_run, num_cells))
    
    for n_run in range(num_run):
        idx1 = trials_test_oddball_ctx_cut[:,n_run] == 1
        trial_ave_rd[0,:,n_run,:] = np.mean(rates4d_cut[idx1,:,n_run,:], axis=0)
        
        idx1 = trials_test_oddball_ctx_cut[:,n_run] == 2
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


res_linkage = linkage(w_r, method='average')

N = len(w_r)
res_ord = seriation(res_linkage,N, N + N -2)

plt.figure()
plt.imshow(w_r[res_ord,:][:,res_ord])
plt.colorbar()
plt.title('Recurrents')
plt.xlabel('W recurrent sorted')
plt.ylabel('W recurrent sorted')

if 0:
    print('Saving RNN %s' % fname_RNN_save)
    plt.savefig(path1 + '/RNN_data/' + fname_RNN_save + 'wrsort_fig.png', dpi=1200)


#%%

cs_ord = 1- squareform(pdist(rates_all[res_ord], metric='cosine'));

plt.figure()
plt.imshow(cs_ord)
plt.title('cosine similarity sorted')



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





#%%
pca = PCA();
pca.fit(rates_all[:,:,0])

#%%
plt.figure()
plt.subplot(1,2,1);
plt.plot(pca.explained_variance_, 'o')
plt.title('Explained Variance'); plt.xlabel('component')

plt.subplot(1,2,2);
plt.plot(pca.components_[0,:], pca.components_[1,:])
plt.title('PCA components'); plt.xlabel('PC1'); plt.ylabel('PC2')

#%%
flat_dist_met = pdist(rates_all[:,:,0], metric='cosine');
cs = 1- squareform(flat_dist_met);
res_linkage = linkage(flat_dist_met, method='average')

N = len(cs)
res_ord = seriation(res_linkage,N, N + N -2)
    
#%%

cs_ord = 1- squareform(pdist(rates_all[res_ord,:,0], metric='cosine'));

plt.figure()
plt.imshow(cs_ord)
plt.title('cosine similarity sorted')

#%% cell tuning

#%% save data

data_save = {"rates_all": rates_all, "loss_all_smooth": loss_all_smooth,
             "input_sig": np.asarray(input_sig.data), "target": np.asarray(target.data),
             "output": outputs_all, "g": g, "dt": dt, "tau": tau, "hidden_size": hidden_size,
             'ti': ti,
             'h2h_weight': np.asarray(rnn.h2h.weight.data), 'train_RNN': train_RNN,
             'i2h_weight': np.asarray(rnn.i2h.weight.data),
             'h2o_weight': np.asarray(rnn.h2o.weight.data),
             'fname_input': fname_input}

#save_fname = 'rnn_out_8_25_21_1_complex_g_tau10_5cycles.mat'

save_fname = 'rnn_out_12_31_21_10tones_200reps_notrain.mat'

savemat(fpath+ save_fname, data_save)



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


