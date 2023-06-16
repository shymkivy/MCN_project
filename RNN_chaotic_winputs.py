# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 15:33:58 2021

@author: Administrator
"""

import sys

#path1 = 'C:/Users/yuriy/Desktop/stuff/RNN_stuff/'
path1 = 'C:/Users/ys2605/Desktop/stuff/RNN_stuff/'

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
from random import sample, random
import math

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy import signal
from scipy.io import loadmat, savemat
import skimage.io

from datetime import datetime

#%% params
compute_loss = 1
train_RNN = 1
save_RNN = 1
load_RNN = 0
plot_deets = 1

#%% input params

params = {'train_type':                     'oddball2',     #   oddball2, freq2  standard, linear, oddball, freq_oddball,
          
          'stim_duration':                  0.5,
          'isi_duration':                   0.5,
          'num_freq_stim':                  10,
          'num_ctx':                        2,
          'oddball_stim':                   [2, 5, 8],
          'dd_frac':                        0.1,
          'dt':                             0.05,
          
          'train_batch_size':               64,
          'train_trials_in_sample':         20,
          'train_num_samples_freq':         1000,
          'train_num_samples_ctx':          20000,

          'train_repeats_per_samp':         1,
          'train_reinit_rate':              0,
          
          
          'test_batch_size':                100,
          'test_trials_in_sample':          400,
          
          'input_size':                     50,
          'hidden_size':                    250,            # number of RNN neurons
          'g':                              1,  # 1            # recurrent connection strength 
          'tau':                            0.5,
          'learning_rate':                  0.001,           # 0.005
          'activation':                     'ReLU',
          
          
          'stim_t_std':                     3,              # 3 or 0
          'input_noise_std':                1/100,
          
          
          'plot_deets':                     0,
          }

now1 = datetime.now()

name_tag = '%s_%dtrain_samp_%s_%dtrials_%dstim_%dbatch_%.4flr_%d_%d_%d_%dh_%dm' % (params['train_type'], 
            params['train_num_samples_ctx'], params['activation'], params['train_trials_in_sample'], params['num_freq_stim'], 
            params['train_batch_size'], params['learning_rate'],
            now1.year, now1.month, now1.day, now1.hour, now1.minute)

#%%

#fname_RNN_load = 'test_20k_std3'
#fname_RNN_load = '50k_20stim_std3';
fname_RNN_load = 'oddball2_20000train_samp_20trials_10stim_64batch_0.0010lr_2023_5_28_13h_15m_RNN'

#fname_RNN_save = 'test_50k_std4'
#fname_RNN_save = '50k_20stim_std3'
fname_RNN_save = name_tag

#%% generate train data

#plt.close('all')

# generate stim templates

stim_templates = {}
stim_templates['freq_input'], stim_templates['freq_output'] = f_gen_stim_output_templates(params)
stim_templates['ctx_output'] = stim_templates['freq_output'][:3,:,:3]


# shape (seq_len, batch_size, input_size/output_size, num_samples)
# train control trials 
#trials_train_cont = f_gen_cont_seq(params['num_freq_stim'], params['train_trials_in_sample'], params['train_batch_size'], params['train_num_samples_freq'])
#input_train_cont, output_train_cont = f_gen_input_output_from_seq(trials_train_cont, stim_templates['freq_input'], stim_templates['freq_output'], params)


#trials_train_cont2 = f_gen_cont_seq(params['num_freq_stim'], params['train_trials_in_sample'], params['train_batch_size'], 1)
#input_train_cont2, output_train_cont2 = f_gen_input_output_from_seq(trials_train_cont2, stim_templates['freq_input'], stim_templates['freq_output'], params)


# train oddball trials 
#trials_train_oddball_freq, trials_train_oddball_ctx = f_gen_oddball_seq(params['oddball_stim'], params['train_trials_in_sample'], params['dd_frac'], params['train_batch_size'], params['train_num_samples_ctx'])

#input_train_oddball, output_train_oddball_freq = f_gen_input_output_from_seq(trials_train_oddball_freq, stim_templates['freq_input'], stim_templates['freq_output'], params)
#_, output_train_oddball_ctx = f_gen_input_output_from_seq(trials_train_oddball_ctx, stim_templates['freq_input'], stim_templates['ctx_output'], params)




#%% initialize RNN 

output_size = params['num_freq_stim'] + 1
output_size_ctx = params['num_ctx'] + 1
hidden_size = params['hidden_size'];
alpha = params['dt']/params['tau'];         

rnn = RNN_chaotic(params['input_size'], params['hidden_size'], output_size, output_size_ctx, alpha, activation=params['activation'])
rnn.init_weights(params['g'])

#%%
#loss = nn.NLLLoss()
loss_freq = nn.CrossEntropyLoss()
loss_ctx = nn.CrossEntropyLoss(weight = torch.tensor([0.1, 0.1, 0.9]))  #1e-10

train_out = {}     # initialize outputs, so they are saved when process breaks

#%%
if load_RNN:
    print('Loading RNN %s' % fname_RNN_load)
    rnn.load_state_dict(torch.load(path1 + '/RNN_data/' + fname_RNN_load))

#%%
if train_RNN:
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

if train_RNN:
    #plt.close('all')
    sm_bin = 50#round(1/params['dt'])*50;
    #trial_len = out_temp_all.shape[1]
    kernel = np.ones(sm_bin)/sm_bin
    
    loss_train = np.asarray(train_out['loss'])
    #loss_train = np.asarray(train_out_cont['loss']).T.flatten()
    loss_train_cont_sm = np.convolve(loss_train, kernel, mode='valid')
    loss_x_sm = np.arange(len(loss_train_cont_sm))+sm_bin/2 #/(trial_len)
    loss_x_raw = np.arange(len(loss_train)) #/(trial_len)
    
    
    plt.figure()
    plt.semilogy(loss_x_raw, loss_train)
    plt.semilogy(loss_x_sm, loss_train_cont_sm)
    plt.legend(('train', 'train smoothed'))
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title(name_tag)



# f_plot_rates(rates_all[:,:, 1], 10)
 #%%
if save_RNN and train_RNN:
    print('Saving RNN %s' % fname_RNN_save)
    torch.save(rnn.state_dict(), path1 + '/RNN_data/' + fname_RNN_save  + '_RNN')
    np.save(path1 + '/RNN_data/' + fname_RNN_save + '_params.npy', params) 
    np.save(path1 + '/RNN_data/' + fname_RNN_save + '_train_out.npy', train_out) 
    
    plt.savefig(path1 + '/RNN_data/' + fname_RNN_save + '_fig.png', dpi=1200)
    
    
#%% gen test data

# test control trials
trials_test_cont = f_gen_cont_seq(params['num_freq_stim'], params['test_trials_in_sample'], params['test_batch_size'], 1)
input_test_cont, output_test_cont = f_gen_input_output_from_seq(trials_test_cont, stim_templates['freq_input'], stim_templates['freq_output'], params)


# test oddball trials
trials_test_oddball_freq, trials_test_oddball_ctx = f_gen_oddball_seq(params['oddball_stim'], params['test_trials_in_sample'], params['dd_frac'], params['test_batch_size'])

input_test_oddball, output_test_oddball_freq = f_gen_input_output_from_seq(trials_test_oddball_freq, stim_templates['freq_input'], stim_templates['freq_output'], params)
_, output_test_oddball_ctx = f_gen_input_output_from_seq(trials_test_oddball_ctx, stim_templates['freq_input'], stim_templates['ctx_output'], params)


if params['plot_deets']:
    input_plot = input_test_cont
    output_plot = output_test_cont
    
    if len(input_plot.shape) > 2:
        input_plot = input_plot[:,:,0]
        output_plot = output_plot[:,:,0]
    
    spec3 = gridspec.GridSpec(ncols=1, nrows=3, height_ratios=[1, 1, 1])
    plt.figure()
    ax1 = plt.subplot(spec3[0])
    ax1.plot(input_plot.std(axis=0))
    plt.title('inputs std; %d inputs' % params['input_size'])
    plt.subplot(spec3[1], sharex=ax1)
    plt.plot(input_plot.mean(axis=0))
    plt.title('inputs mean')
    plt.subplot(spec3[2], sharex=ax1)
    plt.plot(input_plot.max(axis=0))
    plt.title('inputs max')
    
    spec2 = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[1, 1])
    plt.figure()
    ax1 = plt.subplot(spec2[0])
    ax1.imshow(input_plot, aspect="auto")
    plt.title('inputs; %d stim; %d intups; std=%.1f' % (params['num_freq_stim'], params['input_size'], params['stim_t_std']))
    plt.subplot(spec2[1], sharex=ax1)
    plt.imshow(output_plot, aspect="auto")
    plt.title('outputs')
    
    plt.figure()
    plt.plot(np.mean(input_plot, axis=1))
    plt.title('mean spectrogram across time')
    plt.xlabel('inputs')
    plt.ylabel('mean power')
    
    
#%% test
test_cont_freq = f_RNN_test(rnn, loss_freq, input_test_cont, output_test_cont, params)

#%% test oddball
test_oddball_freq = f_RNN_test(rnn, loss_freq, input_test_oddball, output_test_oddball_freq, params)

test_oddball_ctx = f_RNN_test_ctx(rnn, loss_ctx, input_test_oddball, output_test_oddball_ctx, params)

#%% test oddball

#train_oddball = f_RNN_test(rnn, loss_ctx, input_train_oddball_freq, output_train_oddball_ctx, params)


#%%

# plt.close('all')


if train_RNN:
    sm_bin = 50#round(1/params['dt'])*50;
    #trial_len = out_temp_all.shape[1]
    kernel = np.ones(sm_bin)/sm_bin
    
    loss_train = np.asarray(train_out['loss'])# .T.flatten()
    loss_train_cont_sm = np.convolve(loss_train, kernel, mode='valid')
    loss_x = np.arange(len(loss_train_cont_sm)) + sm_bin/2 #/(trial_len)
    loss_x_raw = np.arange(len(loss_train)) #/(trial_len)
    
    loss_test_cont = np.asarray(test_cont_freq['loss'])
    loss_test_cont_sm = np.convolve(loss_test_cont, kernel, mode='valid')
    loss_x_test_raw = np.arange(len(loss_test_cont))
    loss_x_test = np.arange(len(loss_test_cont_sm))  + sm_bin/2#/(trial_len)
    
    loss_test_ob = np.asarray(test_oddball_ctx['loss'])
    loss_test_ob_sm = np.convolve(loss_test_ob, kernel, mode='valid')
    loss_x_test_ob_raw = np.arange(len(loss_test_ob))
    loss_x_test_ob = np.arange(len(loss_test_ob_sm))  + sm_bin/2#/(trial_len)
    
    
    plt.figure()
    plt.semilogy(loss_x_raw, loss_train, 'lightblue')
    plt.semilogy(loss_x, loss_train_cont_sm, 'darkblue')
    plt.semilogy(loss_x_test_raw, loss_test_cont, 'lightgreen')
    plt.semilogy(loss_x_test, loss_test_cont_sm, 'darkgreen')
    plt.semilogy(loss_x_test_ob_raw, loss_test_ob, 'pink')
    plt.semilogy(loss_x_test_ob, loss_test_ob_sm, 'darkred')
    plt.legend(('train', 'train sm', 'test cont', 'test cont sm', 'test oddball', 'test oddball dm'))
    plt.xlabel('trials')
    plt.ylabel('NLL loss')
    plt.title(fname_RNN_save)
    


#%% add loss of final pass to train data

if train_RNN:
    T, batch_size, num_neurons = train_out['rates'].shape
    
    output2 = torch.tensor(train_out['output'])
    target2 =  torch.tensor(train_out['target_idx'])
    train_out['lossT'] = np.zeros((T, batch_size))
    for n_t in range(T):
        for n_bt2 in range(batch_size):
            train_out['lossT'][n_t, n_bt2] = loss_ctx(output2[n_t, n_bt2, :], target2[n_t, n_bt2].long()).item()
    
#%% plot train data
    
if train_RNN:   
    f_plot_rates2(train_out_cont, 'train', num_plot_batches = 5)


#%%

f_plot_rates2(test_cont_freq, 'test_cont', num_plot_batches = 5)

f_plot_rates2(test_oddball_freq, 'test_oddball_freq', num_plot_batches = 5)

f_plot_rates2(test_oddball_ctx, 'test_oddball_ctx', num_plot_batches = 5)


#%%
f_plot_rates(test_cont, input_test_cont, output_test_cont, 'test cont')

f_plot_rates(test_oddball, input_test_oddball, output_test_oddball, 'test oddball')

#%%

f_plot_rates(test_oddball_ctx, input_test_oddball, output_test_oddball_freq, 'test oddball')

f_plot_rates_ctx(test_oddball_ctx, input_test_oddball2, output_test_oddball_ctx2, 'test oddball')




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



#%%
# W is inside i2h.weight

# i2h = nn.Linear(input_size, hidden_size)
# h2h = nn.Linear(hidden_size, hidden_size)
# h2o = nn.Linear(hidden_size, output_size)
# #softmax = nn.LogSoftmax(dim=1)
# tanh1 = nn.Tanh();
# softmax1 = nn.LogSoftmax(dim=1);

#%% fix weights

# wh2h = torch.empty(hidden_size, hidden_size)
# nn.init.normal_(wh2h, mean=0.0, std = 1)

# std1 = g/np.sqrt(hidden_size);

# wh2h = wh2h - np.mean(wh2h.detach().numpy());
# wh2h = wh2h * std1;

# h2h.weight.data = wh2h;

# wi2h = torch.empty(hidden_size, input_size)
# nn.init.normal_(wi2h, mean=0.0, std = 1)

# std1 = g/np.sqrt(hidden_size);

# wi2h = wi2h - np.mean(wi2h.detach().numpy());
# wi2h = wi2h * std1;

# i2h.weight.data = wi2h;
#%%
# rates_all = np.zeros((hidden_size, T));
# outputs_all = np.zeros((output_size, T));

# rate = torch.empty(1, hidden_size);

# nn.init.uniform_(rate, a=0, b=1)

# rates_all[:,0] = rate.detach().numpy()[0,:];

# for n_t in range(T-1):
    
#     rate_new = tanh1(i2h(input_sig[:,n_t]) + h2h(rate))
    
#     rate_new = (1-alpha)*rate + alpha*rate_new
    
#     rates_all[:,n_t+1] = rate_new.detach().numpy()[0,:];
    
#     rate = rate_new;
    
#     output = softmax1(h2o(rate_new))
    
#     outputs_all[:,n_t+1] = output.detach().numpy()[0,:];
    
#     target2 = torch.argmax(target[:,n_t]) * torch.ones(1) # torch.tensor()
    
#     loss2 = loss(output, target2.long())
    
    
    
    
    

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


