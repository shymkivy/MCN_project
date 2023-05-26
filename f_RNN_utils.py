# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 14:08:27 2023

@author: ys2605
"""

import numpy as np
import matplotlib.pyplot as plt
   
from scipy import signal
    
#%%
def f_gen_stim_output_templates(params):
    # generate stim templates
    
    num_stim = params['num_freq_stim']
    stim_duration = params['stim_duration']
    isi_suration = params['isi_duration']
    input_size = params['input_size']
    dt = params['dt']
    stim_t_std = params['stim_t_std']
    
    output_size = num_stim + 1

    stim_bins = np.round(stim_duration/dt).astype(int)
    isi_bins = np.round(isi_suration/dt).astype(int)
    stim_loc = np.round(np.linspace(0, input_size, num_stim+2))[1:-1].astype(int)

    isi_lead = np.floor(isi_bins/2).astype(int)

    gaus_x_range = np.round(4*stim_t_std).astype(int)
    gx = np.arange(-gaus_x_range, (gaus_x_range+1))
    gaus_t = np.exp(-(gx/stim_t_std)**2/2).reshape((gaus_x_range*2+1,1))
    gaus_t = gaus_t/np.sum(gaus_t)

    #plt.figure()
    #plt.plot(gx, gaus_t)
    # (num_stim_inputs x num_t x num_trial_types)
    stim_temp_all = np.zeros((input_size, stim_bins + isi_bins, output_size))
    # (num_stim_outputs x num_t x num_trial_types)
    out_temp_all = np.zeros((output_size, stim_bins + isi_bins, output_size))
    
    # set quiet stim as zero
    out_temp_all[0, :, 0] = 1

    for n_st in range(num_stim):
        stim_temp = np.zeros((input_size, stim_bins + isi_bins))
        stim_temp[stim_loc[n_st], isi_lead:(isi_lead+stim_bins)] = 1
        
        if stim_t_std:
            stim_temp = signal.convolve(stim_temp, gaus_t, mode="same")
        
        stim_temp_all[:,:,n_st+1] = stim_temp
        
        out_temp = np.zeros((output_size, stim_bins + isi_bins))
        out_temp[n_st+1, isi_lead:(isi_lead+stim_bins)] = 1
        out_temp[0, :isi_lead] = 1
        out_temp[0, (isi_lead+stim_bins):] = 1
        
        out_temp_all[:,:,n_st+1] = out_temp
        
    for n_st in range(output_size):
        if params['plot_deets']:
            plt.figure()
            plt.subplot(121)
            plt.imshow(stim_temp_all[:,:,n_st])
            plt.subplot(122)
            plt.imshow(out_temp_all[:,:,n_st])
            plt.suptitle('stim %d' % n_st)

    return stim_temp_all, out_temp_all

#%%
def f_gen_stim_output_templates_thin(params):
    # generate stim templates
    
    num_stim = params['num_freq_stim']
    stim_duration = params['stim_duration']
    isi_suration = params['isi_duration']
    input_size = params['input_size']
    dt = params['dt']
    stim_t_std = params['stim_t_std']
    
    output_size = num_stim + 1

    stim_loc = np.round(np.linspace(0, input_size, num_stim+2))[1:-1].astype(int)

    gaus_x_range = np.round(4*stim_t_std).astype(int)
    gx = np.arange(-gaus_x_range, (gaus_x_range+1))
    gaus_t = np.exp(-(gx/stim_t_std)**2/2)
    gaus_t = gaus_t/np.sum(gaus_t)

    #plt.figure()
    #plt.plot(gx, gaus_t)
    # (num_stim_inputs x num_t x num_trial_types)
    stim_temp_all = np.zeros((input_size, output_size))
    # (num_stim_outputs x num_t x num_trial_types)
    out_temp_all = np.zeros((output_size, output_size))
    
    # set quiet stim as zero
    out_temp_all[0, 0] = 1
    
    for n_st in range(num_stim):
        stim_temp = np.zeros((input_size))
        stim_temp[stim_loc[n_st]] = 1
        
        #print(stim_temp)
        
        if stim_t_std:
            stim_temp = signal.convolve(stim_temp, gaus_t, mode="same")
        
        #print(stim_temp)
        stim_temp_all[:,n_st+1] = stim_temp
        
        out_temp = np.zeros((output_size))
        
        out_temp[n_st+1] = 1

        out_temp_all[:,n_st+1] = out_temp
        
    for n_st in range(output_size):
        if params['plot_deets']:
            plt.figure()
            plt.subplot(121)
            plt.imshow(stim_temp_all[:,n_st].reshape((input_size,1)))
            plt.subplot(122)
            plt.imshow(out_temp_all[:,n_st].reshape((output_size,1)))
            plt.suptitle('stim %d' % n_st)

    return stim_temp_all, out_temp_all


#%%

def f_gen_cont_seq(num_stim, num_trials, batch_size = 1, num_samples = 1):
    
    trials_out = np.ceil(np.random.random(num_trials*batch_size*num_samples)*num_stim).astype(int).reshape((num_trials, batch_size, num_samples))
    
    return trials_out.squeeze()

def f_gen_oddball_seq(oddball_stim, num_trials, dd_frac, batch_size = 1, num_samples = 1):
    
    trials_oddball_freq = np.zeros((num_trials, batch_size* num_samples)).astype(int)
    trials_oddball_ctx = np.zeros((num_trials, batch_size* num_samples)).astype(int)
    
    
    # set dd trials (coin flip)
    idx_dd = np.less_equal(np.random.random((num_trials, batch_size * num_samples)), dd_frac)

    for n_samp in range(num_samples*batch_size):

        idx_dd2 = idx_dd[:, n_samp]
        
        stim_rd = np.random.choice(oddball_stim, size=2, replace=False)
        
        trials_oddball_freq[idx_dd2, n_samp] = stim_rd[1]
        trials_oddball_freq[~idx_dd2, n_samp] = stim_rd[0]
        
        trials_oddball_ctx[idx_dd2, n_samp] = 2
        trials_oddball_ctx[~idx_dd2, n_samp] = 1
        
    trials_oddball_freq2 = trials_oddball_freq.reshape((num_trials, batch_size, num_samples), order='F')
    trials_oddball_ctx2 = trials_oddball_ctx.reshape((num_trials, batch_size, num_samples), order='F')
    
    return trials_oddball_freq2.squeeze(), trials_oddball_ctx2.squeeze() 

#%%

def f_gen_input_output_from_seq(input_trials, stim_templates, output_templates, params):
    
    input_noise_std = params['input_noise_std']
    
    #input_size = params['input_size']
    #trial_len = round((params['stim_duration'] + params['isi_duration'])/params['dt'])
    #output_size = params['num_freq_stim'] + 1
    
    input_size, trial_len, _ = stim_templates.shape
    output_size, _, _ = output_templates.shape
    
    shape1 = input_trials.shape;
    num_trials = shape1[0]
    
    T = trial_len * num_trials
    
    num_samp = 1
    num_batch = 1;
    if len(shape1) > 2:
        num_batch = shape1[1]
        num_samp = shape1[2]
    elif len(shape1) > 1:
        num_batch = shape1[1]

    input_trials = input_trials.reshape((num_trials,num_batch, num_samp))
    
    input_shape = [input_size, T, num_batch, num_samp]
    output_shape = [output_size, T, num_batch, num_samp]
    
    input_mat = stim_templates[:,:,input_trials].reshape(input_shape, order='F') + np.random.normal(0,input_noise_std, input_shape)
    input_mat = input_mat - np.mean(input_mat)
    input_mat/np.std(input_mat)
    
    output_mat = output_templates[:,:,input_trials].reshape(output_shape, order='F')
    
    return input_mat.squeeze(), output_mat.squeeze()

#%%

def f_plot_rates2(rates, num_cells_plot = 999999):
    
    spacing = np.ceil(np.max(rates) - np.min(rates))  
    num_cells = rates.shape[0]
    
    offsets = np.reshape(np.linspace(1, (num_cells)*spacing, num_cells), (num_cells, 1));
    
    num_cells_plot = np.min((num_cells_plot, num_cells))
    
    rates2 = rates + offsets
    
    plt.figure()
    plt.plot(rates2[:num_cells_plot,:].T)
    plt.ylabel('cells')
    plt.xlabel('time')
    

#%%

# def f_load_RNN_inputs():
    
#     fpath = path1 + '/RNN_data/'

#     #fname = 'sim_spec_1_stim_8_24_21.mat'
#     #fname = 'sim_spec_10complex_200rep_stim_8_25_21_1.mat'
#     #fname = 'sim_spec_10tones_200rep_stim_8_25_21_1.mat'

#     fname_input = 'sim_spec_10tones_100reps_0.5isi_50dt_1_1_22_23h_17m.mat';
    
#     data_mat = loadmat(fpath+fname_input)
    
#     data1 = data_mat['spec_data']
#     print(data1.dtype)
    
#     input_mat = data1[0,0]['spec_cut'];
#     fr_cut = data1[0, 0]['fr_cut'];
#     ti = data1[0, 0]['ti'];
#     voc_seq = data1[0, 0]['voc_seq'];
#     num_voc = data1[0, 0]['num_voc'];
#     output_mat = data1[0, 0]['output_mat'];
#     output_mat_delayed = data1[0, 0]['output_mat_delayed'];
#     input_T = data1[0, 0]['ti'][0];
    
#     input_size = input_mat.shape[0];    # number of freqs in spectrogram
#     output_size = output_mat.shape[0];  # number of target output categories 
#     T = input_mat.shape[1];             # time steps of inputs and target outputs
#     dt = input_T[1] - input_T[0];        #1;
    
#     plt.figure()
#     plt.imshow(input_mat)
    
    
#     tau = .5;              # for to bin stim
#     alpha = dt/tau;         # 
    
#     plt.figure()
#     plt.plot(input_mat.std(axis=0))
