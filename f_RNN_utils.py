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
    
    num_stim = params['num_stim']
    stim_duration = params['stim_duration']
    isi_suration = params['isi_suration']
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
    stim_temp_all = np.zeros((input_size, stim_bins + isi_bins, num_stim))
    # (num_stim_outputs x num_t x num_trial_types)
    out_temp_all = np.zeros((output_size, stim_bins + isi_bins, num_stim))
    for n_st in range(num_stim):
        stim_temp = np.zeros((input_size, stim_bins + isi_bins))
        stim_temp[stim_loc[n_st], isi_lead:(isi_lead+stim_bins)] = 1
        
        if stim_t_std:
            stim_temp = signal.convolve(stim_temp, gaus_t, mode="same")
        
        stim_temp_all[:,:,n_st] = stim_temp
        
        out_temp = np.zeros((output_size, stim_bins + isi_bins))
        out_temp[n_st+1, isi_lead:(isi_lead+stim_bins)] = 1
        out_temp[0, :isi_lead] = 1
        out_temp[0, (isi_lead+stim_bins):] = 1
        
        out_temp_all[:,:,n_st] = out_temp
        
        if params['plot_deets']:
            plt.figure()
            plt.subplot(121)
            plt.imshow(stim_temp)
            plt.subplot(122)
            plt.imshow(out_temp)
            plt.suptitle('stim %d' % n_st)

    return stim_temp_all, out_temp_all

#%%
def f_gen_stim_output_templates_thin(params):
    # generate stim templates
    
    num_stim = params['num_stim']
    stim_duration = params['stim_duration']
    isi_suration = params['isi_suration']
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
    stim_temp_all = np.zeros((input_size, stim_bins + isi_bins, num_stim))
    # (num_stim_outputs x num_t x num_trial_types)
    out_temp_all = np.zeros((output_size, stim_bins + isi_bins, num_stim))
    for n_st in range(num_stim):
        stim_temp = np.zeros((input_size, stim_bins + isi_bins))
        stim_temp[stim_loc[n_st], isi_lead:(isi_lead+stim_bins)] = 1
        
        if stim_t_std:
            stim_temp = signal.convolve(stim_temp, gaus_t, mode="same")
        
        stim_temp_all[:,:,n_st] = stim_temp
        
        out_temp = np.zeros((output_size, stim_bins + isi_bins))
        out_temp[n_st+1, isi_lead:(isi_lead+stim_bins)] = 1
        out_temp[0, :isi_lead] = 1
        out_temp[0, (isi_lead+stim_bins):] = 1
        
        out_temp_all[:,:,n_st] = out_temp
        
        if params['plot_deets']:
            plt.figure()
            plt.subplot(121)
            plt.imshow(stim_temp)
            plt.subplot(122)
            plt.imshow(out_temp)
            plt.suptitle('stim %d' % n_st)

    return stim_temp_all, out_temp_all
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
