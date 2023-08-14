# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 14:08:27 2023

@author: ys2605
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import colors
   
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
    
    if num_samples == 1:
        trials_out = trials_out[:,:,0]
    
    return trials_out

def f_gen_oddball_seq(dev_stim, red_stim, num_trials, dd_frac, batch_size = 1, num_samples = 1, can_be_same = False, can_have_no_dd = False):
    
    dev_stim2 = np.asarray(dev_stim)
    red_stim2 = np.asarray(red_stim)
    
    trials_oddball_freq = np.zeros((num_trials, batch_size* num_samples)).astype(int)
    trials_oddball_ctx = np.zeros((num_trials, batch_size* num_samples)).astype(int)
    
    red_dd_freq = np.zeros((2, batch_size* num_samples)).astype(int)
    
    
    # set dd trials (coin flip)
    idx_dd = np.less_equal(np.random.random((num_trials, batch_size * num_samples)), dd_frac)
    
    if not can_have_no_dd:
        num_dd = np.sum(idx_dd, axis=0)
        no_dd_idx = num_dd == 0
        num_no_dd = np.sum(num_dd == 0)
        while num_no_dd:
            new_idx_dd = np.less_equal(np.random.random((num_trials, num_no_dd)), dd_frac)
            
            idx_dd[:, no_dd_idx] = new_idx_dd
            
            num_dd = np.sum(idx_dd, axis=0)
            
            no_dd_idx = num_dd == 0
            
            num_no_dd = np.sum(num_dd == 0)

    for n_samp in range(num_samples*batch_size):

        idx_dd2 = idx_dd[:, n_samp]
        
        
        if can_be_same:
            stim_red = np.random.choice(red_stim2, size=1)
            stim_dev = np.random.choice(dev_stim2, size=1)
        else:
            if dev_stim2.shape[0]>1 or red_stim2.shape[0]>1:
                is_same=1
                while is_same:
                    stim_red = np.random.choice(red_stim2, size=1)
                    stim_dev = np.random.choice(dev_stim2, size=1)
                    if stim_dev != stim_red:
                        is_same = 0
            else:
                stim_red = np.random.choice(red_stim2, size=1)
                stim_dev = np.random.choice(dev_stim2, size=1)
        
        red_dd_freq[0,n_samp] = stim_red
        red_dd_freq[1,n_samp] = stim_dev
        
        trials_oddball_freq[idx_dd2, n_samp] = stim_dev
        trials_oddball_freq[~idx_dd2, n_samp] = stim_red
        
        trials_oddball_ctx[idx_dd2, n_samp] = 2
        trials_oddball_ctx[~idx_dd2, n_samp] = 1
        
    trials_oddball_freq2 = trials_oddball_freq.reshape((num_trials, batch_size, num_samples), order='F')
    trials_oddball_ctx2 = trials_oddball_ctx.reshape((num_trials, batch_size, num_samples), order='F')
    red_dd_freq2 = red_dd_freq.reshape((2, batch_size, num_samples), order='F')
    
    if num_samples == 1:
        trials_oddball_freq2 = trials_oddball_freq2[:,:,0]
        trials_oddball_ctx2 = trials_oddball_ctx2[:,:,0]
    
    return trials_oddball_freq2, trials_oddball_ctx2, red_dd_freq2

#%%

def f_gen_input_output_from_seq(input_trials, stim_templates, output_templates, params):
    #  output (T seq_len, batch_size, input_size/output_size, num_samples)
    
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
    batch_size = 1;
    if len(shape1) > 2:
        batch_size = shape1[1]
        num_samp = shape1[2]
    elif len(shape1) > 1:
        batch_size = shape1[1]

    input_trials = input_trials.reshape((num_trials, batch_size, num_samp))
    
    input_shape = [input_size, T, batch_size, num_samp]
    output_shape = [output_size, T, batch_size, num_samp]
    
    input_mat = stim_templates[:,:,input_trials].reshape(input_shape, order='F') + np.random.normal(0,input_noise_std, input_shape)
    
    if params['normalize_input']:
        input_mat = input_mat - np.mean(input_mat)
        input_mat/np.std(input_mat)
        
    input_mat_out = input_mat.transpose([1,2, 0, 3])
    
    output_mat = output_templates[:,:,input_trials].reshape(output_shape, order='F')
    output_mat_out = output_mat.transpose([1,2, 0, 3])
    
    if num_samp == 1:
        input_mat_out = input_mat_out[:,:,:,0]
        output_mat_out = output_mat_out[:,:,:,0]
        
    return input_mat_out, output_mat_out
    
#%%

def f_plot_rates3(rates, num_cells_plot = 999999):
    
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

def f_plot_examle_inputs(input_plot, output_plot, params, num_plot = 5):
    # (T, trials, input_size)

    T, num_batch, input_size = input_plot.shape
    
    _, _, output_size = output_plot.shape
    
    batch_idx = np.random.choice(num_batch, num_plot)
    
    for n_bt in range(num_plot):
        
        input_temp = input_plot[:,batch_idx[n_bt],:]
        output_temp = output_plot[:,batch_idx[n_bt],:]
        
        spec2 = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[1, 1])
        plt.figure()
        ax1 = plt.subplot(spec2[0])
        ax1.imshow(input_temp.T, aspect="auto")
        plt.title('inputs; %d stim; %d intups; std=%.1f; batch %d' % (params['test_num_freq_stim'], input_size, params['stim_t_std'], batch_idx[n_bt]))
        plt.subplot(spec2[1], sharex=ax1)
        plt.imshow(output_temp.T, aspect="auto")
        plt.title('outputs')
        
        spec3 = gridspec.GridSpec(ncols=1, nrows=3, height_ratios=[1, 1, 1])
        plt.figure()
        ax1 = plt.subplot(spec3[0])
        ax1.plot(input_temp.std(axis=0))
        plt.title('inputs std %d inputs; batch %d' % (params['test_num_freq_stim'], batch_idx[n_bt]))
        plt.subplot(spec3[1], sharex=ax1)
        plt.plot(input_temp.mean(axis=0))
        plt.title('inputs mean')
        plt.subplot(spec3[2], sharex=ax1)
        plt.plot(input_temp.max(axis=0))
        plt.title('inputs max')

        plt.figure()
        plt.plot(np.mean(input_temp, axis=1))
        plt.title('mean spectrogram across time; batch %d' % batch_idx[n_bt])
        plt.xlabel('inputs')
        plt.ylabel('mean power')


#%%

def f_plot_train_loss(train_out, name_tag1, name_tag2):
    sm_bin = 50#round(1/params['dt'])*50;
    #trial_len = out_temp_all.shape[1]
    kernel = np.ones(sm_bin)/sm_bin

    loss_train = np.asarray(train_out['loss'])
    #loss_train = np.asarray(train_out_cont['loss']).T.flatten()
    loss_train_cont_sm = np.convolve(loss_train, kernel, mode='valid')
    loss_x_sm = np.arange(len(loss_train_cont_sm))+sm_bin/2 #/(trial_len)
    loss_x_raw = np.arange(len(loss_train)) #/(trial_len)

    loss_by_tt = np.array(train_out['loss_by_tt'])
    loss_by_tt_sm0 = np.convolve(loss_by_tt[:,0], kernel, mode='valid')
    loss_by_tt_sm1 = np.convolve(loss_by_tt[:,1], kernel, mode='valid')
    loss_by_tt_sm2 = np.convolve(loss_by_tt[:,2], kernel, mode='valid')

    fig1 = plt.figure()
    plt.semilogy(loss_x_raw, loss_train, 'lightgray')
    plt.semilogy(loss_x_sm, loss_train_cont_sm, 'gray')
    plt.legend(('train', 'train smoothed'))
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('train loss\n%s\n%s' % (name_tag1, name_tag2))


    fig2 = plt.figure()
    plt.semilogy(loss_x_raw, loss_train, 'lightgray')
    plt.semilogy(loss_x_raw, loss_by_tt[:,1], 'lightblue')
    plt.semilogy(loss_x_raw, loss_by_tt[:,2], 'pink')

    plt.semilogy(loss_x_sm, loss_train_cont_sm, 'gray')
    plt.semilogy(loss_x_sm, loss_by_tt_sm1, 'blue')
    plt.semilogy(loss_x_sm, loss_by_tt_sm2, 'red')
    plt.legend(('all', 'red', 'dd', 'all sm', 'red sm', 'dd sm'))
    plt.title('train loss deets\n%s\n%s' % (name_tag1, name_tag2))

    fig3 = plt.figure()
    plt.semilogy(loss_x_raw, loss_by_tt[:,0], 'lightgreen')
    plt.semilogy(loss_x_sm, loss_by_tt_sm0, 'green')
    plt.legend(('isi raw', 'isi sm'))
    plt.title('isi loss\n%s\n%s' % (name_tag1, name_tag2))
#%%

def f_plot_train_test_loss(train_out, test_out_cont, test_out_ob, name_tag1, name_tag2):
    sm_bin = 50#round(1/params['dt'])*50;
    #trial_len = out_temp_all.shape[1]
    kernel = np.ones(sm_bin)/sm_bin

    loss_train = np.asarray(train_out['loss'])# .T.flatten()
    loss_train_cont_sm = np.convolve(loss_train, kernel, mode='valid')
    loss_x = np.arange(len(loss_train_cont_sm)) + sm_bin/2 #/(trial_len)
    loss_x_raw = np.arange(len(loss_train)) #/(trial_len)

    loss_test_cont = np.asarray(test_out_cont['loss'])
    loss_test_cont_sm = np.convolve(loss_test_cont, kernel, mode='valid')
    loss_x_test_raw = np.arange(len(loss_test_cont))
    loss_x_test = np.arange(len(loss_test_cont_sm))  + sm_bin/2#/(trial_len)

    loss_test_ob = np.asarray(test_out_ob['loss'])
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
    plt.ylabel('loss')
    plt.title(name_tag1)
    plt.title('loss\n%s\n%s' % (name_tag1, name_tag2))

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
