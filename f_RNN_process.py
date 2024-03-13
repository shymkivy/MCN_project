# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:09:22 2024

@author: ys2605
"""

import numpy as np

from scipy.spatial.distance import pdist, squareform, cdist #
from scipy.signal import correlate

#%%
def f_label_redundants(trials_ctx):
    
    num_trials, num_runs = trials_ctx.shape
    
    num_ctx = np.unique(trials_ctx).shape[0]
    
    if num_ctx == 2:
        red_idx = 0
    elif num_ctx == 3:
        red_idx = 1
    
    forward_label = np.zeros((num_trials, num_runs), dtype=int)
    reverse_label = np.zeros((num_trials, num_runs), dtype=int)
    
    for n_run in range(num_runs):
        
        n_red = 0
        for n_tr in range(num_trials):
            if trials_ctx[n_tr, n_run] == red_idx: # if red
                n_red += 1
            else:
                n_red = 0
            forward_label[n_tr, n_run] = n_red
            
        n_red = -99
        for n_tr in range(num_trials):
            n_tr_rev = num_trials-n_tr-1
            
            if trials_ctx[n_tr_rev, n_run] == red_idx: # if red
                n_red -= 1
            else:
                n_red=0
            reverse_label[n_tr_rev, n_run] = n_red
                  
    return forward_label, reverse_label


#%%

def f_get_rdc_trav(trials_oddball_ctx_cut, rates4d_cut, trials_cont_cut, rates_cont_freq4d_cut, params, red_dd_seq, baseline_subtract=True, base_time = [-0.5, 0]):
    trial_len, num_trials, num_runs, num_cells = rates4d_cut.shape
    _, _, num_runs_cont, _ = rates_cont_freq4d_cut.shape
    
    plot_t1 = (np.arange(trial_len)-trial_len/4)*params['dt']
    base_idx = np.logical_and(plot_t1>base_time[0], plot_t1<base_time[1])

    num_cont_trials, num_cont_runs = trials_cont_cut.shape

    trials_oddball_red_fwr, trials_oddball_red_rev = f_label_redundants(trials_oddball_ctx_cut)


    freqs_all = np.unique(red_dd_seq)
    num_freqs = freqs_all.shape[0]

    trial_ave_rdc = np.zeros((trial_len, 3, num_freqs, num_cells))

    for n_freq in range(num_freqs):
        freq1 = freqs_all[n_freq]
        
        red_run_idx = red_dd_seq[0,:] == freq1
        dev_run_idx = red_dd_seq[1,:] == freq1
        
        red_resp_all = []
        dev_resp_all = []
        for n_run in range(num_runs):
            if red_run_idx[n_run]:
                #red_tr_idx = trials_oddball_red_fwr[:,n_run] == 3
                red_tr_idx = trials_oddball_red_rev[:,n_run] == -3
                red_resp_all.append(rates4d_cut[:,red_tr_idx, n_run,:])
            
            if dev_run_idx[n_run]:
                dev_tr_idx = trials_oddball_ctx_cut[:,n_run] == 1
                dev_resp_all.append(rates4d_cut[:,dev_tr_idx, n_run,:])
        
        red_resp_all2 = np.concatenate(red_resp_all, axis=1)
        dev_resp_all2 = np.concatenate(dev_resp_all, axis=1)
        
        cont_resp_all = []
        for n_run in range(num_runs_cont):
            cont_tr_idx = trials_cont_cut[:,n_run] == freq1
            cont_resp_all.append(rates_cont_freq4d_cut[:,cont_tr_idx,n_run,:])
        
        cont_resp_all2 = np.concatenate(cont_resp_all, axis=1)
        
        red_tr_ave = np.mean(red_resp_all2, axis=1)
        dev_tr_ave = np.mean(dev_resp_all2, axis=1)
        cont_tr_ave = np.mean(cont_resp_all2, axis=1)
      
        if baseline_subtract:
            red_tr_ave3 = red_tr_ave - np.mean(red_tr_ave[base_idx,:], axis=0)
            dev_tr_ave3 = dev_tr_ave - np.mean(dev_tr_ave[base_idx,:], axis=0)
            cont_tr_ave3 = cont_tr_ave - np.mean(cont_tr_ave[base_idx,:], axis=0)
        else:
            red_tr_ave3 = red_tr_ave
            dev_tr_ave3 = dev_tr_ave
            cont_tr_ave3 = cont_tr_ave
        
        trial_ave_rdc[:,0,n_freq,:] = red_tr_ave3
        trial_ave_rdc[:,1,n_freq,:] = dev_tr_ave3
        trial_ave_rdc[:,2,n_freq,:] = cont_tr_ave3
        
    return trial_ave_rdc

#%%
def f_trial_ave_ctx_pad(rates4d_cut, trials_types_cut, pre_dd = 2, post_dd = 2):
    num_t, num_tr, num_run, num_cells = rates4d_cut.shape
    
    num_tr_ave = pre_dd + post_dd + 1
    
    temp_ave4d = np.zeros((num_t, num_tr_ave, num_run, num_cells))
    
    for n_run in range(num_run):
            
        temp_ob = trials_types_cut[:,n_run]
        
        temp_sum1 = np.zeros((num_t, num_tr_ave, num_cells))
        num_dd = 0
        
        for n_tr in range(pre_dd, num_tr-post_dd):
            
            if temp_ob[n_tr]:
                
                if np.sum(temp_ob[n_tr-pre_dd:n_tr+post_dd+1]) == 1:
                        
                    num_dd += 1
                    temp_sum1 += rates4d_cut[:, n_tr-pre_dd:n_tr+post_dd+1, n_run, :]
                
        temp_ave4d[:,:,n_run,:] = temp_sum1/num_dd
        
        trial_ave3d = np.reshape(temp_ave4d, (num_t*num_tr_ave, num_run, num_cells), order = 'F')
        
    return trial_ave3d

#%%
def f_trial_ave_ctx_pad2(rates4d_cut, trials_types_cut, pre_dd = 2, post_dd = 2, limit_1_dd=False, max_trials=999, shuffle_trials=False):
    num_t, num_tr, num_run, num_cells = rates4d_cut.shape
    
    num_tr_ave = pre_dd + post_dd + 1
    
    temp_ave4d = np.zeros((num_t, num_tr_ave, num_run, num_cells))
    
    max_dd_trials = np.max(np.sum(trials_types_cut, axis=0))
    
    num_dd_trials = np.zeros((num_run), dtype=int)
    trial_data_sort = []
    
    for n_run in range(num_run):
            
        temp_ob = trials_types_cut[:,n_run]
        
        num_dd=0
        
        trial_data_sort2 = []
        
        for n_tr in range(pre_dd, num_tr-post_dd):
            
            if temp_ob[n_tr]: # if currently on dd trial
                
                if limit_1_dd:  # if only 1 dd in vicinity
                    get_trial = np.sum(temp_ob[n_tr-pre_dd:n_tr+post_dd+1]) == 1
                else:
                    get_trial = True
            
                if get_trial:
                    
                    trial_data_sort2.append(rates4d_cut[:, n_tr-pre_dd:n_tr+post_dd+1, n_run, :][:,:,None,:])
                    num_dd += 1

        
        trial_data_sort3 = np.concatenate(trial_data_sort2, axis=2)
        
        if shuffle_trials:
            shuff_idx = np.arange(num_dd)
            np.random.shuffle(shuff_idx)
            trial_data_sort3 = trial_data_sort3[:,:,shuff_idx,:]
        
        use_dd = np.min((num_dd,max_trials))
        
        trial_data_sort4 = trial_data_sort3[:,:,:use_dd,:]
        
        num_dd_trials[n_run] = use_dd
        
        temp_ave4d[:,:,n_run,:] = np.mean(trial_data_sort4,axis=2)
        
        trial_data_sort.append(trial_data_sort4)
    
    return temp_ave4d, trial_data_sort, num_dd_trials


#%%
def f_trial_sort_data_pad(rates4d_cut, trials_types_cut, pre_trials = 2, post_trials = 2):
    # get trial ave and sorted single trial data with more trials
    
    num_t, num_tr, num_run, num_cells = rates4d_cut.shape
    
    num_tr_ave = pre_trials + post_trials + 1
    
    trials_per_run = num_tr - pre_trials - post_trials
    
    trial_data_sort = np.zeros((num_t, num_tr_ave, trials_per_run, num_run, num_cells))
    trial_types_pad = np.zeros((num_tr_ave, trials_per_run, num_run))
    for n_run in range(num_run):
            
        for n_tr in range(pre_trials, num_tr-post_trials-1):
            trial_data_sort[:,:,n_tr-pre_trials,n_run,:] = rates4d_cut[:,(n_tr-pre_trials):(n_tr+post_trials+1),n_run,:]
            trial_types_pad[:,n_tr-pre_trials,n_run] = trials_types_cut[(n_tr-pre_trials):(n_tr+post_trials+1),n_run]
    
    if post_trials:
        trials_types_out = trials_types_cut[pre_trials:-post_trials,:]
    else:
        trials_types_out = trials_types_cut[pre_trials:,:]
    
    trial_data_sort4d = np.reshape(trial_data_sort, (num_t*num_tr_ave, trials_per_run, num_run, num_cells), order = 'F')
    trial_types_pad2d = np.reshape(trial_types_pad, (num_tr_ave, trials_per_run, num_run), order = 'F')
    
    return trial_data_sort4d, trials_types_out, trial_types_pad2d

#%%
def f_trial_sort_data_ctx_pad(rates4d_in, trials_types_ctx, trials_types_freq, pre_trials = 2, post_trials = 2, max_trials=999, shuffle_trials=False):
    # get trial ave and sorted single trial data with more trials
    # zero  trials get thrown away
    
    num_t, num_tr, num_run, num_cells = rates4d_in.shape
    num_tr_ave = pre_trials + post_trials + 1
    
    #trials_per_run = num_tr - pre_trials - post_trials
    
    trial_data_sort = []
    dd_freqs_out = []
    num_dd_trials_all = np.zeros((num_run), dtype=int)
  
    for n_run in range(num_run):
        
        num_dd_trials = np.sum(trials_types_ctx[pre_trials:-post_trials-1, n_run]).astype(int)
        
        trial_data_sort2 = np.zeros((num_t, num_tr_ave, num_dd_trials, num_cells))
        dd_type_freq = np.zeros(num_dd_trials, dtype=int)
        
        n_dd = 0
        for n_tr in range(pre_trials, num_tr-post_trials-1):
            if trials_types_ctx[n_tr, n_run]:
                trial_data_sort2[:,:,n_dd,:] = rates4d_in[:,(n_tr-pre_trials):(n_tr+post_trials+1),n_run,:]
                dd_type_freq[n_dd] = trials_types_freq[n_tr, n_run]
                n_dd += 1
                

        if shuffle_trials:
            shuff_idx = np.arange(num_dd_trials)
            np.random.shuffle(shuff_idx)
            trial_data_sort3 = trial_data_sort2[:, :, shuff_idx, :]
            dd_type_freq2 = dd_type_freq[shuff_idx]
        else:
            trial_data_sort3 = trial_data_sort2
            dd_type_freq2 = dd_type_freq
        
        num_dd_trials2 = np.min((num_dd_trials, max_trials))
        
        trial_data_sort4 = trial_data_sort3[:,:,:num_dd_trials2,:]
        dd_type_freq3 = dd_type_freq2[:num_dd_trials2]
        
        trial_data_sort4_3d = np.reshape(trial_data_sort4, (num_t*num_tr_ave, num_dd_trials2, num_cells), order = 'F')
        
        num_dd_trials_all[n_run] = num_dd_trials2
        
        trial_data_sort.append(trial_data_sort4_3d)
        dd_freqs_out.append(dd_type_freq3)
        
    return trial_data_sort, dd_freqs_out, num_dd_trials_all

#%%
def f_trial_ave_pad(rates4d_cut, trials_types_cut, pre_dd = 2, post_dd = 2):
    # get trial ave and sorted single trial data with more trials
    
    num_t, num_tr, num_run, num_cells = rates4d_cut.shape
    
    num_tr_ave = pre_dd + post_dd + 1
    
    temp_ave4d = np.zeros((num_t, num_tr_ave, num_run, num_cells))
    
    max_dd_trials = np.max(np.sum(trials_types_cut, axis=0))
    
    num_dd_trials = np.zeros((num_run))
    trial_data_sort = np.zeros((num_t, num_tr_ave, max_dd_trials, num_run, num_cells))
    
    for n_run in range(num_run):
            
        temp_ob = trials_types_cut[:,n_run]
        
        temp_sum1 = np.zeros((num_t, num_tr_ave, num_cells))
        num_dd = 0
        
        for n_tr in range(pre_dd, num_tr-post_dd):
            
            if temp_ob[n_tr]: # if currently on dd trial
                
                if np.sum(temp_ob[n_tr-pre_dd:n_tr+post_dd+1]) == 1: # if only 1 dd in vicinity
                    
                    trial_data_sort[:,:,num_dd,n_run,:] = rates4d_cut[:, n_tr-pre_dd:n_tr+post_dd+1, n_run, :]
                    num_dd_trials[n_run] +=1
                    
                    num_dd += 1
                    temp_sum1 += rates4d_cut[:, n_tr-pre_dd:n_tr+post_dd+1, n_run, :]
                    
        temp_ave4d[:,:,n_run,:] = temp_sum1/num_dd
    
    return temp_ave4d, trial_data_sort, num_dd_trials

#%%
def f_trial_ave_ctx_rd(rates4d_cut, trials_types_cut, params):
    num_t, num_tr, num_run, num_cells = rates4d_cut.shape
    
    if params['num_ctx'] == 1:
        ctx_pad1 = 0
    elif params['num_ctx'] == 2:
        ctx_pad1 = 1
        
    trial_ave_rd = np.zeros((2, num_t, num_run, num_cells))
    
    for n_run in range(num_run):
        idx1 = trials_types_cut[:,n_run] == 0+ctx_pad1
        trial_ave_rd[0,:,n_run,:] = np.mean(rates4d_cut[:,idx1,n_run,:], axis=1)
        
        idx1 = trials_types_cut[:,n_run] == 1+ctx_pad1
        trial_ave_rd[1,:,n_run,:] = np.mean(rates4d_cut[:,idx1,n_run,:], axis=1)
    
    return trial_ave_rd

#%%
def f_euc_dist(vec1, vec2):
    
    dist_sq = (vec1 - vec2)**2
    
    if len(dist_sq.shape) == 1:
        dist1 = np.sqrt(np.sum(dist_sq, axis=0))
    else:
        dist1 = np.sqrt(np.sum(dist_sq, axis=1))
    
    return dist1

def f_cos_sim(vec1, vec2):
    
    vec1_mag = np.sqrt(np.sum(np.abs(vec1.T)**2, axis=1))
    vec2_mag = np.sqrt(np.sum(np.abs(vec2)**2, axis=0))
    
    vec_angles = np.dot(vec1.T, vec2)/(vec1_mag*vec2_mag)
    
    return vec_angles


#%%
def f_gather_dev_trials(rates4d_cut, trials_oddball_ctx_cut, red_dd_seq):

    num_trials, num_runs = trials_oddball_ctx_cut.shape
    
    freq_red_all = np.unique(red_dd_seq[0,:])
    freqs_dev_all = np.unique(red_dd_seq[1,:])
    num_freq_r = len(freq_red_all)
    num_freq_d = len(freqs_dev_all)
    
    trials_rd = []
    for n_fr in range(num_freq_r):
        trials_d = []
        for n_fd in range(num_freq_d):
            trials_d.append([])
        trials_rd.append(trials_d)
    
    for n_run in range(num_runs):
        freq_red = red_dd_seq[0,n_run]
        freq_dev = red_dd_seq[1,n_run]
        
        red_loc = np.where(freq_red_all==freq_red)[0][0]
        dev_loc = np.where(freqs_dev_all==freq_dev)[0][0]
        
        #dd_idx = trials_oddball_freq_cut[:,n_run] == freq_dev
        dd_idx = trials_oddball_ctx_cut[:,n_run] == 1
        
        temp_rates = rates4d_cut[:,dd_idx,n_run,:]
        
        trials_rd[red_loc][dev_loc].append(temp_rates)
    
    return trials_rd

def f_gather_red_trials(rates4d_cut, trials_oddball_freq_cut, trials_oddball_ctx_cut, red_dd_seq, red_idx = 3):

    num_trials, num_runs = trials_oddball_freq_cut.shape
    
    freq_red_all = np.unique(red_dd_seq[0,:])
    freqs_dev_all = np.unique(red_dd_seq[1,:])
    num_freq_r = len(freq_red_all)
    num_freq_d = len(freqs_dev_all)
    
    trials_rd = []
    for n_fr in range(num_freq_r):
        trials_d = []
        for n_fd in range(num_freq_d):
            trials_d.append([])
        trials_rd.append(trials_d)
    
    for n_run in range(num_runs):
        freq_red = red_dd_seq[0,n_run]
        freq_dev = red_dd_seq[1,n_run]

        red_loc = np.where(freq_red_all==freq_red)[0][0]
        dev_loc = np.where(freqs_dev_all==freq_dev)[0][0]
        
        trials_oddball_freq_cut2 = trials_oddball_freq_cut[:,n_run]
        trials_oddball_ctx_cut2 = trials_oddball_ctx_cut[:,n_run]
        trials_oddball_ctx_cut3 = trials_oddball_ctx_cut2[trials_oddball_freq_cut2>0]
        
        temp_rates = rates4d_cut[:,:,n_run,:]
        temp_rates2 = temp_rates[:,trials_oddball_freq_cut2>0,:]
        
        trials_oddball_red_fwr, trials_oddball_red_rev = f_label_redundants(trials_oddball_ctx_cut3[:,None])
        
        if red_idx>0:
            red_idx2 = trials_oddball_red_fwr[:,0] == red_idx
        else:
            red_idx2 = trials_oddball_red_rev[:,0] == red_idx
                
        temp_rates3 = temp_rates2[:,red_idx2,:]
        
        trials_rd[red_loc][dev_loc].append(temp_rates3)
    
    return trials_rd

#%%
def f_analyze_trial_vectors(trials_rd, params, base_time = [-.250, 0], on_time = [.2, .5]):
    
    num_freq_r = len(trials_rd)
    num_freq_d = len(trials_rd[0])
    
    trial_len, num_trials, num_cells = trials_rd[0][0][0].shape
    
    plot_t1 = (np.arange(trial_len)-trial_len/4)*params['dt']
    
    base_idx = np.logical_and(plot_t1>base_time[0], plot_t1<base_time[1])
    on_idx = np.logical_and(plot_t1>on_time[0], plot_t1<on_time[1])
    
    base_mean_rd = np.zeros((num_freq_r, num_freq_d, num_cells))
    on_mean_rd = np.zeros((num_freq_r, num_freq_d, num_cells))
    
    base_std_rd = np.zeros((num_freq_r, num_freq_d, num_cells))
    on_std_rd = np.zeros((num_freq_r, num_freq_d, num_cells))
    
    base_dist_means_rd = np.zeros((num_freq_r, num_freq_d))
    on_dist_means_rd = np.zeros((num_freq_r, num_freq_d))
    
    mean_indiv_mag_rd = np.zeros((num_freq_r, num_freq_d))
    mean_mag_rd = np.zeros((num_freq_r, num_freq_d))
    mean_angles_rd = np.zeros((num_freq_r, num_freq_d))
    
    mean_vec_dir = np.zeros((num_freq_r, num_freq_d, num_cells))
     
    for n_fr in range(num_freq_r):
        for n_fd in range(num_freq_d):
            temp_data = np.concatenate(trials_rd[n_fr][n_fd], axis=1)
            
            # baselibes and on of each trial
            temp_base = np.mean(temp_data[base_idx,:,:], axis=0)
            temp_on = np.mean(temp_data[on_idx,:,:], axis=0)
            
            # base and on trial ave
            temp_base_mean = np.mean(temp_base, axis=0)
            temp_on_mean = np.mean(temp_on, axis=0)
            
            # save trial aves
            base_mean_rd[n_fr, n_fd,:] = temp_base_mean
            on_mean_rd[n_fr, n_fd,:] = temp_on_mean
            
            mean_vec_dir[n_fr, n_fd,:] = temp_on_mean - temp_base_mean
            
            # base and on trial std
            base_std_rd[n_fr, n_fd,:] = np.std(temp_base, axis=0)
            on_std_rd[n_fr, n_fd,:] = np.std(temp_on, axis=0)
            
            # distances from mean on each trial
            base_dist_means_rd[n_fr, n_fd] = np.mean(f_euc_dist(temp_base, temp_base_mean))
            on_dist_means_rd[n_fr, n_fd] = np.mean(f_euc_dist(temp_on, temp_on_mean))
            
            mean_indiv_mag_rd[n_fr, n_fd] = np.mean(f_euc_dist(temp_base, temp_on))
            
            mean_mag_rd[n_fr, n_fd] = f_euc_dist(temp_base_mean, temp_on_mean)
            
            trial_directions = temp_on - temp_base
            mean_direction = np.mean(trial_directions, axis=0)
    
            vec_angles = f_cos_sim(trial_directions.T, mean_direction)
            #vec_angles2 = 1 - pdist(np.vstack((mean_direction, trial_directions)), 'cosine')
            
            mean_angles_rd[n_fr, n_fd] = np.mean(vec_angles)
    
    data_out = {'base_dist_mean':       base_dist_means_rd,
                'on_dist_mean':         on_dist_means_rd,
                'mean_indiv_mag':       mean_indiv_mag_rd,
                'mean_mag':             mean_mag_rd,
                'mean_angles':          mean_angles_rd,
                'mean_vec_dir':         mean_vec_dir}
    
    return data_out


def f_analyze_cont_trial_vectors(rates_cont4d, trials_cont_cut, freqs_list, params, base_time = [-.250, 0], on_time = [.2, .5]):
    
    trial_len, num_trials, num_runs, num_cells = rates_cont4d.shape
    
    plot_t1 = (np.arange(trial_len)-trial_len/4)*params['dt']
    
    base_idx = np.logical_and(plot_t1>base_time[0], plot_t1<base_time[1])
    on_idx = np.logical_and(plot_t1>on_time[0], plot_t1<on_time[1])
    
    num_trials, num_runs = trials_cont_cut.shape
    
    freqs_all = np.unique(freqs_list)
    num_freqs = freqs_all.shape[0]
    
    trials_all = []
    
    n_run = 0
    for n_freq in range(num_freqs):
        freq1 = freqs_all[n_freq]
        trials_all2 = []
        for n_run in range(num_runs):
            tr_idx = trials_cont_cut[:,n_run] == freq1
        
            temp_data = rates_cont4d[:,tr_idx,n_run,:]
            
            trials_all2.append(temp_data)
        
        trials_all.append(trials_all2)
    
    
    base_mean = np.zeros((num_freqs, num_runs, num_cells))
    on_mean = np.zeros((num_freqs, num_runs, num_cells))
    
    base_std = np.zeros((num_freqs, num_runs, num_cells))
    on_std = np.zeros((num_freqs, num_runs, num_cells))
    
    base_dist_mean = np.zeros((num_freqs, num_runs))
    on_dist_mean = np.zeros((num_freqs, num_runs))
    
    mean_indiv_mag = np.zeros((num_freqs, num_runs))
    
    mean_mag = np.zeros((num_freqs, num_runs))
    
    mean_angles = np.zeros((num_freqs, num_runs))
    
    mean_vec_dir = np.zeros((num_freqs, num_runs, num_cells))
    
    for n_freq in range(num_freqs):
        for n_run in range(num_runs):
            temp_data = trials_all[n_freq][n_run]
            
            temp_base = np.mean(temp_data[base_idx,:,:], axis=0)
            temp_on = np.mean(temp_data[on_idx,:,:], axis=0)
            
            temp_base_mean = np.mean(temp_base, axis=0)
            temp_on_mean = np.mean(temp_on, axis=0)
            
            base_std[n_freq, n_run,:] = np.std(temp_base, axis=0)
            on_std[n_freq, n_run,:] = np.std(temp_on, axis=0)
            
            base_mean[n_freq, n_run,:] = temp_base_mean
            on_mean[n_freq, n_run,:] = temp_on_mean
            
            mean_vec_dir[n_freq, n_run,:] = temp_on_mean - temp_base_mean
            
            base_dist_mean[n_freq, n_run] = np.mean(f_euc_dist(temp_base, temp_base_mean))
            on_dist_mean[n_freq, n_run] = np.mean(f_euc_dist(temp_on, temp_on_mean))
            
            mean_indiv_mag[n_freq, n_run] = np.mean(f_euc_dist(temp_base, temp_on))
            
            mean_mag[n_freq, n_run] = f_euc_dist(temp_base_mean, temp_on_mean)
            
            trial_directions = temp_on - temp_base
            mean_direction = np.mean(trial_directions, axis=0)
    
            vec_angles = f_cos_sim(trial_directions.T, mean_direction)
            #vec_angles2 = 1 - pdist(np.vstack((mean_direction, trial_directions)), 'cosine')
            
            mean_angles[n_freq, n_run] = np.mean(vec_angles)
    
    data_out = {'base_dist_mean':       base_dist_mean,
                'on_dist_mean':         on_dist_mean,
                'mean_indiv_mag':       mean_indiv_mag,
                'mean_mag':             mean_mag,
                'mean_angles':          mean_angles,
                'mean_vec_dir':         mean_vec_dir}
    
    return data_out
    
#%%

def f_get_trace_tau(trace, sm_bin = 10):
    
    #sm_bin = 10#round(1/params['dt'])*50;
    #trial_len = out_temp_all.shape[1]
    kernel = np.ones(sm_bin)/sm_bin
    
    tracen = trace - np.mean(trace)
    tracen = tracen/np.std(tracen)
    
    corr1 = correlate(tracen, tracen)
    corr1_sm = np.convolve(corr1, kernel, mode='same')
    
    corr1_smn = corr1_sm - np.mean(corr1_sm)
    corr1_smn = corr1_smn/np.max(corr1_smn)
    
    corr1_smn2 = corr1_smn[len(trace):]
    
    tau_corr = np.where(corr1_smn2 < 0.5)[0][0]
    
    # x = np.arange(corr_len)+1
    # y = corr1[num_trials2*num_run:num_trials2*num_run+corr_len]
    
    # yn = y - np.min(y)+0.01
    # yn = yn/np.max(yn)
    
    # fit = np.polyfit(x, np.log(yn), 1)  
    
    # y_fit = np.exp(x*fit[0]+fit[1])
    
    # tau_corr = np.log(1/2)/fit[0]*params['dt']
    
    return tau_corr

    