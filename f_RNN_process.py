# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:09:22 2024

@author: ys2605
"""

import numpy as np

#%%
def f_label_redundants(trials_ctx):
    
    num_trials, num_runs = trials_ctx.shape
    
    forward_label = np.zeros((num_trials, num_runs), dtype=int)
    reverse_label = np.zeros((num_trials, num_runs), dtype=int)
    
    for n_run in range(num_runs):
        
        n_red = 0
        for n_tr in range(num_trials):
            if trials_ctx[n_tr, n_run] == 0: # if red
                n_red += 1
            else:
                n_red = 0
            forward_label[n_tr, n_run] = n_red
            
        n_red = -99
        for n_tr in range(num_trials):
            n_tr_rev = num_trials-n_tr-1
            
            if trials_ctx[n_tr_rev, n_run] == 0: # if red
                n_red -= 1
            else:
                n_red=0
            reverse_label[n_tr_rev, n_run] = n_red
                  
    return forward_label, reverse_label


#%%

def f_get_rdc_trav(trials_oddball_ctx_cut, rates4d_cut, trials_test_cont_cut, rates_cont_freq4d_cut, plot_t1, red_dd_seq, baseline_subtract=True):
    trial_len, num_trials, num_runs, num_cells = rates4d_cut.shape

    num_cont_trials, num_cont_runs = trials_test_cont_cut.shape

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
        
        cont_run = 0
        cont_tr_idx = trials_test_cont_cut[:,cont_run] == freq1
        cont_resp_all = rates_cont_freq4d_cut[:,cont_tr_idx,cont_run,:]
        
        red_tr_ave = np.mean(red_resp_all2, axis=1)
        dev_tr_ave = np.mean(dev_resp_all2, axis=1)
        cont_tr_ave = np.mean(cont_resp_all, axis=1)
      
        if baseline_subtract:
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