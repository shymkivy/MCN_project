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
from scipy.spatial.distance import pdist, squareform, cdist #
from scipy.signal import correlate
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

num_rnn0 = 10


dparams = {}
dparams['num_trials'] = 1000
dparams['num_batch'] = 50
dparams['num_dev_stim'] = 10
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

if 'red_count' not in ob_data1.keys():
    red_count = np.zeros((num_trials, num_run))
    ctx_data = ob_data1['trials_test_oddball_ctx']
    for n_run in range(num_run):
        red_count1 = 1
        for n_tr in range(num_trials):
            if not ctx_data[n_tr,n_run]:
                red_count[n_tr,n_run] = red_count1
                red_count1 += 1
            else:
                red_count1 = 1
        
    ob_data1['red_count'] = red_count

#%%

x1 = nn.Linear(5, 4)

mean_cell_weights = np.zeros((num_rnn, num_cells))

for n_rnn in range(num_rnn):
    temp_rnn = rnn_all[n_rnn]
    Wr = temp_rnn.h2h.weight.detach().numpy()
    
    for n_cell in range(num_cells):
        cell_weights = Wr[:,n_cell]
    
        mean_cell_weights[n_rnn, n_cell] = np.mean(cell_weights)

mean_cell_weights0 = np.zeros((num_rnn0, num_cells))
for n_rnn in range(num_rnn0):
    temp_rnn = rnn0_all[n_rnn]
    Wr = temp_rnn.h2h.weight.detach().numpy()
    
    for n_cell in range(num_cells):
        cell_weights = Wr[:,n_cell]
    
        mean_cell_weights0[n_rnn, n_cell] = np.mean(cell_weights)   

#%%


num_corr_pts = 50
corr_len = 300
corr_neur_len = 100

sm_bin = 10#round(1/params['dt'])*50;
#trial_len = out_temp_all.shape[1]
kernel = np.ones(sm_bin)/sm_bin


tau_neur_all = np.zeros((num_rnn, num_run, num_cells))
has_data_neur = np.zeros((num_rnn, num_run, num_cells), dtype=bool)
tau_corr_all = np.zeros((num_rnn, num_run))


for n_rnn in range(num_rnn):
    test_data = test_ob_all[n_rnn]
    
    rates4d = test_data['rates4d_cut']
    rates_cut = test_data['rates_cut']
    
    
    for n_run in range(num_run):
        red_count_cut = ob_data1['red_count'][num_skip_trials:,:]
        last_red = np.where(red_count_cut[:,n_run] == 0)[0]-1
        
        trial_ave1 = np.mean(np.mean(rates4d[:,last_red,n_run,:], axis=1), axis=0)
        
        dist1 = cdist(np.reshape(trial_ave1, (1,num_cells)), np.reshape(rates_cut[:,n_run,:], (num_t2,num_cells)), 'euclidean')[0]
        
        dist1n = dist1 - np.mean(dist1)
        dist1n = dist1n/np.std(dist1n)
        
        corr1 = correlate(dist1n, dist1n)
        corr1_sm = np.convolve(corr1, kernel, mode='same')
        
        corr1_smn = corr1_sm - np.mean(corr1_sm)
        corr1_smn = corr1_smn/np.max(corr1_smn)
        
        corr1_smn2 = corr1_smn[num_t2:]
        
        tau_corr = np.where(corr1_smn2 < 0.5)[0][0]*params['dt']
        
        
        
        # plt.figure()
        # plt.plot(corr1_smn2)
        
        # x = np.arange(corr_len)+1
        # y = corr1[num_trials2*num_run:num_trials2*num_run+corr_len]
        
        # yn = y - np.min(y)+0.01
        # yn = yn/np.max(yn)
        
        # fit = np.polyfit(x, np.log(yn), 1)  
        
        # y_fit = np.exp(x*fit[0]+fit[1])
        
        # tau_corr = np.log(1/2)/fit[0]*params['dt']
        
        tau_corr_all[n_rnn, n_run] = tau_corr
        
        plt.figure()
        plt.plot(dist1n)
        plt.title('Run %d' %n_run)
        
        plt.figure()
        plt.plot(corr1)
        
        plt.figure()
        plt.plot(yn)
        plt.plot(y_fit)
        
        
        for n_nr in range(num_cells):
            neur = rates_cut[:,n_run,n_nr]
            
            if np.sum(neur) > 0.1:
                
                neurn = neur - np.mean(neur)
                neurn = neurn/np.std(neurn)
                
                corr2 = correlate(neurn, neurn)
                
                corr2_sm = np.convolve(corr2, kernel, mode='same')
                
                corr2_smn = corr2_sm - np.mean(corr2_sm)
                corr2_smn = corr2_smn/np.max(corr2_smn)
                
                corr2_smn2 = corr2_smn[num_t2:]
                
                tau_neur = np.where(corr2_smn2 < 0.5)[0][0]*params['dt']
                
                # x = np.arange(corr_neur_len)+1
                # y = corr2[num_trials2*num_run:num_trials2*num_run+corr_neur_len]
                
                # yn = y - np.min(y)+0.01
                # yn = yn/np.max(yn)
                
                # fit = np.polyfit(x, np.log(yn), 1)  
                
                # y_fit = np.exp(x*fit[0]+fit[1])
                
                # tau_neur = np.log(1/2)/fit[0]*params['dt']
                
                tau_neur_all[n_rnn, n_run, n_nr] = tau_neur
                has_data_neur[n_rnn, n_run, n_nr] = True
                # plt.figure()
                # plt.plot(neur)
                
                # plt.figure()
                # plt.plot(np.log(yn))
                # plt.plot(x*fit[0]+fit[1])
                
                # plt.figure()
                # plt.plot(yn)
                # plt.plot(np.exp(x*fit[0] + fit[1]))
    
                # plt.figure()
                # plt.plot(yn)
                # plt.plot(y_fit)
                # plt.title('neruon %d' % n_nr)
    
            
                # plt.figure()
                # plt.plot(np.exp(-0.05*x))
        
        
        
        # plt.figure()
        # plt.plot(tau_all)


tau0_neur_all = np.zeros((num_rnn0, num_run, num_cells))
has_data0_neur = np.zeros((num_rnn0, num_run, num_cells), dtype=bool)
tau0_corr_all = np.zeros((num_rnn0, num_run))


for n_rnn in range(num_rnn0):
    test_data = test0_ob_all[n_rnn]
    
    rates4d = test_data['rates4d_cut']
    rates_cut = test_data['rates_cut']
    
    
    for n_run in range(num_run):
        red_count_cut = ob_data1['red_count'][num_skip_trials:,:]
        last_red = np.where(red_count_cut[:,n_run] == 0)[0]-1
        
        trial_ave1 = np.mean(np.mean(rates4d[:,last_red,n_run,:], axis=1), axis=0)
        
        dist1 = cdist(np.reshape(trial_ave1, (1,num_cells)), np.reshape(rates_cut[:,n_run,:], (num_t2,num_cells)), 'euclidean')[0]
        
        dist1n = dist1 - np.mean(dist1)
        dist1n = dist1n/np.std(dist1n)
        
        corr1 = correlate(dist1n, dist1n)
        corr1_sm = np.convolve(corr1, kernel, mode='same')
        
        corr1_smn = corr1_sm - np.mean(corr1_sm)
        corr1_smn = corr1_smn/np.max(corr1_smn)
        
        corr1_smn2 = corr1_smn[num_t2:]
        
        tau_corr = np.where(corr1_smn2 < 0.5)[0][0]*params['dt']
        
        tau0_corr_all[n_rnn, n_run] = tau_corr

        for n_nr in range(num_cells):
            neur = rates_cut[:,n_run,n_nr]
            
            if np.sum(neur) > 0.1:
                
                neurn = neur - np.mean(neur)
                neurn = neurn/np.std(neurn)
                
                corr2 = correlate(neurn, neurn)
                
                corr2_sm = np.convolve(corr2, kernel, mode='same')
                
                corr2_smn = corr2_sm - np.mean(corr2_sm)
                corr2_smn = corr2_smn/np.max(corr2_smn)
                
                corr2_smn2 = corr2_smn[num_t2:]
                
                tau_neur = np.where(corr2_smn2 < 0.5)[0][0]*params['dt']
                
                tau0_neur_all[n_rnn, n_run, n_nr] = tau_neur
                has_data0_neur[n_rnn, n_run, n_nr] = True

#%%

plt.figure()
for n_rnn in range(num_rnn):
    plt.plot(np.ones((num_run))+(np.random.rand(num_run)-0.5)/4, tau_corr_all[n_rnn], '.')
for n_rnn in range(num_rnn0):
    plt.plot(np.ones((num_run))*2+(np.random.rand(num_run)-0.5)/4, tau0_corr_all[n_rnn], '.')  
plt.ylabel('tau dist')

plt.figure()
for n_rnn in range(num_rnn):
    plt.plot(np.ones((num_run))+(np.random.rand(num_run)-0.5)/4, np.log(tau_corr_all[n_rnn]), '.')
for n_rnn in range(num_rnn0):
    plt.plot(np.ones((num_run))*2+(np.random.rand(num_run)-0.5)/4, np.log(tau0_corr_all[n_rnn]), '.')  
plt.ylabel('log(tau) dist')



plt.figure()
for n_rnn in range(num_rnn):
    plt.plot(np.ones((num_run))+(np.random.rand(num_run)-0.5)/4, np.log(np.mean(tau_neur_all[n_rnn,:,:], axis=1)), '.')
for n_rnn in range(num_rnn0):
    plt.plot(np.ones((num_run))*2+(np.random.rand(num_run)-0.5)/4, np.log(np.mean(tau0_neur_all[n_rnn,:,:], axis=1)), '.')  
plt.ylabel('log(tau) neuron, ave cells')

plt.figure()
for n_rnn in range(num_rnn):
    plt.plot(np.ones((num_cells))+(np.random.rand(num_cells)-0.5)/4, np.log(np.mean(tau_neur_all[n_rnn,:,:], axis=0)), '.')
for n_rnn in range(num_rnn0):
    plt.plot(np.ones((num_cells))*2+(np.random.rand(num_cells)-0.5)/4, np.log(np.mean(tau0_neur_all[n_rnn,:,:], axis=0)), '.')  
plt.ylabel('log(tau) neuron, ave run')
    
#%%

plt.figure()
for n_rnn in range(num_rnn):
    plt.plot(mean_cell_weights[n_rnn,:], np.log(np.mean(tau_neur_all[n_rnn,:,:], axis=0)), '.')
plt.xlabel('mean weights')
plt.ylabel('log(tau) decay')
plt.title('trained')

plt.figure()   
for n_rnn in range(num_rnn0):
    plt.plot(mean_cell_weights0[n_rnn,:], np.log(np.mean(tau0_neur_all[n_rnn,:,:], axis=0)), '.')  
plt.xlabel('mean weights')
plt.ylabel('log(tau) decay')
plt.title('untrained')


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




