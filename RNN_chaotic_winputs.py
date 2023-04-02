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
from f_RNN_chaotic import *

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

#%% loading inputs


#fname_RNN_load = 'test_20k_std3'
fname_RNN_load = 'test_20k';

fname_RNN_save = 'test_50k_std3'

#%% params
load_input = 0
compute_loss = 1
train_RNN = 1
save_RNN = 1
load_RNN = 0
plot_deets = 0

#%% loading inputs

plt.close('all')

if load_input:
    
    fpath = path1 + '/RNN_data/'

    #fname = 'sim_spec_1_stim_8_24_21.mat'
    #fname = 'sim_spec_10complex_200rep_stim_8_25_21_1.mat'
    #fname = 'sim_spec_10tones_200rep_stim_8_25_21_1.mat'

    fname_input = 'sim_spec_10tones_100reps_0.5isi_50dt_1_1_22_23h_17m.mat';
    
    data_mat = loadmat(fpath+fname_input)
    
    data1 = data_mat['spec_data']
    print(data1.dtype)
    
    input_mat = data1[0,0]['spec_cut'];
    fr_cut = data1[0, 0]['fr_cut'];
    ti = data1[0, 0]['ti'];
    voc_seq = data1[0, 0]['voc_seq'];
    num_voc = data1[0, 0]['num_voc'];
    output_mat = data1[0, 0]['output_mat'];
    output_mat_delayed = data1[0, 0]['output_mat_delayed'];
    input_T = data1[0, 0]['ti'][0];
    
    input_size = input_mat.shape[0];    # number of freqs in spectrogram
    output_size = output_mat.shape[0];  # number of target output categories 
    T = input_mat.shape[1];             # time steps of inputs and target outputs
    dt = input_T[1] - input_T[0];        #1;
    
    plt.figure()
    plt.imshow(input_mat)
    
    
    tau = .5;              # for to bin stim
    alpha = dt/tau;         # 
    
    plt.figure()
    plt.plot(input_mat.std(axis=0))
    
else:
    
    stim_duration = 0.5
    isi_suration = 0.5
    
    num_train_trials = 50000;
    num_test_trials = 1000;
    
    input_size = 50;
    num_stim = 10;
    dt = .05;
    tau = .5;              # in sec
    alpha = dt/tau;         # 
    
    input_noise_std = 1/10
    
    #stim_t_std = 0
    stim_t_std = 3
    
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
        
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(stim_temp)
        # plt.subplot(122)
        # plt.imshow(out_temp)
      
    trials_train = np.floor(np.random.random(num_train_trials)*num_stim).astype(int)
    trials_test = np.floor(np.random.random(num_test_trials)*num_stim).astype(int)
    
    T = (stim_bins + isi_bins) * len(trials_train)
    T_test = (stim_bins + isi_bins) * len(trials_test)
    
    input_mat = stim_temp_all[:,:,trials_train].reshape((input_size, T), order='F') + np.random.normal(0,input_noise_std,(input_size, T))
    output_mat = out_temp_all[:,:,trials_train].reshape((output_size, T), order='F')
    
    input_mat_test = stim_temp_all[:,:,trials_test].reshape((input_size, T_test), order='F') + np.random.normal(0,input_noise_std,(input_size, T_test))
    output_mat_test = out_temp_all[:,:,trials_test].reshape((output_size, T_test), order='F')
    
    # stim_vec = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # stim_trace = stim_temp_all[:,:,stim_vec].reshape((input_size, (stim_bins + isi_bins) * len(stim_vec)), order='F')
        
    # count = np.zeros((num_stim))

    # for n_tr in range(num_stim):
    #     count[n_tr] = np.sum(trials_train==n_tr)
    #     #count[n_tr] = np.sum(np.logical_and(np.less(trials, n_tr+1), np.greater_equal(trials, n_tr)))

    # plt.figure()
    # plt.plot(count)
    

if plot_deets:
    spec3 = gridspec.GridSpec(ncols=1, nrows=3, height_ratios=[1, 1, 1])
    
    plt.figure()
    ax1 = plt.subplot(spec3[0])
    ax1.plot(input_mat.std(axis=0))
    plt.title('inputs std; %d inputs' % input_size)
    plt.subplot(spec3[1], sharex=ax1)
    plt.plot(input_mat.mean(axis=0))
    plt.title('inputs mean')
    plt.subplot(spec3[2], sharex=ax1)
    plt.plot(input_mat.max(axis=0))
    plt.title('inputs max')
    
    spec2 = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[1, 1])
    
    plt.figure()
    ax1 = plt.subplot(spec2[0])
    ax1.imshow(input_mat, aspect="auto")
    plt.title('inputs; %d stim; %d intups; std=%.1f' % (num_stim, input_size, stim_t_std))
    plt.subplot(spec2[1], sharex=ax1)
    plt.imshow(output_mat, aspect="auto")
    plt.title('outputs')
    

#%% initialize RNN 

hidden_size = 250;      # number of neurons

g = 5;          # recurrent connection strength 

#%% normalize inputs and set as tensor arrays

#input_sig = torch.zeros(input_size, T)

#input_mat = np.random.randn(input_size,T)

input_mat_n = input_mat - np.mean(input_mat)
input_mat_n = input_mat_n/np.std(input_mat_n)
input_sig = torch.tensor(input_mat_n[:,0:T]).float()

input_mat_test_n = input_mat_test - np.mean(input_mat_test)
input_mat_test_n = input_mat_test_n/np.std(input_mat_test_n)
input_sig_test = torch.tensor(input_mat_test_n[:,0:T_test]).float()

target = torch.tensor(output_mat[:,0:T]).float()
target_test = torch.tensor(output_mat_test[:,0:T_test]).float()


plt.figure()
plt.plot(np.std(input_mat_n, axis=0))
plt.title('std of inputs vs time')

#%% initialize RNN
rnn = RNN_chaotic(input_size, hidden_size, output_size, alpha)

#%% initialize rate and weights

rate = rnn.init_rate();
rate_test = rnn.init_rate();

rnn.init_weights(g)

# can adjust bias here 
#rnn.h2h.bias.data  = rnn.h2h.bias.data -2
#np.std(np.asarray(rnn.h2h.weight ).flatten())

#%% plot RNN parameters

f_plot_rnn_params(rnn, rate, input_sig, text_tag = 'initial ')

w1 = np.asarray(rnn.h2h.weight.data);

np.std(w1)

#%% set learning params

loss = nn.NLLLoss()

iteration1 = [];
iteration1.append(0);

iteration_test = []
iteration_test.append(0);

if train_RNN:
    learning_rate = 0.005
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)   

#%%
num_cycles = 1;

rates_all = np.zeros((hidden_size, T));
outputs_all = np.zeros((output_size, T));
loss_all = np.zeros((T));

rates_all_test = np.zeros((hidden_size, T_test))
outputs_all_test = np.zeros((output_size, T_test));
loss_all_test = np.zeros((T_test));

#%%
if load_RNN:
    rnn.load_state_dict(torch.load(path1 + '/RNN_data/' + fname_RNN_load))

#%% train

for n_t in range(T-1):
    
    if train_RNN:
        optimizer.zero_grad()
    
    output, rate_new = rnn.forward(input_sig[:,n_t], rate)
    
    rates_all[:,n_t+1] = rate_new.detach().numpy()[0,:];
    
    rate = rate_new.detach();

    outputs_all[:,n_t+1] = output.detach().numpy()[0,:];
    
    target2 = torch.argmax(target[:,n_t]) * torch.ones(1) # torch.tensor()
    loss2 = loss(output, target2.long())
    
    if train_RNN:
        loss2.backward() # retain_graph=True
        optimizer.step()
        
    loss_all[n_t] = loss2.item()
    iteration1.append(iteration1[-1]+1);

print('Done')
    
#%%
if save_RNN and train_RNN:
    torch.save(rnn.state_dict(), path1 + '/RNN_data/' + fname_RNN_save)

#%% test

for n_t in range(T_test-1):
    
    output, rate_new = rnn.forward(input_sig_test[:,n_t], rate_test)
    
    rates_all_test[:,n_t+1] = rate_new.detach().numpy()[0,:];
    
    rate_test = rate_new.detach();

    outputs_all_test[:,n_t+1] = output.detach().numpy()[0,:];
    
    target2 = torch.argmax(target_test[:,n_t]) * torch.ones(1) # torch.tensor()
    loss2 = loss(output, target2.long())
      
    loss_all_test[n_t] = loss2.item()
    iteration_test.append(iteration_test[-1]+1);

print('Done')
    

#%%
f_plot_rnn_params(rnn, rate, input_sig, text_tag = 'final ')

#%%

sm_bin = round(1/dt)*50;

kernel = np.ones(sm_bin)/sm_bin

loss_all_smooth = np.convolve(loss_all, kernel, mode='valid')

loss_all_test_smooth = np.convolve(loss_all_test, kernel, mode='valid')

plt.figure()
plt.plot(loss_all_smooth)
plt.plot(loss_all_test_smooth)
plt.legend(('train', 'test'))

#%%

f_plot_rates(rates_all, input_sig, target, outputs_all, loss_all, 'train')

f_plot_rates(rates_all_test, input_sig_test, target_test, outputs_all_test, loss_all_test, 'test')


#%%




#%% analyze tuning 

plt.close('all')


n_cell = 0
n_tr = 0

num_cells = hidden_size;

trial_ave_win = [-5,15]

num_t = trial_ave_win[1] - trial_ave_win[0]

stim_times = np.diff(output_mat_test[1,:], prepend=0)
on_times = np.where(np.greater(stim_times, 0))[0]

colors1 = plt.cm.jet(np.linspace(0, 1, num_stim))

plot_t = np.asarray(range(trial_ave_win[0], trial_ave_win[1]))

on_times_all = []
num_trials_all = np.zeros((num_stim), dtype=int)
for n_stim in range(num_stim):
    stim_times = np.diff(output_mat_test[n_stim+1,:], prepend=0)
    on_times_trace = np.greater(stim_times, 0)
    on_times = np.where(on_times_trace)[0]
    
    on_times_all.append(on_times)
    num_trials_all[n_stim] = len(on_times)


trial_all = []
trial_ave_all = np.zeros((num_cells, num_stim, num_t))

for n_cell in range(num_cells):
    cell_trace = rates_all_test[n_cell,:]
    
    trial_all_stim = []
    
    for n_stim in range(num_stim):
        on_times = on_times_all[n_stim]
        num_tr = num_trials_all[n_stim]
        
        trial_all2 = np.zeros((num_tr, num_t))
        
        for n_tr in range(num_tr):
            trial_all2[n_tr, :] = cell_trace[(on_times[n_tr] + trial_ave_win[0]):(on_times[n_tr] + trial_ave_win[1])]
        trial_all_stim.append(trial_all2)
        
        trial_ave2 = np.sum(trial_all2, axis=0)
        base = np.mean(trial_ave2[:-trial_ave_win[0]])
        
        trial_ave_all[n_cell, n_stim, :] = trial_ave2 - base
        
    trial_all.append(trial_all_stim)

trial_resp_all = np.mean(trial_ave_all[:,:,-trial_ave_win[0]:(-trial_ave_win[0]+stim_bins)], axis=2)

trial_max_idx = np.argmax(trial_resp_all, axis=1)
idx1_sort = trial_max_idx.argsort()

trial_resp_all_sort = trial_resp_all[idx1,:]

plt.figure()
plt.imshow(trial_resp_all_sort, aspect="auto")

plt.figure()
for n_st in range(num_stim):
    pop_ave = trial_resp_all[trial_max_idx == n_st, :].mean(axis=0)
    x_lab = np.arange(num_stim) - n_st

    plt.plot(x_lab, pop_ave)



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

