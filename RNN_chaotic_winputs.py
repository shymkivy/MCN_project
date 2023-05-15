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

params = {'train_type':             'freq_oddball',     # standard, linear, oddball, freq_oddball
          'num_train_trials':       20,             # for linear 20000; for standard 20, for ~400 bins total
          'num_train_bouts':        500,            # 1 for linert; 10for standard many
          'bout_reinit_rate':       0,
          'bout_num_iterations':    1,
          'dt':                     0.05,
          'num_test_trials':        200,
          'stim_duration':          0.5,
          'isi_suration':           0.5,
          'input_size':             50,
          'hidden_size':            250,            # number of RNN neurons
          'g':                      5,              # recurrent connection strength 
          'learning_rate':          0.05,           # 0.005
          'num_stim':               20,
          'num_ctx':                2,
          'tau':                    0.5,
          'stim_t_std':             3,              # 3 or 0
          'input_noise_std':        1/100,
          'oddball_stim':           [1,2,3,4,5, 6,7,8,9,10],
          'dd_frac':                0.1,
          'plot_deets':             0,
          }

now1 = datetime.now()

name_tag = '%s_%dtrials_%dstim_std%.0f_%d_%d_%d_%dh_%dm' % (params['train_type'], params['num_train_trials'],
            params['num_stim'], params['stim_t_std'], now1.year, now1.month, now1.day, now1.hour, now1.minute)

#%%

#fname_RNN_load = 'test_20k_std3'
#fname_RNN_load = '50k_20stim_std3';
fname_RNN_load = name_tag  + '_RNN'

#fname_RNN_save = 'test_50k_std4'
#fname_RNN_save = '50k_20stim_std3'
fname_RNN_save = name_tag

#%% generate train data

#plt.close('all')

# generate stim templates

stim_temp_all, out_temp_all = f_gen_stim_output_templates(params)


out_temp_ctx_all = out_temp_all[1:3,:,1:3]

stim_temp2, out_temp2 = f_gen_stim_output_templates_thin(params)


oddball_stim = params['oddball_stim'].copy()
oddball_stim_flip = params['oddball_stim'].copy()
oddball_stim_flip.reverse()


trials_train_cont = f_gen_cont_seq(params['num_stim'], params['num_train_trials'], params['num_train_bouts'])

trials_train_oddball_freq, trials_train_oddball_ctx = f_gen_oddball_seq(oddball_stim, params['num_train_trials'], params['dd_frac'], params['num_train_bouts'])

trials_test_cont = f_gen_cont_seq(params['num_stim'], params['num_test_trials'], 1)
trials_test_oddball, _ = f_gen_oddball_seq(oddball_stim, params['num_test_trials'], params['dd_frac'], 1)
trials_test_oddball_flip, _ = f_gen_oddball_seq(oddball_stim_flip, params['num_test_trials'], params['dd_frac'], 1)



input_train_cont, output_train_cont = f_gen_input_output_from_seq(trials_train_cont, stim_temp_all, out_temp_all, params)


input_train_oddball_freq, output_train_oddball_freq = f_gen_input_output_from_seq(trials_train_oddball_freq, stim_temp_all, out_temp_all, params)

_, output_train_oddball_ctx = f_gen_input_output_from_seq(trials_train_oddball_ctx, stim_temp_all, out_temp_ctx_all, params)


input_test_cont, output_test_cont = f_gen_input_output_from_seq(trials_test_cont, stim_temp_all, out_temp_all, params)

input_test_oddball, output_test_oddball = f_gen_input_output_from_seq(trials_test_oddball, stim_temp_all, out_temp_all, params)

input_test_oddball_flip, output_test_oddball_flip = f_gen_input_output_from_seq(trials_test_oddball_flip, stim_temp_all, out_temp_all, params)


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
    plt.title('inputs; %d stim; %d intups; std=%.1f' % (num_stim, params['input_size'], params['stim_t_std']))
    plt.subplot(spec2[1], sharex=ax1)
    plt.imshow(output_plot, aspect="auto")
    plt.title('outputs')
    
    plt.figure()
    plt.plot(np.mean(input_plot, axis=1))
    plt.title('mean spectrogram across time')
    plt.xlabel('inputs')
    plt.ylabel('mean power')

#%% initialize RNN 

output_size = params['num_stim'] + 1
output_size_ctx = params['num_ctx']
hidden_size = params['hidden_size'];
alpha = params['dt']/params['tau'];         

rnn = RNN_chaotic(params['input_size'], params['hidden_size'], output_size, output_size_ctx, alpha)
rnn.init_weights(params['g'])

#loss = nn.NLLLoss()
loss = nn.CrossEntropyLoss()
loss_ctx = nn.CrossEntropyLoss(weight = torch.tensor([0.1, 0.9]))

#%%
if load_RNN:
    print('Loading RNN %s' % fname_RNN_load)
    rnn.load_state_dict(torch.load(path1 + '/RNN_data/' + fname_RNN_load))

#%%
if train_RNN:
    if params['train_type'] == 'standard':
        train_cont = f_RNN_trial_train(rnn, loss, input_train_cont, output_train_cont, params)
    elif params['train_type'] == 'freq_oddball':
        train_cont = f_RNN_trial_freq_ctx_train(rnn, loss, loss_ctx, input_train_oddball_freq, output_train_oddball_freq, output_train_oddball_ctx, params)
    elif params['train_type'] == 'oddball':
        train_cont = f_RNN_trial_ctx_train(rnn, loss_ctx, input_train_oddball_freq, output_train_oddball_ctx, params)
        
        #train_cont = f_RNN_trial_ctx_train(rnn, loss, input_train_oddball_freq, output_train_oddball_freq, output_train_oddball_ctx, params)
    
    else:
        train_cont = f_RNN_linear_train(rnn, loss, input_train_cont, output_train_cont, params)
        
else:
    print('running without training')
    train_cont = f_RNN_test(rnn, loss, input_train_cont, output_train_cont, params)
    
    
# f_plot_rates(rates_all[:,:, 1], 10)
 #%%
if save_RNN and train_RNN:
    print('Saving RNN %s' % fname_RNN_save)
    torch.save(rnn.state_dict(), path1 + '/RNN_data/' + fname_RNN_save  + '_RNN')
    np.save(path1 + '/RNN_data/' + fname_RNN_save + '_params.npy', params) 

#%% test
test_cont = f_RNN_test(rnn, loss, input_test_cont, output_test_cont, params)

#%% test oddball
test_oddball = f_RNN_test(rnn, loss, input_test_oddball, output_test_oddball, params)

#%% test oddball
train_oddball = f_RNN_test(rnn, loss_ctx, input_train_oddball_freq, output_train_oddball_ctx, params)


#%% test oddball flip
test_oddball_flip = f_RNN_test(rnn, loss, input_test_oddball_flip, output_test_oddball_flip, params)

#%%

sm_bin = round(1/params['dt'])*50;
trial_len = out_temp_all.shape[1]
kernel = np.ones(sm_bin)/sm_bin

loss_train = train_cont['loss'].T.flatten()

loss_train_cont_sm = np.convolve(loss_train, kernel, mode='valid')
loss_test_cont_sm = np.convolve(test_cont['loss'], kernel, mode='valid')

loss_x = np.arange(len(loss_train_cont_sm))/(trial_len)
loss_x_raw = np.arange(len(loss_train))/(trial_len)
loss_x_test = np.arange(len(loss_test_cont_sm))/(trial_len)


plt.figure()
plt.plot(loss_x_raw, loss_train)
plt.plot(loss_x_test, loss_test_cont_sm)
plt.legend(('train', 'test'))
plt.xlabel('trials')
plt.ylabel('NLL loss')
plt.title(fname_RNN_save)


plt.figure()
plt.plot(loss_x, loss_train_cont_sm)
plt.plot(loss_x_test, loss_test_cont_sm)
plt.legend(('train', 'test'))
plt.xlabel('trials')
plt.ylabel('NLL loss')
plt.title(fname_RNN_save)

#%%

f_plot_rates(train_cont, input_train_cont, output_train_cont, 'train')

f_plot_rates(test_cont, input_test_cont, output_test_cont, 'test cont')

f_plot_rates(test_oddball, input_test_oddball, output_test_oddball, 'test oddball')

f_plot_rates(test_oddball_flip, input_test_oddball_flip, output_test_oddball_flip, 'test oddball flip')


#%% analyze tuning 

plt.close('all')


trial_len = out_temp_all.shape[1]
output_calc = output_test_cont
rates_calc = test_cont['rates']

num_cells = params['hidden_size'];

trial_ave_win = [-5,15]

num_t = trial_ave_win[1] - trial_ave_win[0]

stim_times = np.diff(output_calc[1,:], prepend=0)
on_times = np.where(np.greater(stim_times, 0))[0]

colors1 = plt.cm.jet(np.linspace(0, 1, num_stim))

plot_t = np.asarray(range(trial_ave_win[0], trial_ave_win[1]))

on_times_all = []
num_trials_all = np.zeros((num_stim), dtype=int)
for n_stim in range(num_stim):
    stim_times = np.diff(output_calc[n_stim+1,:], prepend=0)
    on_times_trace = np.greater(stim_times, 0)
    on_times = np.where(on_times_trace)[0]
    
    on_times_all.append(on_times)
    num_trials_all[n_stim] = len(on_times)


trial_all = []
trial_ave_all = np.zeros((num_cells, num_stim, num_t))

for n_cell in range(num_cells):
    cell_trace = rates_calc[n_cell,:]
    
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

trial_resp_all = np.mean(trial_ave_all[:,:,-trial_ave_win[0]:(-trial_ave_win[0]+trial_len)], axis=2)

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






