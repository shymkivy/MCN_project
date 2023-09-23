# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 12:27:38 2023

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

from f_analysis import *
from f_RNN import *
from f_RNN_chaotic import *
from f_RNN_utils import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import colors
import matplotlib.cm as cm


#%%


flist = [#'oddball2_1ctx_20000trainsamp_25neurons_ReLU_20trials_50stim_100batch_0.0010lr_2023_8_14_13h_42m_RNN',
         'oddball2_1ctx_80000trainsamp_25neurons_ReLU_20trials_50stim_100batch_0.0010lr_2023_8_16_18h_48m_RNN',
         'oddball2_1ctx_80000trainsamp_25neurons_ReLU_0.10tau_20trials_50stim_100batch_0.0010lr_2023_9_13_11h_41m_RNN',
         'oddball2_1ctx_80000trainsamp_25neurons_ReLU_0.50tau_20trials_50stim_100batch_0.0010lr_2023_9_11_14h_19m_RNN',
         'oddball2_1ctx_80000trainsamp_25neurons_ReLU_1.00tau_20trials_50stim_100batch_0.0010lr_2023_9_14_11h_58m_RNN',
         'oddball2_1ctx_80000trainsamp_25neurons_ReLU_2.00tau_20trials_50stim_100batch_0.0010lr_2023_9_15_12h_18m_RNN',
         'oddball2_1ctx_160000trainsamp_25neurons_ReLU_1.00tau_20trials_50stim_100batch_0.0010lr_2023_9_16_13h_10m_RNN',
         'oddball2_1ctx_160000trainsamp_25neurons_ReLU_0.30tau_20trials_50stim_100batch_0.0010lr_2023_9_17_14h_33m_RNN',
         'oddball2_1ctx_160000trainsamp_25neurons_ReLU_0.10tau_20trials_50stim_100batch_0.0010lr_2023_9_18_16h_17m_RNN']        # dt = 0.01





max_it_plot = 80000


sm_bin = 100#round(1/params['dt'])*50;
kernel = np.ones(sm_bin)/sm_bin


num_files = len(flist)

#%% load all

param_all = []
train_out_all = []
tau_all = np.zeros((num_files))

for n_fil in range(num_files):


    params = np.load(path1 + '/RNN_data/' + flist[n_fil][:-4] + '_params.npy', allow_pickle=True).item()
    
    train_out = np.load(path1 + '/RNN_data/' + flist[n_fil][:-4] + '_train_out.npy', allow_pickle=True).item()
    
    tau_all[n_fil] = params['tau']
    param_all.append(params)
    train_out_all.append(train_out)
    
    
idx1 = np.argsort(tau_all)
#%%

# plt.close('all')

col_vals = np.linspace(np.min(tau_all),np.max(tau_all),100)
colors1 = cm.jet((col_vals-np.min(col_vals))/(np.max(col_vals)-np.min(col_vals)))


col_idx_all = np.argmin(np.abs(np.reshape(tau_all, [num_files, 1]) - np.reshape(col_vals, [1, 100])), axis=1)

if 0:
    plt.figure()
    plt.imshow(colors1[:,:3].reshape((100,1,3)), aspect=.2)
    plt.ylabel('color map')
    plt.xticks([])
    plt.yticks(np.linspace(0, 99, 10), np.round(np.linspace(np.min(col_vals), np.max(col_vals), 10),1))

#%%

plt.figure()

leg_all = []

n_fil = 0

for n_fil2 in range(num_files):
    n_fil = idx1[n_fil2]
    params = param_all[n_fil]
    train_out = train_out_all[n_fil]
    tau1 = tau_all[n_fil]
     
    leg_all.append(str(tau1))
    
    loss_train = np.asarray(train_out['loss'])
    loss_train_sm = np.convolve(loss_train, kernel, mode='valid')
    
    
    num_it1 = np.min([loss_train_sm.shape[0], max_it_plot])
    
    loss_train_sm2 = loss_train_sm[:num_it1]
    
    #loss_train = np.asarray(train_out_cont['loss']).T.flatten()
    
    loss_x_sm = np.arange(num_it1)+sm_bin/2 #/(trial_len)
    #loss_x_raw = np.arange(len(loss_train)) #/(trial_len)
    
    
    plt.semilogy(loss_x_sm, loss_train_sm2, color=colors1[col_idx_all[n_fil]])
    

plt.xlabel('iterations')
plt.ylabel('loss')
plt.legend(leg_all)


plt.title('train loss vs tau\n')    











