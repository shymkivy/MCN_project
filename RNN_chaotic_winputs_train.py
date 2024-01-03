# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 15:33:58 2021

@author: Administrator
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
sys.path.append(path1 + 'RNN_scripts')

from f_RNN import f_RNN_trial_ctx_train2, f_RNN_trial_freq_train2 #, f_plot_rnn_weights#, f_trial_ave_pad
from f_RNN_chaotic import RNN_chaotic
from f_RNN_utils import f_gen_stim_output_templates, f_plot_train_loss


import numpy as np
#from random import sample, random
import torch
import torch.nn as nn

from datetime import datetime

#%% input params

params = {'train_type':                     'freq2',     #   oddball2, freq2  standard, linear, oddball, freq_oddball,
          'device':                         'cuda',         # 'cpu', 'cuda'
          
          'stim_duration':                  0.5,
          'isi_duration':                   0.5,
          'num_freq_stim':                  50,
          'num_ctx':                        1,
          'oddball_stim':                   np.arange(50)+1, # np.arange(10)+1, #[3, 6], #np.arange(10)+1,
          'dd_frac':                        0.1,
          'dt':                             0.05,
          
          'train_batch_size':               100,
          'train_trials_in_sample':         20,
          'train_num_samples':              40000,
          'train_loss_weights':             [0.05, 0.95], # isi, red, dd [1e-5, 1e-5, 1] [0.05, 0.05, 0.9], [0.05, 0.95]  [1/.5, 1/.45, 1/0.05]
          'train_add_noise':                1,               # sqrt(2*dt/tau*sigma_req^2) * norm(0,1)

          'train_repeats_per_samp':         1,
          'train_reinit_rate':              0,
          
          'test_batch_size':                100,
          'test_trials_in_sample':          400,
          'test_oddball_stim':              np.arange(10)+1,        #[3, 5, 7],
          'test_num_freq_stim':             10,
          
          'input_size':                     50,
          'hidden_size':                    25,            # number of RNN neurons
          'g':                              1,  # 1            # recurrent connection strength 
          'tau':                            .5,
          'learning_rate':                  0.001,           # 0.005
          'activation':                     'ReLU',             # ReLU tanh
          'normalize_input':                False,
          
          'stim_t_std':                     3,              # 3 or 0
          'input_noise_std':                1/100,
          
          'plot_deets':                     0,
          }

now1 = datetime.now()
params['train_date'] = now1


#%%

if 'train_date' in params.keys():
    now2 = params['train_date']
else:
    now2 = now1
if 'activation' not in params.keys():
    params['activation'] = 'tanh'
        

save_tag = ''

name_tag1 = '%s%s_%dctx_%dtrainsamp_%dneurons_%s_%dtau_%ddt' % (save_tag, params['train_type'], params['num_ctx'],
            params['train_num_samples'], params['hidden_size'], params['activation'], params['tau']*1000, params['dt']*1000)

name_tag2 = '%dtrials_%dstim_%dbatch_%.4flr_noise%d_%d_%d_%d_%dh_%dm' % (params['train_trials_in_sample'], params['num_freq_stim'], params['train_batch_size'], params['learning_rate'], params['train_add_noise'],
             now2.year, now2.month, now2.day, now2.hour, now2.minute)

name_tag  = name_tag1 + '_' + name_tag2

fname_RNN_save = name_tag

#%% generate train data
# generate stim templates

stim_templates = f_gen_stim_output_templates(params)
trial_len = round((params['stim_duration'] + params['isi_duration'])/params['dt'])

#%% initialize RNN 

if 'device' not in params.keys():
    params['device'] = 'cpu'

output_size = params['num_freq_stim'] + 1
output_size_ctx = params['num_ctx'] + 1
hidden_size = params['hidden_size'];
alpha = params['dt']/params['tau'];         

rnn = RNN_chaotic(params['input_size'], params['hidden_size'], output_size, output_size_ctx, alpha, params['train_add_noise'], activation=params['activation']).to(params['device'])
rnn.init_weights(params['g'])


#%%
if 'train_loss_weights' not in params.keys():
    params['train_loss_weights'] = [0.1, 0.1, 0.9]

#loss = nn.NLLLoss()

loss_freq = nn.CrossEntropyLoss()

loss_ctx = nn.CrossEntropyLoss(weight = torch.tensor(params['train_loss_weights']).to(params['device']))

# if params['num_ctx'] > 1:
#     loss_ctx = nn.CrossEntropyLoss(weight = torch.tensor(params['train_loss_weights']).to(params['device']))  #1e-10
# else:
#     loss_ctx = nn.BCELoss(weight = torch.tensor(params['train_loss_weights']).to(params['device']))

train_out = {}     # initialize outputs, so they are saved when process breaks

#%% training

if params['train_type'] == 'oddball2':
    
    f_RNN_trial_ctx_train2(rnn, loss_ctx, stim_templates, params, train_out)
    
elif params['train_type'] == 'freq2':
    
    f_RNN_trial_freq_train2(rnn, loss_freq, stim_templates, params, train_out)
    
# elif params['train_type'] == 'standard':
#     train_out = f_RNN_trial_train(rnn, loss, input_train_cont, output_train_cont, params)
# elif params['train_type'] == 'freq_oddball':
#     train_out = f_RNN_trial_freq_ctx_train(rnn, loss, loss_ctx, input_train_oddball, output_train_oddball_freq, output_train_oddball_ctx, params)
# elif params['train_type'] == 'oddball':
#     train_out = f_RNN_trial_ctx_train(rnn, loss_ctx, input_train_oddball, output_train_oddball_ctx, params)
    
#     #train_cont = f_RNN_trial_ctx_train(rnn, loss, input_train_oddball_freq, output_train_oddball_freq, output_train_oddball_ctx, params)

# else:
#     train_out = f_RNN_linear_train(rnn, loss, input_train_cont, output_train_cont, params)
    
  
#%%
#plt.close('all')
figs = f_plot_train_loss(train_out, name_tag1, name_tag2)
    
# f_plot_rates(rates_all[:,:, 1], 10)

#%% saving

print('Saving RNN %s' % fname_RNN_save)
torch.save(rnn.state_dict(), path1 + '/RNN_data/' + fname_RNN_save  + '_RNN')
np.save(path1 + '/RNN_data/' + fname_RNN_save + '_params.npy', params) 
np.save(path1 + '/RNN_data/' + fname_RNN_save + '_train_out.npy', train_out) 

for key1 in figs.keys():
    figs[key1].savefig(path1 + '/RNN_data/' + fname_RNN_save + '_' + key1 + '.png', dpi=1200)

