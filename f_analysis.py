# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 14:55:50 2021

@author: ys2605
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from random import sample, random

#%% for getting sorting order from linkage
def seriation(Z,N,cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z
            
        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))
    
    
#%%

def f_plot_rates(rnn_data, input_sig, target, title_tag):
    
    rates_all = rnn_data['rates']
    
    outputs_all = rnn_data['outputs']
    if 'lossT' in rnn_data.keys():
        loss_all = rnn_data['lossT']
    else:
        loss_all = rnn_data['loss']
    
    shape1 = rates_all.shape
    
    iter1 = 0
    
    if len(shape1) == 4:
        rates_all = rates_all[:,:,iter1,-1]
        if 'lossT' in rnn_data.keys():
            loss_all = loss_all[:,iter1,-1]
        else:
            loss_all = loss_all[iter1,-1]
        outputs_all = outputs_all[:,:,iter1,-1]
        input_sig = input_sig[:,:,-1]
        target = target[:,:,-1]
        name_tag = 'trial train; bout%d; iter%d' % (shape1[3], iter1)
    else:
        name_tag = 'linear train'
    
    num_plots = 10;
    
    plot_cells = np.sort(sample(range(rates_all.shape[0]), num_plots));
    spec = gridspec.GridSpec(ncols=1, nrows=6, height_ratios=[4, 1, 2, 2, 2, 1])
    
    plt.figure()
    ax1 = plt.subplot(spec[0])
    for n_plt in range(num_plots):  
        shift = n_plt*2.5    
        ax1.plot(rates_all[plot_cells[n_plt],:]+shift)
    plt.title(title_tag + ' example cells' + name_tag)
    plt.axis('off')
   # plt.xticks([])
    plt.subplot(spec[1], sharex=ax1)
    plt.plot(np.mean(rates_all, axis=0))
    plt.title('population average')
    plt.axis('off')
    plt.subplot(spec[2], sharex=ax1)
    plt.imshow(input_sig.data, aspect="auto") #   , aspect=10
    plt.title('inputs')
    plt.axis('off')
    plt.subplot(spec[3], sharex=ax1)
    plt.imshow(target.data, aspect="auto") # , aspect=100
    plt.title('target')
    plt.axis('off')
    plt.subplot(spec[4], sharex=ax1)
    plt.imshow(outputs_all, aspect="auto") # , aspect=100
    plt.title('outputs')
    plt.axis('off')
    plt.subplot(spec[5], sharex=ax1)
    plt.plot(loss_all) # , aspect=100
    plt.title('loss')
    plt.axis('off')

#%%

def f_plot_rates_ctx(rnn_data, input_sig, target, title_tag):
    
    rates_all = rnn_data['rates']
    
    outputs_all = rnn_data['outputs_ctx']
    if 'lossT' in rnn_data.keys():
        loss_all = rnn_data['lossT']
    else:
        loss_all = rnn_data['loss_ctx']
    
    shape1 = rates_all.shape
    
    iter1 = 0
    
    if len(shape1) == 4:
        rates_all = rates_all[:,:,iter1,-1]
        if 'lossT' in rnn_data.keys():
            loss_all = loss_all[:,iter1,-1]
        else:
            loss_all = loss_all[iter1,-1]
        outputs_all = outputs_all[:,:,iter1,-1]
        input_sig = input_sig[:,:,-1]
        target = target[:,:,-1]
        name_tag = 'trial train; bout%d; iter%d' % (shape1[3], iter1)
    else:
        name_tag = 'linear train'
    
    num_plots = 10;
    
    plot_cells = np.sort(sample(range(rates_all.shape[0]), num_plots));
    spec = gridspec.GridSpec(ncols=1, nrows=6, height_ratios=[4, 1, 2, 2, 2, 1])
    
    plt.figure()
    ax1 = plt.subplot(spec[0])
    for n_plt in range(num_plots):  
        shift = n_plt*2.5    
        ax1.plot(rates_all[plot_cells[n_plt],:]+shift)
    plt.title(title_tag + ' example cells' + name_tag)
    plt.axis('off')
   # plt.xticks([])
    plt.subplot(spec[1], sharex=ax1)
    plt.plot(np.mean(rates_all, axis=0))
    plt.title('population average')
    plt.axis('off')
    plt.subplot(spec[2], sharex=ax1)
    plt.imshow(input_sig.data, aspect="auto") #   , aspect=10
    plt.title('inputs')
    plt.axis('off')
    plt.subplot(spec[3], sharex=ax1)
    plt.imshow(target.data, aspect="auto") # , aspect=100
    plt.title('target')
    plt.axis('off')
    plt.subplot(spec[4], sharex=ax1)
    plt.imshow(outputs_all, aspect="auto") # , aspect=100
    plt.title('outputs')
    plt.axis('off')
    plt.subplot(spec[5], sharex=ax1)
    plt.plot(loss_all) # , aspect=100
    plt.title('loss')
    #plt.axis('off')

#%%
def f_plot_rnn_params(rnn, rate, input_sig, text_tag=''):
    n_hist_bins = 20;
    
    w1 = np.asarray(rnn.h2h.weight.data).flatten();
    w2 = np.asarray(rnn.i2h.weight.data).flatten();
    r1 = np.asarray(rate).flatten()
    i1 = np.asarray(input_sig).flatten();
    
    plt.figure()
    plt.subplot(4,1,1);
    plt.hist(w1,bins=n_hist_bins);
    plt.title(text_tag + 'h2h weights; std=%.2f' % np.std(w1))
    plt.subplot(4,1,2);
    plt.hist(w2,bins=n_hist_bins);
    plt.title(text_tag + 'i2h weights; std=%.2f' % np.std(w2))
    plt.subplot(4,1,3);
    plt.hist(r1,bins=n_hist_bins);
    plt.title(text_tag + 'rates; std=%.2f' % np.std(r1))
    plt.subplot(4,1,4);
    plt.hist(i1,bins=n_hist_bins);
    plt.title(text_tag + 'inputs; std=%.2f' % np.std(i1))
    
    
    