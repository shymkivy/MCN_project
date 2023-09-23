# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 14:55:50 2021

@author: ys2605
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from random import sample, random

from sklearn.decomposition import PCA

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
    
    outputs_all = rnn_data['output']
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
    #plt.axis('off')
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

def f_plot_rates2(rnn_data, title_tag, num_plot_batches = 1, num_plot_cells = 10, randomize=True):
    
    rates = rnn_data['rates']
    input_sig = rnn_data['input']
    output = rnn_data['output']
    target = rnn_data['target']
    
    T, batch_size, num_cells = rates.shape
    
    num_plot_batches2 = min(num_plot_batches, batch_size)
    
    if 'lossT' in rnn_data.keys():
        lossT = rnn_data['lossT']
        num_sp = 7
        height_ratios1 = [4, 1, 2, 2, 2, 8, 8]
    else:
        num_sp = 6
        height_ratios1 = [4, 1, 2, 2, 2, 1]
    
    
    if 1:
        ratesn = rates - np.mean(rates, axis=0)
        ratesn = ratesn/np.std(ratesn, axis=0)/3
    else:
        ratesn = rates
    
    spec = gridspec.GridSpec(ncols=1, nrows=num_sp, height_ratios=height_ratios1)
    
    if randomize:
        plot_batches = np.sort(sample(range(batch_size), num_plot_batches2))
    else:
        plot_batches = range(batch_size)[:num_plot_batches2]
    
    for n_bt in range(num_plot_batches2):
        bt = plot_batches[n_bt]
        
        plot_cells = np.sort(sample(range(num_cells), num_plot_cells))
        
        plt.figure()
        ax1 = plt.subplot(spec[0])
        for n_plt in range(num_plot_cells):  
            shift = n_plt*2.5    
            ax1.plot(ratesn[:,bt,plot_cells[n_plt]]+shift)
        plt.title('%s; batch %d; example cells' % (title_tag, bt))
        # plt.axis('off')
        #plt.xticks([])
        plt.subplot(spec[1], sharex=ax1)
        plt.plot(np.mean(rates[:,bt,:], axis=1))
        plt.ylabel('population average')
        plt.axis('off')
        plt.subplot(spec[2], sharex=ax1)
        plt.imshow(input_sig[:,bt,:].T, aspect="auto") #   , aspect=10
        plt.title('inputs')
        plt.axis('off')
        plt.subplot(spec[3], sharex=ax1)
        plt.imshow(target[:,bt,:].T, aspect="auto", interpolation='none') # , aspect=100
        plt.title('target')
        plt.axis('off')
        plt.subplot(spec[4], sharex=ax1)
        plt.imshow(output[:,bt,:].T, aspect="auto", interpolation='none') # , aspect=100
        plt.title('outputs')
        plt.axis('off')
        plt.subplot(spec[5], sharex=ax1)
        plt.plot(output[:,bt,:]) # , aspect=100
        plt.title('outputs 2')
        plt.axis('off')
        if 'lossT' in rnn_data.keys():
            plt.subplot(spec[6], sharex=ax1)
            plt.plot(lossT[:,bt]) # , aspect=100
            plt.title('loss')
            plt.axis('off')

#%% 
def f_plot_rates_only(rnn_data, title_tag = '', num_plot_batches = 1, num_plot_cells = 10, preprocess = True, norm_std_fac = 6, start_from = 0, plot_extra = 0):
    rates = rnn_data['rates']
    
    #rates = test_spont['rates']
    
    
    T, batch_size, num_cells = rates.shape
    
    rates2 = rates[start_from:,:,:]
    
    means1 = np.mean(rates2, axis=0)
    
    stds1 = np.std(rates2, axis=0)
    
    
    plt.figure()
    plt.plot(means1)
    plt.xlabel('batches')
    plt.ylabel('mean magnitude')
    plt.title('cell means across batches')
    
    plt.figure()
    plt.plot(stds1)
    plt.xlabel('batches')
    plt.ylabel('std magnitude')
    plt.title('cell stds across batches')
    
    
    # plt.figure()
    # plt.plot(stds1, means1, 'o')
    
    num_plot_batches2 = min(num_plot_batches, batch_size)
    
    plot_batches = np.sort(sample(range(batch_size), num_plot_batches2))
    
    for n_bt in range(num_plot_batches2):
        bt = plot_batches[n_bt]
        
        rates3 = rates2[:,bt,:]
        
        #rates3 = rates[:,bt,:]
        
        if preprocess:
            rates3n = rates3 - means1[bt,:]
        
            stds2 = stds1[bt,:]
            idx1 = stds2 > 0
        
            rates3n[:,idx1] = rates3n[:,idx1]/stds2[idx1]/norm_std_fac
        else:
            rates3n = rates3
        
        plot_cells = np.sort(sample(range(num_cells), num_plot_cells))
        
        plt.figure()
        for n_plt in range(num_plot_cells):  
            shift = n_plt    
            plt.plot(rates3n[:,plot_cells[n_plt]]+shift)
        plt.title('%s; batch %d; example cells' % (title_tag, bt))
        
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
    
    
    