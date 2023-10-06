

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 22:47:28 2023

@author: yuriy
"""
import numpy as np
import torch
import torch.nn as nn

import time

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec

from scipy import linalg
from scipy.spatial.distance import pdist, squareform #, cdist

from f_RNN_utils import f_gen_oddball_seq, f_gen_input_output_from_seq, f_gen_cont_seq

#%%

def f_RNN_linear_train(rnn, loss, input_train, output_train, params):
    
    hidden_size = params['hidden_size'];     
    output_size = params['num_freq_stim'] + 1
    learning_rate = params['learning_rate']
    
    rate = rnn.init_rate();
    
    #optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate) 
    optimizer = torch.optim.AdamW(rnn.parameters(), lr=learning_rate) 
    
    # initialize 

    T = input_train.shape[1]

    input_sig = torch.tensor(input_train).float()
    target = torch.tensor(output_train).float()

    rates_all = np.zeros((hidden_size, T));
    outputs_all = np.zeros((output_size, T));
    loss_all = np.zeros((T));

    # can adjust bias here 
    #rnn.h2h.bias.data  = rnn.h2h.bias.data -2
    #np.std(np.asarray(rnn.h2h.weight ).flatten())

    #
    if params['plot_deets']:
        plt.figure()
        plt.plot(np.std(input_test_cont, axis=0))
        plt.title('std of inputs vs time')

        f_plot_rnn_params(rnn, rate, input_sig, text_tag = 'initial ')


    print('Starting linear training')
    
    start_time = time.time()
    
    for n_t in range(T):
        
        optimizer.zero_grad()
        
        output, rate_new = rnn.forward_linear(input_sig[:,n_t], rate)
        
        target2 = torch.argmax(target[:,n_t]) # * torch.ones(1) # torch.tensor()
        
        # crossentropy
        loss2 = loss(output, target2.long())
        output_sm = output
        
        # for nnnlosss
        #output_sm = rnn.softmax1(output)   
        #loss2 = loss(output_sm, target2.long())
        
        rates_all[:,n_t] = rate_new.detach().numpy()
        
        rate = rate_new.detach();

        outputs_all[:,n_t] = output_sm.detach().numpy()
        

        loss2.backward() # retain_graph=True
        optimizer.step()
            
        loss_all[n_t] = loss2.item()
        
        # Compute the running loss every 10 steps
        if (n_t % 1000) == 0:
            print('Step %d/%d, Loss %0.3f, Time %0.1fs' % (n_t, T, loss2.item(), time.time() - start_time))
        
    print('Done')
    
    if params['plot_deets']:
        f_plot_rnn_params(rnn, rate, input_sig, text_tag = 'final ')

    
    rnn_out = {'rates':         rates_all,
               'outputs':       outputs_all,
               'loss':          loss_all,
               }
    return rnn_out
    
#%%

def f_RNN_trial_train(rnn, loss, input_train, output_train, params):
    
    hidden_size = params['hidden_size'];     
    output_size = params['num_freq_stim'] + 1
    reinit_rate = params['train_reinit_rate']
    num_it = params['train_repeats_per_samp']
    learning_rate = params['learning_rate']
    
    #optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate) 
    optimizer = torch.optim.AdamW(rnn.parameters(), lr=learning_rate) 

    # initialize 

    input_size, T, num_bouts = input_train.shape

    input_sig = torch.tensor(input_train).float()
    target = torch.tensor(output_train).float()
    
    
    
    #outputs_all = torch.zeros((output_size, T, num_it, num_bouts))
    #rates_all = torch.zeros((hidden_size, T, num_it, num_bouts))
    
    rates_all = np.zeros((hidden_size, T, num_it, num_bouts));
    outputs_all = np.zeros((output_size, T, num_it, num_bouts));
    loss_all = np.zeros((num_it, num_bouts));
    loss_all_T = np.zeros((T, num_it, num_bouts));

    # can adjust bias here 
    #rnn.h2h.bias.data  = rnn.h2h.bias.data -2
    #np.std(np.asarray(rnn.h2h.weight ).flatten())

    #
    if params['plot_deets']:
        plt.figure()
        plt.plot(np.std(input_train, axis=0))
        plt.title('std of inputs vs time')

        f_plot_rnn_params(rnn, rate, input_sig, text_tag = 'initial ')

    print('Starting trial training')
    
    start_time = time.time()
    
    for n_bt in range(num_bouts):
         
        rate_start = rnn.init_rate()
        
        for n_it in range(num_it):
            
            optimizer.zero_grad()
            
            output, rate = rnn.forward_freq(input_sig[:,:, n_bt], rate_start)
            
            target2 = torch.argmax(target[:,:, n_bt], dim =0) * torch.ones(T)
            
            # for crossentropy
            loss2 = loss(output.T, target2.long())
            output_sm = output
            
            # for nnnlosss
            #output_sm = rnn.softmax(output)        
            #loss2 = loss(output_sm.T, target2.long())
            
            loss2.backward() # retain_graph=True
            optimizer.step()
            
            rates_all[:,:, n_it, n_bt] = rate.detach().numpy()
            outputs_all[:,:, n_it, n_bt] = output_sm.detach().numpy()
            
            loss_all[n_it, n_bt] = loss2.item()
            
            for n_t in range(T):
                loss_all_T[n_t, n_it, n_bt] = loss(output_sm[:,n_t], target2[n_t].long()).item()
            
            if reinit_rate:
                rate_start = rnn.init_rate()
            else:
                rate_start = rate[:,-1].detach()

            # Compute the running loss every 10 steps
            if ((n_it) % 10) == 0:
                print('bout %d, Step %d, Loss %0.3f, Time %0.1fs' % (n_bt, n_it, loss2.item(), time.time() - start_time))


    print('Done')
    
    if params['plot_deets']:
        f_plot_rnn_params(rnn, rate, input_sig, text_tag = 'final ')
        
        plt.figure()
        plt.plot(loss_all)
        plt.title('bouts loss')
    
    rnn_out = {'rates':         rates_all,
               'outputs':       outputs_all,
               'loss':          loss_all,
               'lossT':         loss_all_T,
               }
    return rnn_out   

#%%

def f_RNN_trial_ctx_train(rnn, loss, input_train, output_train_ctx, params):
    
    hidden_size = params['hidden_size'];     
    num_stim = params['num_freq_stim'] + 1
    output_size = params['num_ctx'] + 1
    reinit_rate = params['train_reinit_rate']
    num_it = params['train_repeats_per_samp']
    learning_rate = params['learning_rate']
    
    #optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate) 
    optimizer = torch.optim.AdamW(rnn.parameters(), lr=learning_rate) 

    # initialize 

    input_size, T, num_bouts = input_train.shape

    input_sig = torch.tensor(input_train).float()
    target_ctx = torch.tensor(output_train_ctx).float()
    
    
    if 1: # params['plot_deets']
        plots1 = 4;
        plt.figure()
        for n1 in range(plots1):
            idx1 = np.random.choice(num_bouts)
            plt.subplot(plots1*2, 1, (n1+1)*2-1)
            plt.imshow(input_train[:,:,idx1], aspect="auto")
            plt.title('run %d' % idx1)
            plt.ylabel('input')
            plt.subplot(plots1*2, 1, (n1+1)*2)
            plt.imshow(output_train_ctx[:,:,idx1], aspect="auto")
            plt.ylabel('target')
        plt.show()
    
    
    #outputs_all = torch.zeros((output_size, T, num_it, num_bouts))
    #rates_all = torch.zeros((hidden_size, T, num_it, num_bouts))
    
    rates_all = np.zeros((hidden_size, T, num_it, num_bouts));
    outputs_all = np.zeros((output_size, T, num_it, num_bouts));
    loss_all = np.zeros((num_it, num_bouts));
    loss_all_T = np.zeros((T, num_it, num_bouts));

    # can adjust bias here 
    #rnn.h2h.bias.data  = rnn.h2h.bias.data -2
    #np.std(np.asarray(rnn.h2h.weight ).flatten())

    #
    if params['plot_deets']:
        plt.figure()
        plt.plot(np.std(input_train, axis=0))
        plt.title('std of inputs vs time')

        f_plot_rnn_params(rnn, rate, input_sig, text_tag = 'initial ')

    print('Starting trial training')
    
    start_time = time.time()
    
    for n_bt in range(num_bouts):
         
        rate_start = rnn.init_rate()
        
        for n_it in range(num_it):
            
            optimizer.zero_grad()
            
            output_ctx, rate = rnn.forward_ctx(input_sig[:,:, n_bt], rate_start)
            
            target2_ctx = torch.argmax(target_ctx[:,:, n_bt], dim =0) * torch.ones(T)
            
            output_sm = output_ctx
            
            loss2 = loss(output_ctx.T, target2_ctx.long())
            
            # for nnnlosss
            #output_sm = rnn.softmax(output)        
            #loss2 = loss(output_sm.T, target2.long())
            
            loss2.backward() # retain_graph=True
            optimizer.step()
            
            rates_all[:,:, n_it, n_bt] = rate.detach().numpy()
            outputs_all[:,:, n_it, n_bt] = output_sm.detach().numpy()
            
            loss_all[n_it, n_bt] = loss2.item()
            
            for n_t in range(T):
                loss_all_T[n_t, n_it, n_bt] = loss(output_sm[:,n_t], target2_ctx[n_t].long()).item()
            
            if reinit_rate:
                rate_start = rnn.init_rate()
            else:
                rate_start = rate[:,-1].detach()

            # Compute the running loss every 10 steps
            if ((n_it) % 10) == 0:
                print('bout %d, Step %d, Loss %0.3f, Time %0.1fs' % (n_bt, n_it, loss2.item(), time.time() - start_time))


    print('Done')
    
    if params['plot_deets']:
        f_plot_rnn_params(rnn, rate, input_sig, text_tag = 'final ')
        
        plt.figure()
        plt.plot(loss_all)
        plt.title('bouts loss')
    
    rnn_out = {'rates':         rates_all,
               'outputs':       outputs_all,
               'loss':          loss_all,
               'lossT':         loss_all_T,
               }
    return rnn_out   

#%%

def f_RNN_trial_ctx_train2(rnn, loss, stim_templates, params, rnn_out = {}):
    
    hidden_size = params['hidden_size'];     
    output_size = params['num_ctx'] + 1
    reinit_rate = params['train_reinit_rate']
    num_rep = params['train_repeats_per_samp']
    learning_rate = params['learning_rate']
    
    loss_strat = 1
    
    T = round((params['stim_duration'] + params['isi_duration'])/params['dt'] * params['train_trials_in_sample'])
    num_samp = params['train_num_samples_ctx']
    batch_size = params['train_batch_size']
    
    
    #optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate) 
    optimizer = torch.optim.AdamW(rnn.parameters(), lr=learning_rate) 

    # initialize 
    #rnn_out['loss'] = np.zeros((num_samp, num_rep))
    rnn_out['loss'] = []
    rnn_out['loss_by_tt'] = []
    #rnn_out['rates'] = np.zeros((hidden_size, T, num_rep, num_samp))
    #rnn_out['outputs'] = np.zeros((output_size, T, num_rep, num_samp))

    print('Starting ctx trial training')
    
    start_time = time.time()
    
    for n_samp in range(num_samp):
         
        rate_start = rnn.init_rate(params['train_batch_size']).to(params['device'])
        
        # get sample
        
        trials_train_oddball_freq, trials_train_oddball_ctx, _ = f_gen_oddball_seq(params['oddball_stim'], params['oddball_stim'], params['train_trials_in_sample'], params['dd_frac'], params['num_ctx'], params['train_batch_size'], 1)

        input_train_oddball, _ = f_gen_input_output_from_seq(trials_train_oddball_freq, stim_templates['freq_input'], stim_templates['freq_output'], params)
        _, output_train_oddball_ctx = f_gen_input_output_from_seq(trials_train_oddball_ctx, stim_templates['freq_input'], stim_templates['ctx_output'], params)
        
        input_sig = torch.tensor(input_train_oddball).float().to(params['device'])
        target_ctx = torch.tensor(output_train_oddball_ctx).float().to(params['device'])
        
        for n_rep in range(num_rep):
            
            optimizer.zero_grad()
            
            output_ctx, rate = rnn.forward_ctx(input_sig, rate_start)
            
            target_ctx2 = (torch.argmax(target_ctx, dim =2) * torch.ones(T, batch_size).to(params['device'])).long()
            
            loss2 = f_RNN_trial_ctx_get_loss(output_ctx, target_ctx2, loss, loss_strat)

            # for nnnlosss
            #output_sm = rnn.softmax(output)        
            #loss2 = loss(output_sm.T, target2.long())
            
            loss2.backward() # retain_graph=True
            optimizer.step()
            
            loss_deet = np.zeros((params['num_ctx']+1));
            for n_targ in range(params['num_ctx']+1):
                idx1 = target_ctx2 == n_targ
                target_ctx5 = target_ctx2[idx1]
                output_ctx5 = output_ctx[idx1]
                
                loss3 = loss(output_ctx5, target_ctx5)
                
                loss_deet[n_targ] = loss3.item()
                

            #rnn_out['loss'][n_samp, n_rep] = loss2.item()
            rnn_out['loss'].append(loss2.item())
            rnn_out['loss_by_tt'].append(loss_deet)
            
            rnn_out['rates'] = rate.detach().cpu().numpy()
            rnn_out['input'] = input_sig.detach().cpu().numpy()
            rnn_out['output'] = output_ctx.detach().cpu().numpy()
            rnn_out['target'] = target_ctx.detach().cpu().numpy()
            rnn_out['target_idx'] = target_ctx2.detach().cpu().numpy()
            
            if num_rep>1:
                if reinit_rate:
                    rate_start = rnn.init_rate()
                else:
                    rate_start = rate[:,-1].detach()
                rep_tag = ', rep %d' % n_rep
            else:
                rep_tag = ''
            if params['num_ctx'] == 1:
                ctx_tag = '(non-d,d) = (%.2f, %.2f)' % (loss_deet[0], loss_deet[1])
            elif params['num_ctx'] == 2:
                ctx_tag = '(isi,r,d) = (%.2f, %.2f, %.2f)' % (loss_deet[0], loss_deet[1], loss_deet[2])
            
            if ((n_samp) % 10) == 0:
                
                print('sample %d%s, Loss %0.3f, Time %0.1fs; loss by tt %s' % (n_samp, rep_tag, loss2.item(), time.time() - start_time, ctx_tag))

    print('Done')
    
    if params['plot_deets']:
        
        plt.figure()
        plt.plot(np.std(rnn_out['input'], axis=2))
        plt.title('std of inputs vs time')

        # f_plot_rnn_params(rnn, rnn_out['rates'], rnn_out['input'], text_tag = 'initial ')
        # f_plot_rnn_params(rnn, rnn_out['rates'], rnn_out['input'], text_tag = 'final ')
        
        
    if params['plot_deets']:

        plots1 = 4;
        plt.figure()
        for n1 in range(plots1):
            idx1 = np.random.choice(batch_size)
            plt.subplot(plots1*2, 1, (n1+1)*2-1)
            plt.imshow(rnn_out['input'][:,idx1,:].T, aspect="auto")
            plt.title('run %d' % idx1)
            plt.ylabel('input')
            plt.subplot(plots1*2, 1, (n1+1)*2)
            plt.imshow(rnn_out['output'][:,idx1,:].T, aspect="auto")
            plt.ylabel('target')
        plt.show()
    
    if params['plot_deets']:
        
        plt.figure()
        plt.plot(rnn_out['loss'])
        plt.title('loss')
    
    return rnn_out   

def f_RNN_trial_ctx_get_loss(output_ctx, target_ctx2, loss, loss_strat):
    if loss_strat == 1:
        output_ctx3 = output_ctx.permute((1, 2, 0))
        target_ctx3 = target_ctx2.permute((1, 0))
        
        loss2 = loss(output_ctx3, target_ctx3)
    
    elif loss_strat == 2:
        # probably equivalent to first
        output_ctx3 = output_ctx.reshape((T*batch_size, output_size))
        target_ctx3 = target_ctx2.reshape((T*batch_size))
        
        #output_ctx2 = output_ctx.permute((1, 2, 0))

        loss2 = loss(output_ctx3, target_ctx3)
    else:
        # computes separately and sums after
        loss4 = []
        for n_bt in range(batch_size):
            output_ctx3 = output_ctx[:,n_bt,:]
            target_ctx3 = target_ctx2[:,n_bt]
            loss3 = loss(output_ctx3, target_ctx3)
            loss4.append(loss3)
        
        loss2 = sum(loss4)/batch_size
        
    return loss2

#%%

def f_RNN_trial_freq_train2(rnn, loss, stim_templates, params, rnn_out = {}):
    
    hidden_size = params['hidden_size'];     
    output_size = params['num_freq_stim'] + 1
    reinit_rate = params['train_reinit_rate']
    num_rep = params['train_repeats_per_samp']
    learning_rate = params['learning_rate']
    
    input_size = params['input_size']
    
    loss_strat = 1
    
    T = round((params['stim_duration'] + params['isi_duration'])/params['dt'] * params['train_trials_in_sample'])
    num_samp = params['train_num_samples_freq']
    batch_size = params['train_batch_size']
    
    
    #optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate) 
    optimizer = torch.optim.AdamW(rnn.parameters(), lr=learning_rate) 

    # initialize 
    #rnn_out['loss'] = np.zeros((num_samp, num_rep))
    rnn_out['loss'] = []
    #rnn_out['rates'] = np.zeros((hidden_size, T, num_rep, num_samp))
    #rnn_out['outputs'] = np.zeros((output_size, T, num_rep, num_samp))

    print('Starting freq trial training')
    
    start_time = time.time()
    
    for n_samp in range(num_samp):
         
        rate_start = rnn.init_rate(params['train_batch_size'])
        
        # get sample
        trials_test_cont = f_gen_cont_seq(params['num_freq_stim'], params['train_trials_in_sample'], params['train_batch_size'], 1)
        input_test_cont, output_test_cont = f_gen_input_output_from_seq(trials_test_cont, stim_templates['freq_input'], stim_templates['freq_output'], params)

        input_sig = torch.tensor(input_test_cont).float()
        target = torch.tensor(output_test_cont).float()
        
        for n_rep in range(num_rep):
            
            optimizer.zero_grad()
            
            output, rate = rnn.forward_freq(input_sig, rate_start)
            
            target2 = (torch.argmax(target, dim =2) * torch.ones(T, batch_size)).long()
            
            
            if loss_strat == 1:
                output3 = output.permute((1, 2, 0))
                target3 = target2.permute((1, 0))
                
                loss2 = loss(output3, target3)
            
            elif loss_strat == 2:
                # probably equivalent to first
                target3 = target2.reshape((T*batch_size))
                output3 = output.reshape((T*batch_size, output_size))
                #output_ctx2 = output_ctx.permute((1, 2, 0))
    
                loss2 = loss(output3, target3)
            else:
                # computes separately and sums after
                loss4 = []
                for n_bt in range(batch_size):
                    target3 = target2[:,n_bt]
                    output3 = output[:,n_bt,:]
                    loss3 = loss(output3, target3)
                    loss4.append(loss3)
                
                loss2 = sum(loss4)/batch_size
            
            # for nnnlosss
            #output_sm = rnn.softmax(output)        
            #loss2 = loss(output_sm.T, target2.long())
            
            loss2.backward() # retain_graph=True
            optimizer.step()

            #rnn_out['loss'][n_samp, n_rep] = loss2.item()
            rnn_out['loss'].append(loss2.item())
            
            rnn_out['rates'] = rate.detach().numpy()
            rnn_out['input'] = input_sig.detach().numpy()
            rnn_out['output'] = output.detach().numpy()
            rnn_out['target'] = target.detach().numpy()
            rnn_out['target_idx'] = target2.detach().numpy()
            
            if num_rep>1:
                if reinit_rate:
                    rate_start = rnn.init_rate()
                else:
                    rate_start = rate[:,-1].detach()
                    
                # Compute the running loss every 10 steps
                if ((n_rep) % 10) == 0:
                    print('sample %d, rep %d, Loss %0.3f, Time %0.1fs' % (n_samp, n_rep, loss2.item(), time.time() - start_time))
            else:
                # Compute the running loss every 10 steps
                if ((n_rep) % 10) == 0:
                    print('sample %d, Loss %0.3f, Time %0.1fs' % (n_samp, loss2.item(), time.time() - start_time))

    print('Done')
    
    if params['plot_deets']:
        
        plt.figure()
        plt.plot(np.std(rnn_out['input'], axis=2))
        plt.title('std of inputs vs time')

        # f_plot_rnn_params(rnn, rnn_out['rates'], rnn_out['input'], text_tag = 'initial ')
        # f_plot_rnn_params(rnn, rnn_out['rates'], rnn_out['input'], text_tag = 'final ')
        
        
    if params['plot_deets']:

        plots1 = 4;
        plt.figure()
        for n1 in range(plots1):
            idx1 = np.random.choice(batch_size)
            plt.subplot(plots1*2, 1, (n1+1)*2-1)
            plt.imshow(rnn_out['input'][:,idx1,:].T, aspect="auto")
            plt.title('run %d' % idx1)
            plt.ylabel('input')
            plt.subplot(plots1*2, 1, (n1+1)*2)
            plt.imshow(rnn_out['output'][:,idx1,:].T, aspect="auto")
            plt.ylabel('target')
        plt.show()
    
    if params['plot_deets']:
        
        plt.figure()
        plt.plot(rnn_out['loss'])
        plt.title('loss')
    
    return rnn_out   

#%%

def f_RNN_trial_freq_ctx_train(rnn, loss, loss_ctx, input_train, output_train, output_train_ctx, params):
    
    hidden_size = params['hidden_size'];     
    output_size = params['num_freq_stim'] + 1
    output_size_ctx = params['num_ctx'] + 1
    reinit_rate = params['train_reinit_rate']
    num_it = params['train_repeats_per_samp']
    learning_rate = params['learning_rate']
    
    #optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate) 
    optimizer = torch.optim.AdamW(rnn.parameters(), lr=learning_rate) 

    # initialize 

    input_size, T, num_bouts = input_train.shape

    input_sig = torch.tensor(input_train).float()
    target = torch.tensor(output_train).float()
    target_ctx = torch.tensor(output_train_ctx).float()
    
    
    #outputs_all = torch.zeros((output_size, T, num_it, num_bouts))
    #rates_all = torch.zeros((hidden_size, T, num_it, num_bouts))
    
    rates_all = np.zeros((hidden_size, T, num_it, num_bouts));
    outputs_all = np.zeros((output_size, T, num_it, num_bouts));
    loss_all = np.zeros((num_it, num_bouts));
    loss_all_T = np.zeros((T, num_it, num_bouts));

    # can adjust bias here 
    #rnn.h2h.bias.data  = rnn.h2h.bias.data -2
    #np.std(np.asarray(rnn.h2h.weight ).flatten())

    #
    if params['plot_deets']:
        plt.figure()
        plt.plot(np.std(input_train, axis=0))
        plt.title('std of inputs vs time')

        f_plot_rnn_params(rnn, rate, input_sig, text_tag = 'initial ')

    print('Starting trial training')
    
    start_time = time.time()
    
    for n_bt in range(num_bouts):
         
        rate_start = rnn.init_rate()
        
        for n_it in range(num_it):
            
            optimizer.zero_grad()
            
            output, output_ctx, rate = rnn.forward_ctx(input_sig[:,:, n_bt], rate_start)
            
            target2 = torch.argmax(target[:,:, n_bt], dim =0) * torch.ones(T)
            
            target2_ctx = torch.argmax(target_ctx[:,:, n_bt], dim =0) * torch.ones(T)
            

            # for crossentropy
            loss2 = loss(output.T, target2.long())
            output_sm = output
            
            loss2_ctx = loss_ctx(output_ctx.T, target2_ctx.long())
            
            total_loss = loss2 + loss2_ctx
            
            # for nnnlosss
            #output_sm = rnn.softmax(output)        
            #loss2 = loss(output_sm.T, target2.long())
            
            total_loss.backward() # retain_graph=True
            optimizer.step()
            
            rates_all[:,:, n_it, n_bt] = rate.detach().numpy()
            outputs_all[:,:, n_it, n_bt] = output_sm.detach().numpy()
            
            loss_all[n_it, n_bt] = loss2.item()
            
            for n_t in range(T):
                loss_all_T[n_t, n_it, n_bt] = loss(output_sm[:,n_t], target2[n_t].long()).item()
            
            if reinit_rate:
                rate_start = rnn.init_rate()
            else:
                rate_start = rate[:,-1].detach()

            # Compute the running loss every 10 steps
            if ((n_it) % 10) == 0:
                print('bout %d, Step %d, Loss freq %0.3f, loss ctx %0.3f, Time %0.1fs' % (n_bt, n_it, loss2.item(), loss2_ctx.item(), time.time() - start_time))


    print('Done')
    
    if params['plot_deets']:
        f_plot_rnn_params(rnn, rate, input_sig, text_tag = 'final ')
        
        plt.figure()
        plt.plot(loss_all)
        plt.title('bouts loss')
    
    rnn_out = {'rates':         rates_all,
               'output':       outputs_all,
               'loss':          loss_all,
               'lossT':         loss_all_T,
               }
    return rnn_out   
#%%
def f_RNN_test(rnn, loss, input_test, output_test, params, paradigm='freq'):
    hidden_size = params['hidden_size'];  
    
    T, batch_size, input_size = input_test.shape
    output_size = output_test.shape[2]
    
    input1 = torch.tensor(input_test).float().to(params['device'])
    target = torch.tensor(output_test).float().to(params['device'])
    
    rate_start = rnn.init_rate(batch_size).to(params['device'])
    
    if paradigm == 'freq':
        output, rates = rnn.forward_freq(input1, rate_start)
    elif paradigm == 'ctx':
        output, rates = rnn.forward_ctx(input1, rate_start)

    output3 = output.permute((1, 2, 0))
    target2 = (torch.argmax(target, dim =2) * torch.ones(T, batch_size).to(params['device'])).long()
    target3 = target2.permute((1, 0))
    
    loss2 = loss(output3, target3)
    
    lossT = np.zeros((T, batch_size))
    loss3 = np.zeros((batch_size))
    
    
    for n_bt2 in range(batch_size):
        loss3[n_bt2] = loss(output[:, n_bt2, :], target2[:, n_bt2]).item()
        
        for n_t in range(T):
            lossT[n_t, n_bt2] = loss(output[n_t, n_bt2, :], target2[n_t, n_bt2]).item()

    
    rnn_out = {'rates':         rates.detach().cpu().numpy(),
               'input':         input1.detach().cpu().numpy(),
               'target':        target.detach().cpu().numpy(),
               'target_idx':    target2.detach().cpu().numpy(),
               
               'output':        output.detach().cpu().numpy(),
               'loss':          loss3,
               'lossT':         lossT
               }
    
    print('done')
    
    return rnn_out

#%%

def f_RNN_test_ctx(rnn, loss, input_test, output_test, params):
    hidden_size = params['hidden_size'];  
    
    T, batch_size, input_size = input_test.shape
    output_size = output_test.shape[2]

    input1 = torch.tensor(input_test).float().to(params['device'])
    target = torch.tensor(output_test).float().to(params['device'])

    rate_start = rnn.init_rate(batch_size).to(params['device'])
    
    output, rates = rnn.forward_ctx(input1, rate_start)
    
    output3 = output.permute((1, 2, 0))
    target2 = (torch.argmax(target, dim =2) * torch.ones(T, batch_size).to(params['device'])).long()
    target3 = target2.permute((1, 0))
    loss2 = loss(output3, target3)
    
    lossT = np.zeros((T, batch_size))
    loss3 = np.zeros((batch_size))
    
    for n_bt2 in range(batch_size):
        loss3[n_bt2] = loss(output[:, n_bt2, :], target2[:, n_bt2]).item()
        for n_t in range(T):
            lossT[n_t, n_bt2] = loss(output[n_t, n_bt2, :], target2[n_t, n_bt2]).item()

    rnn_out = {'rates':         rates.detach().cpu().numpy(),
               'input':         input1.detach().cpu().numpy(),
               'target':        target.detach().cpu().numpy(),
               'target_idx':    target2.detach().cpu().numpy(),
               
               'output':        output.detach().cpu().numpy(),
               'loss':          loss3,
               'lossT':         lossT,
               }


    print('done')
    
    return rnn_out

#%%
def f_RNN_test_spont(rnn, input_spont, params):
    
    if not len(input_spont):
        print('not coded yet')
    else:
        T, batch_size, input_size = input_spont.shape
        input1 = torch.tensor(input_spont).float().to(params['device'])
    
    rate_start = rnn.init_rate(batch_size).to(params['device'])
    
    _, rates = rnn.forward_freq(input1, rate_start)
    
    
    rnn_out = {'rates':         rates.detach().cpu().numpy(),
               'input':         input1.detach().cpu().numpy(),
               
               }
    
    
    print('done')
    
    return rnn_out


#%%

def f_gen_dset(dparams, params, stim_templates, stim_sample='equal'):
    
    if stim_sample=='equal':
        dev_stim = np.round(np.linspace(0,params['num_freq_stim']+1, dparams['num_dev_stim']+2))[1:-1].astype(int)
        
        red_stim = np.round(np.linspace(0,params['num_freq_stim']+1, dparams['num_red_stim']+2))[1:-1].astype(int)
        
    elif stim_sample=='random':
        dev_stim = np.random.choice(np.arange(params['num_freq_stim'])+1, size=dparams['num_dev_stim'], replace=False)
        
        red_stim = np.random.choice(np.arange(params['num_freq_stim'])+1, size=dparams['num_red_stim'], replace=False)

    # test oddball trials
    trials_test_oddball_freq, trials_test_oddball_ctx, red_dd_seq = f_gen_oddball_seq(dev_stim, red_stim, dparams['num_trials'], params['dd_frac'], params['num_ctx'], dparams['num_batch'], can_be_same = False)
    #trials_test_oddball_freq, trials_test_oddball_ctx = f_gen_oddball_seq([5], params['test_oddball_stim'], params['test_trials_in_sample'], params['dd_frac'], params['num_ctx'], params['test_batch_size'], can_be_same = True)

    input_test_oddball, output_test_oddball_freq = f_gen_input_output_from_seq(trials_test_oddball_freq, stim_templates['freq_input'], stim_templates['freq_output'], params)
    _, output_test_oddball_ctx = f_gen_input_output_from_seq(trials_test_oddball_ctx, stim_templates['freq_input'], stim_templates['ctx_output'], params)


    data_out = {'dev_stim':                     dev_stim,
                'red_stim':                     red_stim,
                'trials_test_oddball_freq':     trials_test_oddball_freq,
                'trials_test_oddball_ctx':      trials_test_oddball_ctx,
                'red_dd_seq':                   red_dd_seq,
                'input_test_oddball':           input_test_oddball,
                'output_test_oddball_freq':     output_test_oddball_freq,
                'output_test_oddball_ctx':      output_test_oddball_ctx,
                }
    
    return data_out

def f_trial_ave_ctx_pad(rates4d_cut, trials_test_oddball_ctx_cut, pre_dd = 2, post_dd = 2):
    num_t, num_tr, num_run, num_cells = rates4d_cut.shape
    
    num_tr_ave = pre_dd + post_dd + 1
    
    temp_ave4d = np.zeros((num_t, num_tr_ave, num_run, num_cells))
    
    for n_run in range(num_run):
            
        temp_ob = trials_test_oddball_ctx_cut[:,n_run]
        
        temp_sum1 = np.zeros((num_t, num_tr_ave, num_cells))
        num_dd = 0
        
        for n_tr in range(pre_dd, num_tr-post_dd):
            
            if temp_ob[n_tr]:
                
                if np.sum(temp_ob[n_tr-pre_dd:n_tr+post_dd+1]) == 1:
                        
                    num_dd += 1
                    temp_sum1 += rates4d_cut[:, n_tr-pre_dd:n_tr+post_dd+1, n_run, :]
                
        temp_ave4d[:,:,n_run,:] = temp_sum1/num_dd
    
    return temp_ave4d
    
def f_trial_ave_ctx_rd(rates4d_cut, trials_test_oddball_ctx_cut, params):
    num_t, num_tr, num_run, num_cells = rates4d_cut.shape
    
    if params['num_ctx'] == 1:
        ctx_pad1 = 0
    elif params['num_ctx'] == 2:
        ctx_pad1 = 1
        
    trial_ave_rd = np.zeros((2, num_t, num_run, num_cells))
    
    for n_run in range(num_run):
        idx1 = trials_test_oddball_ctx_cut[:,n_run] == 0+ctx_pad1
        trial_ave_rd[0,:,n_run,:] = np.mean(rates4d_cut[:,idx1,n_run,:], axis=1)
        
        idx1 = trials_test_oddball_ctx_cut[:,n_run] == 1+ctx_pad1
        trial_ave_rd[1,:,n_run,:] = np.mean(rates4d_cut[:,idx1,n_run,:], axis=1)
    
    return trial_ave_rd

def f_run_dred(rates2d_cut, subtr_mean=0, method=1):
    # subtract mean 
    if subtr_mean:
        rates_mean = np.mean(rates2d_cut, axis=0)
        rates_in = rates2d_cut - rates_mean;
    else:
        rates_mean = np.zeros((rates2d_cut.shape[1]))
        rates_in = rates2d_cut

    if method==1:
        pca = PCA();
        pca.fit(rates_in)
        proj_data = pca.fit_transform(rates_in)
        components = pca.components_.T
        #V2 = pca.components_
        #US = pca.fit_transform(rates_in)
        exp_var = pca.explained_variance_ratio_
        mean_all = rates_mean + pca.mean_
    elif method==2:
        U, S, V = linalg.svd(rates_in, full_matrices=False)
        #data_back = np.dot(U * S, V)
        #US = U*S
        proj_data = U*S
        components = V.T
        Ssq = S*S
        exp_var = Ssq / np.sum(Ssq)
        mean_all = rates_mean
    
    return proj_data, exp_var, components, mean_all

def f_plot_dred_rates(trials_test_oddball_ctx_cut, comp_out3d, comp_out4d, ob_data1, pl_params, params, title_tag=''):
    num_runs_plot = pl_params['num_runs_plot']
    plot_trials = pl_params['plot_trials'] #800
    color_ctx = pl_params['color_ctx']  # 0 = red; 1 = dd
    mark_red = pl_params['mark_red']
    mark_dd = pl_params['mark_dd']
    
    colors1 = cm.jet(np.linspace(0,1,params['num_freq_stim']))
    
    trial_len, num_tr, num_batch, num_cells = comp_out4d.shape
    
    plot_patches = range(num_runs_plot)#[0, 1, 5]
    
    plot_T = plot_trials*trial_len
    
    plot_pc = pl_params['plot_pc']
    for n_pcpl in range(len(plot_pc)):
        plot_pc2 = plot_pc[n_pcpl]
        plt.figure()
        #plt.subplot(1,2,2);
        for n_bt in plot_patches: #num_bouts
            temp_ob_tr = trials_test_oddball_ctx_cut[:,n_bt]
            
            red_idx = temp_ob_tr == round(params['num_ctx']-1)
            dd_idx = temp_ob_tr == params['num_ctx']
            
            temp_comp4d = comp_out4d[:,:plot_trials,n_bt,:]
            
            plt.plot(comp_out3d[:plot_T, n_bt, plot_pc2[0]-1], comp_out3d[:plot_T, n_bt, plot_pc2[1]-1], color=colors1[ob_data1['red_dd_seq'][color_ctx,n_bt]-1,:])
            
            if mark_red:
                plt.plot(temp_comp4d[4:15,:,plot_pc2[0]-1][:,red_idx[:plot_trials]], temp_comp4d[4:15,:,plot_pc2[1]-1][:,red_idx[:plot_trials]], '.b')
                plt.plot(temp_comp4d[4,:,plot_pc2[0]-1][red_idx[:plot_trials]], temp_comp4d[4,:,plot_pc2[1]-1][red_idx[:plot_trials]], 'ob')
        
            if mark_dd: 
                plt.plot(temp_comp4d[4:15,:,plot_pc2[0]-1][:,dd_idx[:plot_trials]], temp_comp4d[4:15,:,plot_pc2[1]-1][:,dd_idx[:plot_trials]], '.r')
                plt.plot(temp_comp4d[4,:,plot_pc2[0]-1][dd_idx[:plot_trials]], temp_comp4d[4,:,plot_pc2[1]-1][dd_idx[:plot_trials]], 'or')
  
        plt.title('PCA components; %s' % title_tag); plt.xlabel('PC%d' % plot_pc2[0]); plt.ylabel('PC%d' % plot_pc2[1])
        
    
    if 0:
        n_bt  = 0
        
        plt.figure()
        plt.plot(comp_out3d[:T, n_bt, 0], comp_out3d[:T, n_bt, 1])
        plt.plot(comp_out3d[0, n_bt, 0], comp_out3d[0, n_bt, 1], '*')
        plt.title('PCA components; bout %d' % n_bt); plt.xlabel('PC1'); plt.ylabel('PC2')
        
        plot_T = 800
        idx3 = np.linspace(0, plot_T-trial_len, round(plot_T/trial_len)).astype(int)
        
        plt.figure()
        plt.plot(comp_out3d[:plot_T, n_bt, 0], comp_out3d[:plot_T, n_bt, 1])
        plt.plot(comp_out3d[idx3, n_bt, 0], comp_out3d[idx3, n_bt, 1], 'o')
        plt.plot(comp_out3d[0, n_bt, 0], comp_out3d[0, n_bt, 1], '*')
        plt.title('PCA components; bout %d' % n_bt); plt.xlabel('PC1'); plt.ylabel('PC2')

def f_plot_traj_speed(rates, ob_data, n_run, start_idx = 0, title_tag = ''):
    dist1 = squareform(pdist(rates[start_idx:,n_run,:], metric='euclidean'))
    dist2 = np.diag(dist1, 1)
    
    spec2 = gridspec.GridSpec(ncols=1, nrows=3, height_ratios=[3, 1, 2])
    
    plt.figure()
    ax1 = plt.subplot(spec2[0])
    plt.imshow(ob_data['input_test_oddball'][start_idx:-1,n_run,:].T, aspect="auto", interpolation='none')
    plt.title(title_tag)
    plt.subplot(spec2[1], sharex=ax1)
    plt.imshow(ob_data['output_test_oddball_ctx'][start_idx:-1,n_run,:].T, aspect="auto", interpolation='none')
    plt.subplot(spec2[2], sharex=ax1)
    plt.plot(dist2)
    plt.ylabel('euclidean dist')
    plt.xlabel('trials')    

def f_plot_resp_distances(rates4d_cut, trials_test_oddball_ctx_cut, ob_data1, params, choose_idx = 'center', variab_tr_idx = 0, plot_tr_idx = 0, title_tag=''):

    if variab_tr_idx:
        var_seq = ob_data1['dev_stim']
    else:
        var_seq = ob_data1['red_stim']
      
    trial_len, num_tr, num_batch, num_cells = rates4d_cut.shape
    
    #
    trial_ave_rd = f_trial_ave_ctx_rd(rates4d_cut, trials_test_oddball_ctx_cut, params)
    
    if choose_idx == 'center':
        cur_tr = var_seq[round(len(var_seq)/2)]
    elif choose_idx == 'sample':
        cur_tr = np.random.choice(var_seq, size=1)[0]
        
    idx_cur = ob_data1['red_dd_seq'][variab_tr_idx,:] == cur_tr
    base_resp = np.mean(trial_ave_rd[plot_tr_idx,:,idx_cur,:], axis=0)
    base_resp1d = np.reshape(base_resp, (trial_len*num_cells), order='F')
    
    num_var = len(var_seq)
    
    dist_all = np.zeros((num_var))
    dist_all_cos = np.zeros((num_var))
    has_data = np.zeros((num_var), dtype=bool)
    
    for n_tr in range(num_var):
        idx1 = ob_data1['red_dd_seq'][variab_tr_idx,:] == var_seq[n_tr]
        if np.sum(idx1):
            temp1 = np.mean(trial_ave_rd[plot_tr_idx,:,idx1,:], axis=0)
            temp1_1d = np.reshape(temp1, (trial_len*num_cells), order='F')
            
            dist_all[n_tr] = pdist(np.vstack((base_resp1d,temp1_1d)), metric='euclidean')[0]
            dist_all_cos[n_tr] = pdist(np.vstack((base_resp1d,temp1_1d)), metric='cosine')[0]
            has_data[n_tr] = 1
        
    
    plt.figure()
    plt.plot(var_seq[has_data], dist_all[has_data])
    plt.ylabel('euclidean dist')
    if variab_tr_idx:
        plt.title('const red, var dd; ref tr = %d; %s' % (cur_tr, title_tag))
        plt.xlabel('dd stim')
    else:
        plt.title('const dd, var red; ref tr = %d; %s' % (cur_tr, title_tag))
        plt.xlabel('red stim')
    
    plt.figure()
    plt.plot(var_seq[has_data], dist_all_cos[has_data])
    plt.ylabel('cosine dist')
    plt.xlabel('red stim')
    if variab_tr_idx:
        plt.title('const red, var dd; ref tr = %d; %s' % (cur_tr, title_tag))
        plt.xlabel('dd stim')
    else:
        plt.title('const dd, var red; ref tr = %d; %s' % (cur_tr, title_tag))
        plt.xlabel('red stim')


def f_plot_mmn(rates4d_cut, trials_test_oddball_ctx_cut, params, title_tag):
    
    trial_len, num_tr, num_batch, num_cells = rates4d_cut.shape
    plot_time = np.arange(trial_len)-params['isi_duration']/params['dt']


    if params['num_ctx'] == 1:
        ctx_pad1 = 0
    elif params['num_ctx'] == 2:
        ctx_pad1 = 1
    
    trial_ave_ctx = np.zeros((trial_len, 2, num_batch, num_cells))
    for n_run in range(num_batch):
        for n_ctx in range(2):
            idx1 = trials_test_oddball_ctx_cut[:,n_run] == n_ctx+ctx_pad1
            trial_ave_ctx[:, n_ctx, n_run,:] = np.mean(rates4d_cut[:,idx1,n_run,:], axis=1)
    
    trial_ave_ctxn = trial_ave_ctx - np.mean(trial_ave_ctx[:5,:,:,:], axis=0)
    
    n_run = 2
    plt.figure(); 
    for n_run in range(num_batch):
        plt.plot(plot_time, np.mean(trial_ave_ctxn[:,0,n_run,:], axis=1), 'b')
        plt.plot(plot_time, np.mean(trial_ave_ctxn[:,1,n_run,:], axis=1), 'r')
    plt.title(title_tag)

def f_plot_dred_pcs(data3d, comp_list, color_idx, color_ctx, colors, title_tag=''):
    
    num_t, num_run, num_pcs = data3d.shape
    
    for n_pl in range(len(comp_list)):
        plt.figure()
        for n_run in range(num_run):
            plt.plot(data3d[:,n_run, comp_list[n_pl][0]], data3d[:,n_run, comp_list[n_pl][1]], color=colors[color_idx[color_ctx,n_run]-1,:])
            plt.plot(data3d[0,n_run, comp_list[n_pl][0]], data3d[0,n_run, comp_list[n_pl][1]], 'o', color=colors[color_idx[color_ctx,n_run]-1,:])
            plt.xlabel('comp %d' % comp_list[n_pl][0])
            plt.ylabel('comp %d' % comp_list[n_pl][1])
        plt.title('%s; pl %d' % (title_tag, n_pl))

#%%

def f_plot_rnn_weights(rnn, rnn0, rnn0c=[]):
    
    alpha1 = 0.3
    density1 = False
    
    wr1 = rnn.h2h.weight.detach().cpu().numpy().flatten()
    wi1 = rnn.i2h.weight.detach().cpu().numpy().flatten()
    wo1 = rnn.h2o_ctx.weight.detach().cpu().numpy().flatten()
    
    br1 = rnn.h2h.bias.detach().cpu().numpy().flatten()
    bi1 = rnn.i2h.bias.detach().cpu().numpy().flatten()
    bo1 = rnn.h2o_ctx.bias.detach().cpu().numpy().flatten()
    
    
    wr0 = rnn0.h2h.weight.detach().cpu().numpy().flatten()
    wi0 = rnn0.i2h.weight.detach().cpu().numpy().flatten()
    wo0 = rnn0.h2o_ctx.weight.detach().cpu().numpy().flatten()
    
    br0 = rnn0.h2h.bias.detach().cpu().numpy().flatten()
    bi0 = rnn0.i2h.bias.detach().cpu().numpy().flatten()
    bo0 = rnn0.h2o_ctx.bias.detach().cpu().numpy().flatten()
    
    if rnn0c != []:
        wr0c = rnn0c.h2h.weight.detach().cpu().numpy().flatten()
        wi0c = rnn0c.i2h.weight.detach().cpu().numpy().flatten()
        wo0c = rnn0c.h2o_ctx.weight.detach().cpu().numpy().flatten()
        
        br0c = rnn0c.h2h.bias.detach().cpu().numpy().flatten()
        bi0c = rnn0c.i2h.bias.detach().cpu().numpy().flatten()
        bo0c = rnn0c.h2o_ctx.bias.detach().cpu().numpy().flatten()
    
        # compensate
        rnn0c.h2h.weight.data = rnn0c.h2h.weight.data/np.std(wr0c)*np.std(wr1)
        rnn0c.i2h.weight.data = rnn0c.i2h.weight.data/np.std(wi0c)*np.std(wi1)
        rnn0c.h2o_ctx.weight.data = rnn0c.h2o_ctx.weight.data/np.std(wo0c)*np.std(wo1)
        
        rnn0c.h2h.bias.data = rnn0c.h2h.bias.data/np.std(br0c)*np.std(br1)
        rnn0c.i2h.bias.data = rnn0c.i2h.bias.data/np.std(bi0c)*np.std(bi1)
        rnn0c.h2o_ctx.bias.data = rnn0c.h2o_ctx.bias.data/np.std(bo0c)*np.std(bo1)
    
        # get again
        wr0c = rnn0c.h2h.weight.detach().cpu().numpy().flatten()
        wi0c = rnn0c.i2h.weight.detach().cpu().numpy().flatten()
        wo0c = rnn0c.h2o_ctx.weight.detach().cpu().numpy().flatten()
        
        br0c = rnn0c.h2h.bias.detach().cpu().numpy().flatten()
        bi0c = rnn0c.i2h.bias.detach().cpu().numpy().flatten()
        bo0c = rnn0c.h2o_ctx.bias.detach().cpu().numpy().flatten()
    
    
    plt.figure()
    plt.subplot(311)
    plt.hist(wr1, density=density1, alpha=alpha1)
    plt.hist(wr0, density=density1, alpha=alpha1)
    if rnn0c != []:
        plt.hist(wr0c, density=density1, alpha=alpha1)
        plt.legend(('trained; std=%.2f' % np.std(wr1), 'untrained; std=%.2f' % np.std(wr0), 'untrained comp; std=%.2f' % np.std(wr0c)))
    else:
        plt.legend(('trained; std=%.2f' % np.std(wr1), 'untrained; std=%.2f' % np.std(wr0)))
    plt.title('Recurrent W')
    
    plt.subplot(312)
    plt.hist(wi1, density=density1, alpha=alpha1)
    plt.hist(wi0, density=density1, alpha=alpha1)
    if rnn0c != []:
        plt.hist(wi0c, density=density1, alpha=alpha1)
        plt.legend(('trained; std=%.2f' % np.std(wi1), 'untrained; std=%.2f' % np.std(wi0), 'untrained comp; std=%.2f' % np.std(wi0c)))
    else:
        plt.legend(('trained; std=%.2f' % np.std(wi1), 'untrained; std=%.2f' % np.std(wi0)))
    plt.title('Input W')
    
    plt.subplot(313)
    plt.hist(wo1, density=density1, alpha=alpha1)
    plt.hist(wo0, density=density1, alpha=alpha1)
    if rnn0c != []:
        plt.hist(wo0c, density=density1, alpha=alpha1)
        plt.legend(('trained; std=%.2f' % np.std(wo1), 'untrained; std=%.2f' % np.std(wo0), 'untrained comp; std=%.2f' % np.std(wo0c)))
    else:
        plt.legend(('trained; std=%.2f' % np.std(wo1), 'untrained; std=%.2f' % np.std(wo0)))
    plt.title('Output W')
    
    
    plt.figure()
    plt.subplot(311)
    plt.hist(br1, density=density1, alpha=alpha1)
    plt.hist(br0, density=density1, alpha=alpha1)
    if rnn0c != []:
        plt.hist(br0c, density=density1, alpha=alpha1)
        plt.legend(('trained; std=%.2f' % np.std(br1), 'untrained; std=%.2f' % np.std(br0), 'untrained comp; std=%.2f' % np.std(br0c)))
    else:
        plt.legend(('trained; std=%.2f' % np.std(br1), 'untrained; std=%.2f' % np.std(br0)))
    plt.title('Recurrent bias')
    
    plt.subplot(312)
    plt.hist(bi1, density=density1, alpha=alpha1)
    plt.hist(bi0, density=density1, alpha=alpha1)
    if rnn0c != []:
        plt.hist(bi0c, density=density1, alpha=alpha1)
        plt.legend(('trained; std=%.2f' % np.std(bi1), 'untrained; std=%.2f' % np.std(bi0), 'untrained comp; std=%.2f' % np.std(bi0c)))
    else:
        plt.legend(('trained; std=%.2f' % np.std(bi1), 'untrained; std=%.2f' % np.std(bi0)))
    plt.title('Input bias')
    
    plt.subplot(313)
    plt.hist(bo1, density=density1, alpha=alpha1)
    plt.hist(bo0, density=density1, alpha=alpha1)
    if rnn0c != []:
        plt.hist(bo0c, density=density1, alpha=alpha1)
        plt.legend(('trained; std=%.2f' % np.std(bo1), 'untrained; std=%.2f' % np.std(bo0), 'untrained comp; std=%.2f' % np.std(bo0c)))
    else:
        plt.legend(('trained; std=%.2f' % np.std(bo1), 'untrained; std=%.2f' % np.std(bo0)))
    plt.title('Output bias')
