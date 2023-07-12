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
    
    input_size = params['input_size']
    
    loss_strat = 1
    
    T = round((params['stim_duration'] + params['isi_duration'])/params['dt'] * params['train_trials_in_sample'])
    num_samp = params['train_num_samples_ctx']
    batch_size = params['train_batch_size']
    
    
    #optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate) 
    optimizer = torch.optim.AdamW(rnn.parameters(), lr=learning_rate) 

    # initialize 
    #rnn_out['loss'] = np.zeros((num_samp, num_rep))
    rnn_out['loss'] = []
    #rnn_out['rates'] = np.zeros((hidden_size, T, num_rep, num_samp))
    #rnn_out['outputs'] = np.zeros((output_size, T, num_rep, num_samp))

    print('Starting ctx trial training')
    
    start_time = time.time()
    
    for n_samp in range(num_samp):
         
        rate_start = rnn.init_rate(params['train_batch_size']).to(params['device'])
        
        # get sample
        
        trials_train_oddball_freq, trials_train_oddball_ctx = f_gen_oddball_seq(params['oddball_stim'], params['train_trials_in_sample'], params['dd_frac'], params['train_batch_size'], 1)

        input_train_oddball, _ = f_gen_input_output_from_seq(trials_train_oddball_freq, stim_templates['freq_input'], stim_templates['freq_output'], params)
        _, output_train_oddball_ctx = f_gen_input_output_from_seq(trials_train_oddball_ctx, stim_templates['freq_input'], stim_templates['ctx_output'], params)
        
        input_sig = torch.tensor(input_train_oddball).float().to(params['device'])
        target_ctx = torch.tensor(output_train_oddball_ctx).float().to(params['device'])
        
        for n_rep in range(num_rep):
            
            optimizer.zero_grad()
            
            output_ctx, rate = rnn.forward_ctx(input_sig, rate_start)
            
            target_ctx2 = (torch.argmax(target_ctx, dim =2) * torch.ones(T, batch_size).to(params['device'])).long()
            
            if loss_strat == 1:
                output_ctx3 = output_ctx.permute((1, 2, 0))
                target_ctx3 = target_ctx2.permute((1, 0))
                
                loss2 = loss(output_ctx3, target_ctx3)
            
            elif loss_strat == 2:
                # probably equivalent to first
                target_ctx3 = target_ctx2.reshape((T*batch_size))
                output_ctx3 = output_ctx.reshape((T*batch_size, output_size))
                #output_ctx2 = output_ctx.permute((1, 2, 0))
    
                loss2 = loss(output_ctx3, target_ctx3)
            else:
                # computes separately and sums after
                loss4 = []
                for n_bt in range(batch_size):
                    target_ctx3 = target_ctx2[:,n_bt]
                    output_ctx3 = output_ctx[:,n_bt,:]
                    loss3 = loss(output_ctx3, target_ctx3)
                    loss4.append(loss3)
                
                loss2 = sum(loss4)/batch_size
            
            
            
            # for nnnlosss
            #output_sm = rnn.softmax(output)        
            #loss2 = loss(output_sm.T, target2.long())
            
            loss2.backward() # retain_graph=True
            optimizer.step()

            #rnn_out['loss'][n_samp, n_rep] = loss2.item()
            rnn_out['loss'].append(loss2.item())
            
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

