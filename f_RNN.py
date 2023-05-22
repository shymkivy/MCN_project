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
    num_it = params['num_iterations_per_samp']
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
    num_it = params['num_iterations_per_samp']
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

def f_RNN_trial_freq_ctx_train(rnn, loss, loss_ctx, input_train, output_train, output_train_ctx, params):
    
    hidden_size = params['hidden_size'];     
    output_size = params['num_freq_stim'] + 1
    reinit_rate = params['train_reinit_rate']
    num_it = params['num_iterations_per_samp']
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
               'outputs':       outputs_all,
               'loss':          loss_all,
               'lossT':         loss_all_T,
               }
    return rnn_out   
#%%
def f_RNN_test(rnn, loss, input_test, output_test, params):
    hidden_size = params['hidden_size'];  
    
    T = input_test.shape[1]
    output_size = output_test.shape[0]
    
    input_sig_test = torch.tensor(input_test).float()
    target_test = torch.tensor(output_test).float()
    
    rate_test = rnn.init_rate();
    
    rates_all_test = np.zeros((hidden_size, T))
    outputs_all_test = np.zeros((output_size, T));
    loss_all_test = np.zeros((T));

    for n_t in range(T):
        
        output, rate_new = rnn.forward_linear(input_sig_test[:,n_t], rate_test)
        
        target2 = torch.argmax(target_test[:,n_t])# * torch.ones(1) # torch.tensor()
        
        # crossentropy
        loss2 = loss(output, target2.long())
        output_sm = output
        
        # nnnloss
        #output_sm = rnn.softmax(output)
        #loss2 = loss(output_sm, target2.long())
        
        
        rates_all_test[:,n_t] = rate_new.detach().numpy();
        
        rate_test = rate_new.detach();

        outputs_all_test[:,n_t] = output_sm.detach().numpy();
        
        
        
          
        loss_all_test[n_t] = loss2.item()
        
    print('done')
    
    rnn_out = {'rates':         rates_all_test,
               'outputs':       outputs_all_test,
               'loss':          loss_all_test,
               }
    return rnn_out

#%%

def f_RNN_test_ctx(rnn, loss, loss_ctx, input_test, output_test, output_test_ctx, params):
    hidden_size = params['hidden_size'];  
    
    T = input_test.shape[1]
    output_size = output_test.shape[0]
    output_size_ctx = output_test_ctx.shape[0]
    
    input_sig_test = torch.tensor(input_test).float()
    target_test = torch.tensor(output_test).float()
    target_test_ctx = torch.tensor(output_test_ctx).float()
    
    rate_test = rnn.init_rate();
    
    rates_all_test = np.zeros((hidden_size, T))
    outputs_all_test = np.zeros((output_size, T));
    outputs_all_test_ctx = np.zeros((output_size_ctx, T));
    loss_all_test = np.zeros((T));
    loss_all_test_ctx = np.zeros((T));
    
    for n_t in range(T):
        
        output, output_ctx, rate_new = rnn.forward_linear_ctx(input_sig_test[:,n_t], rate_test)
        
        target2 = torch.argmax(target_test[:,n_t])# * torch.ones(1) # torch.tensor()
        
        # crossentropy
        loss2 = loss(output, target2.long())
        output_sm = output
        
        
        target2_ctx = torch.argmax(target_test_ctx[:,n_t])
        
        loss2_ctx = loss_ctx(output_ctx, target2_ctx.long())
        output_ctx_sm = output_ctx
        
        # nnnloss
        #output_sm = rnn.softmax(output)
        #loss2 = loss(output_sm, target2.long())
        
        
        rates_all_test[:,n_t] = rate_new.detach().numpy();
        
        rate_test = rate_new.detach();

        outputs_all_test[:,n_t] = output_sm.detach().numpy();
        
        outputs_all_test_ctx[:,n_t] = output_ctx_sm.detach().numpy();
        
        loss_all_test[n_t] = loss2.item()
        loss_all_test_ctx[n_t] = loss2_ctx.item()
        
    print('done')
    
    rnn_out = {'rates':         rates_all_test,
               'outputs':       outputs_all_test,
               'loss':          loss_all_test,
               'outputs_ctx':   outputs_all_test_ctx,
               'loss_ctx':      loss_all_test_ctx,
               }
    
    return rnn_out

