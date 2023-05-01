# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 22:47:28 2023

@author: yuriy
"""
import numpy as np
import torch
import torch.nn as nn

#%%

def f_RNN_linear_train(rnn, loss, input_train, output_train, params):
    
    hidden_size = params['hidden_size'];     
    output_size = params['num_stim'] + 1
    
    rate = rnn.init_rate();
    
    
    learning_rate = 0.005
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)   

    # initialize 

    T = input_train.shape[1]

    input_sig = torch.tensor(input_train).float()
    target = torch.tensor(output_train).float()

    rates_all = np.zeros((hidden_size, T));
    outputs_all = np.zeros((output_size, T));
    loss_all = np.zeros((T));
    iterations_all = np.zeros((T));

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

    rates_all[:,0] = rate.detach().numpy()[0,:]
    
    for n_t in range(T-1):
        
        optimizer.zero_grad()
        
        output, rate_new = rnn.forward(input_sig[:,n_t], rate)
        
        rates_all[:,n_t+1] = rate_new.detach().numpy()[0,:];
        
        rate = rate_new.detach();

        outputs_all[:,n_t+1] = output.detach().numpy()[0,:];
        
        target2 = torch.argmax(target[:,n_t]) * torch.ones(1) # torch.tensor()
        loss2 = loss(output, target2.long())
        
        loss2.backward() # retain_graph=True
        optimizer.step()
            
        loss_all[n_t] = loss2.item()
        iterations_all[n_t] = n_t;

    print('Done')
    
    if params['plot_deets']:
        f_plot_rnn_params(rnn, rate, input_sig, text_tag = 'final ')

    
    rnn_out = {'rates':         rates_all,
               'outputs':       outputs_all,
               'loss':          loss_all,
               'iterations':    iterations_all,
               }
    return rnn_out
    
#%%

def f_RNN_trial_train(rnn, loss, input_train, output_train, params):
    
    input_train = input_train_cont
    output_train = output_train_cont
    
    hidden_size = params['hidden_size'];     
    output_size = params['num_stim'] + 1
    
    learning_rate = 0.005
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)   

    # initialize 

    input_size, T, num_bouts = input_train.shape

    input_sig = torch.tensor(input_train).float()
    target = torch.tensor(output_train).float()
    
    
    outputs_all = torch.zeros((output_size, T, num_bouts))
    rates_all = torch.zeros((hidden_size, T, num_bouts))
    
    rates_all = np.zeros((hidden_size, T, num_bouts));
    outputs_all = np.zeros((output_size, T, num_bouts));
    loss_all = np.zeros((T, num_bouts));
    iterations_all = np.zeros((T, num_bouts));

    # can adjust bias here 
    #rnn.h2h.bias.data  = rnn.h2h.bias.data -2
    #np.std(np.asarray(rnn.h2h.weight ).flatten())

    #
    if params['plot_deets']:
        plt.figure()
        plt.plot(np.std(input_train, axis=0))
        plt.title('std of inputs vs time')

        f_plot_rnn_params(rnn, rate, input_sig, text_tag = 'initial ')


    print('Starting linear training')
    
    for n_bt in range(num_bouts):
        rate = rnn.init_rate();
        
        rates_all[:,0, n_bt] = rate
        
        for n_t in range(T-1):
            
            #optimizer.zero_grad()
            
            output, rate_new = rnn.forward(input_sig[:,n_t, n_bt], rate)
            
            rates_all[:,n_t+1, n_bt] = rate_new;
            
            #rates_all[:,n_t+1, n_bt] = rate_new.detach().numpy()[0,:];
            
            rate = rate_new.detach();
            
            outputs_all[:,n_t+1, n_bt] = output
            
            #outputs_all[:,n_t+1, n_bt] = output.detach().numpy()[0,:];
            
            #target2 = torch.argmax(target[:,:, n_bt]) * torch.ones(1) # torch.tensor()
        
        
        
        plt.figure()
        plt.imshow(outputs_all[:,:,n_bt])
        
        
        plt.figure()
        plt.plot(rates_all[:,:, n_bt])
        
        
        
        target2 = torch.argmax(target[:,:, n_bt], dim =0) * torch.ones(T) # torch.tensor()
        
        loss2 = loss(outputs_all[:,:,n_bt], target2.long())
        
            #loss2 = loss(output, target2.long())
            #loss2.backward() # retain_graph=True
            #optimizer.step()
                
            #loss_all[n_t] = loss2.item()
            #iterations_all[n_t] = n_t;

    print('Done')
    
    if params['plot_deets']:
        f_plot_rnn_params(rnn, rate, input_sig, text_tag = 'final ')

    
    rnn_out = {'rates':         rates_all,
               'outputs':       outputs_all,
               'loss':          loss_all,
               'iterations':    iterations_all,
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
    iterations_all_test = np.zeros((T));
    
    for n_t in range(T-1):
        
        output, rate_new = rnn.forward(input_sig_test[:,n_t], rate_test)
        
        rates_all_test[:,n_t+1] = rate_new.detach().numpy()[0,:];
        
        rate_test = rate_new.detach();

        outputs_all_test[:,n_t+1] = output.detach().numpy()[0,:];
        
        target2 = torch.argmax(target_test[:,n_t]) * torch.ones(1) # torch.tensor()
        loss2 = loss(output, target2.long())
          
        loss_all_test[n_t] = loss2.item()
        iterations_all_test[n_t] = n_t
        
    print('done')
    
    rnn_out = {'rates':         rates_all_test,
               'outputs':       outputs_all_test,
               'loss':          loss_all_test,
               'iterations':    iterations_all_test,
               }
    return rnn_out

