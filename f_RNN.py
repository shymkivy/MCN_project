# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 22:47:28 2023

@author: yuriy
"""
import numpy as np
import torch
import torch.nn as nn

#%%
def f_train(Z,N,cur_index):
    1
    
    
    
#%%
def f_test(rnn, loss, input_mat_test, output_mat_test, hidden_size):
    
    T_test = input_mat_test.shape[1]
  
    output_size = output_mat_test.shape[0]
 
    input_sig_test = torch.tensor(input_mat_test[:,0:T_test]).float()

    target_test = torch.tensor(output_mat_test[:,0:T_test]).float()
    
    rate_test = rnn.init_rate();
    
    iteration_test = []
    iteration_test.append(0);

    rates_all_test = np.zeros((hidden_size, T_test))
    outputs_all_test = np.zeros((output_size, T_test));
    loss_all_test = np.zeros((T_test));
    
    for n_t in range(T_test-1):
        
        output, rate_new = rnn.forward(input_sig_test[:,n_t], rate_test)
        
        rates_all_test[:,n_t+1] = rate_new.detach().numpy()[0,:];
        
        rate_test = rate_new.detach();

        outputs_all_test[:,n_t+1] = output.detach().numpy()[0,:];
        
        target2 = torch.argmax(target_test[:,n_t]) * torch.ones(1) # torch.tensor()
        loss2 = loss(output, target2.long())
          
        loss_all_test[n_t] = loss2.item()
        iteration_test.append(iteration_test[-1]+1);
        
    print('done')
        
    return rates_all_test, outputs_all_test, loss_all_test, iteration_test