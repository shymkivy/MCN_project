# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 14:59:38 2021

@author: ys2605
"""
#%%

import numpy as np
import torch
import torch.nn as nn

#%%

class RNN_chaotic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size_freq, output_size_ctx, alpha, add_noise, activation='tanh'): # 
        super(RNN_chaotic, self).__init__()
        
        self.add_noise = add_noise
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size_freq
        self.output_size_ctx = output_size_ctx
        self.alpha = alpha
        self.i2h = nn.Linear(input_size, hidden_size)   #, device=self.device
        self.h2h = nn.Linear(hidden_size, hidden_size)  #, device=self.device
        self.h2o = nn.Linear(hidden_size, output_size_freq) #, device=self.device
        self.h2o_ctx = nn.Linear(hidden_size, output_size_ctx)
        self.softmax = nn.LogSoftmax(dim=0)
        if activation == 'tanh':
            self.activ = nn.Tanh()
        elif activation == 'ReLU':
            self.activ = nn.ReLU()
            
        #self.sigmoid = nn.Sigmoid()
        
    def init_weights(self, g):
        
        std1 = g/np.sqrt(self.hidden_size);
        self.sigma_rec = std1
        
        # recurrent
        wh2h = self.h2h.weight.data
        nn.init.normal_(wh2h, mean=0.0, std = 1)
        
        mean1 = wh2h.mean()
        wh2h = wh2h - mean1;
        wh2h = wh2h * std1;
        
        self.h2h.weight.data = wh2h;
        
        # input to hidden
        wi2h = self.i2h.weight.data
        nn.init.normal_(wi2h, mean=0.0, std = 1)
        
        mean1 = wi2h.mean()
        wi2h = wi2h - mean1;
        wi2h = wi2h * std1;
    
        self.i2h.weight.data = wi2h
        
        # hidden to output
        wh2o = self.h2o.weight.data
        nn.init.normal_(wh2o, mean=0.0, std = 1)
        wh2o_ctx = self.h2o_ctx.weight.data
        nn.init.normal_(wh2o_ctx, mean=0.0, std = 1)
        
        mean1 = wh2o.mean()
        wh2o = wh2o - mean1;
        wh2o = wh2o * std1;
    
        self.h2o.weight.data = wh2o
    
    def init_rate(self, batch_size=1):
        rate = torch.empty((batch_size, self.hidden_size)) # , device=self.device
        nn.init.uniform_(rate, a=-1, b=1)
        return rate
        
    def recurrence(self, input_sig, rate):
        # can try relu here
        
        if self.add_noise:
            noise1 = torch.randn(rate.shape).to(input_sig.device.type)*np.sqrt(2*self.sigma_rec**2/self.alpha)
        else:
            noise1 = 0
        
        rate_new = self.activ(self.i2h(input_sig) + self.h2h(rate)+noise1)
        rate_new = (1-self.alpha)*rate + self.alpha*rate_new
        
        return rate_new
    
    def forward_linear(self, input_sig, rate):
        
        rate_new = self.recurrence(input_sig, rate)
        output = self.h2o(rate_new)
        
        return output, rate_new
    
    def forward_linear_ctx(self, input_sig, rate):
        
        rate_new = self.recurrence(input_sig, rate)
        output = self.h2o(rate_new)
        output_ctx = self.h2o_ctx(rate_new)
        
        return output, output_ctx, rate_new
        
    def forward_freq(self, input_sig, rate):
        
        rate_all = []
        num_steps = input_sig.size(0)
 
        for n_st in range(num_steps):
            rate = self.recurrence(input_sig[n_st,:,:], rate)
            rate_all.append(rate)
            
        rate_all2 = torch.stack(rate_all, dim=0)
        #outputs_all2 = torch.stack(outputs_all, dim=1)
            
        output = self.h2o(rate_all2)
        
        return output, rate_all2
    
    def forward_ctx(self, input_sig, rate):
        
        rate_all = []
        num_steps = input_sig.size(0)
        
        for n_st in range(num_steps):
            rate = self.recurrence(input_sig[n_st,:,:], rate)
            rate_all.append(rate)
            
        rate_all2 = torch.stack(rate_all, dim=0)
        #outputs_all2 = torch.stack(outputs_all, dim=1)
            
        output_ctx = self.h2o_ctx(rate_all2)
        
        return output_ctx, rate_all2
    
    def forward_dual(self, input_sig, rate):
        
        rate_all = []
        num_steps = input_sig.size(0)
       
        for n_st in range(num_steps):
            rate = self.recurrence(input_sig[n_st,:,:], rate)
            rate_all.append(rate)
            

        rate_all2 = torch.stack(rate_all, dim=0)
        
        output = self.h2o(rate_all2)
        output_ctx = self.h2o_ctx(rate_all2)
        
        return output, output_ctx, rate_all2
    
    def softmax1(self, output):
        
        output_sm = self.softmax(output)
        
        return output_sm
    
    
    

    