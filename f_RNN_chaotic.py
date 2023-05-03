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
    def __init__(self, input_size, hidden_size, output_size, alpha):
        super(RNN_chaotic, self).__init__()
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.alpha = alpha
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size);
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=0)
        self.tanh = nn.Tanh()
        #self.sigmoid = nn.Sigmoid()
        
    def init_weights(self, g):
        
        
        wh2h = torch.empty(self.hidden_size, self.hidden_size)
        nn.init.normal_(wh2h, mean=0.0, std = 1)
        
        std1 = g/np.sqrt(self.hidden_size);
        
        wh2h = wh2h - np.mean(wh2h.detach().numpy());
        wh2h = wh2h * std1;
        
        self.h2h.weight.data = wh2h;
        
        wi2h = torch.empty(self.hidden_size, self.input_size)
        nn.init.normal_(wi2h, mean=0.0, std = 1)
        
        std1 = g/np.sqrt(self.hidden_size);
        
        wi2h = wi2h - np.mean(wi2h.detach().numpy());
        wi2h = wi2h * std1;
        
        self.i2h.weight.data = wi2h;
        
    def recurrence(self, input_sig, rate):
        # can try relu here
        rate_new = self.tanh(self.i2h(input_sig) + self.h2h(rate))
        rate_new = (1-self.alpha)*rate + self.alpha*rate_new
        
        return rate_new
    
    def forward_linear(self, input_sig, rate):
        
        rate_new = self.recurrence(input_sig, rate)
        output = self.softmax(self.h2o(rate_new))
        
        return output, rate_new
        
    def forward(self, input_sig, rate):
        
        rate_all = []
        #outputs_all = []
        num_steps = input_sig.size(1)
        
        for n_st in range(num_steps):
            rate = self.recurrence(input_sig[:,n_st], rate)
            rate_all.append(rate)
            
            #outputs_all.append(self.softmax(self.h2o(rate)))
            
        rate_all2 = torch.stack(rate_all, dim=1)
        #outputs_all2 = torch.stack(outputs_all, dim=1)
            
        output = self.softmax(self.h2o(rate_all2.T).T)
        
        return output, rate_all2
    
    def init_rate(self):
        rate = torch.empty(self.hidden_size);
        nn.init.uniform_(rate, a=-1, b=1)
        return rate
    

    