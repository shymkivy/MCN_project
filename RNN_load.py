# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 02:59:40 2021

@author: ys2605
"""

import torch
import torch.nn as nn
#%%

fpath = 'C://Users//ys2605//Desktop//stuff//RNN_stuff//RNN_data//'


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
        self.softmax = nn.LogSoftmax(dim=1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
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
        
        
    
    def forward(self, input_sig, rate):
        
        rate_new = self.tanh(self.i2h(input_sig) + self.h2h(rate))
    
        rate_new = (1-self.alpha)*rate + self.alpha*rate_new
        
        output = self.softmax(self.h2o(rate_new))
        
        return output, rate_new
    
        
    def init_rate(self):
        rate = torch.empty(1, self.hidden_size);
        nn.init.uniform_(rate, a=0, b=1)
        return rate

#%%
#input_size = 50;

hidden_size = 250;

input_size = 97#input_mat.shape[0];

output_size = 11#output_mat.shape[0]

#T = 10000;

T = 1#input_mat.shape[1]

g = 5;


dt = 1#input_T[0];        #1;
tau = .1;              # in sec

alpha = 1#dt/tau;   


#%%

rnn = RNN_chaotic(input_size, hidden_size, output_size, alpha)

#%%

rnn.load_state_dict(torch.load(fpath+'test'))



