# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:11:53 2024

@author: ys2605
"""

import numpy as np

from scipy import linalg
from sklearn.decomposition import PCA

import torch
import torch.nn as nn

#%%
def f_run_dred(rates2d, subtr_mean=0, method=1):
    # subtract mean 
    if subtr_mean:
        rates_mean = np.mean(rates2d, axis=0)
        rates_in = rates2d - rates_mean;
    else:
        rates_mean = np.zeros((rates2d.shape[1]))
        rates_in = rates2d

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

#%%

def f_run_dred_wrap(test_data, subtr_mean=0, method=1):
    test_data['dred_rates2d'], test_data['exp_var'], test_data['dred_comp'], test_data['dred_mean'] = f_run_dred(test_data['rates2d_cut'], subtr_mean=subtr_mean, method=method)
    #comp_out3d = np.reshape(proj_data, (trial_len*num_trials_cut, num_runs, num_cells), order = 'F')
    test_data['dred_rates4d'] = np.reshape(test_data['dred_rates2d'], test_data['rates4d_cut'].shape, order = 'F')


def f_proj_onto_dred(test_data, dred_comp):
    test_data['dred_proj_rates2d'] = np.dot(test_data['rates2d_cut'], dred_comp)
    #comp_outf_const3d = np.reshape(proj_dataf_const, (trial_len*num_trials_cut, red_stim_const.shape[0], num_cellsf), order = 'F')
    test_data['dred_proj_rates4d'] = np.reshape(test_data['dred_proj_rates2d'], test_data['rates4d_cut'].shape, order = 'F')

#%% dred wit hpytorch

class dred_torch(nn.Module):
    def __init__(self, data_in, k=2) -> None:
        super(regress, self).__init__()
        n_row, n_col = data_in.shape
        self.beta = nn.parameter.Parameter(beta0).float()
          
    def fit(self, x):
        out = torch.matmul(x, self.beta)
        
        t_beta0_init = torch.randn((x_tr.shape[1],1)).float()
        
        reg_model = regress(t_beta0_init)
        
        x0t = torch.tensor(x_tr).float()
        y1t = torch.tensor(y_tr).float()
        
        optimizer = torch.optim.SGD(reg_model.parameters(), lr=0.01)
        
        num_it = round(1e3)
        
        tot_loss = np.zeros((num_it))

        for n_it in range(num_it):
    
            optimizer.zero_grad()
            
            out = reg_model.forward(x0t)
            
            #loss = torch.mean(torch.abs((out - y1t))**1)
            loss = torch.mean((out - y1t)**2)
            #loss = torch.mean((out - y1t)**2) + torch.mean(torch.abs((out - y1t))**1)
            
            loss.backward()
            optimizer.step()
            
            tot_loss[n_it] = loss.item()
    
        plt.figure()
        plt.plot(np.log(tot_loss))
        
        
        beta0t = reg_model.beta.detach().numpy()
        
        y_pred0train = np.dot(x_tr, beta0t)
        y_pred0test = np.dot(x_tst, beta0t)
          
        return out