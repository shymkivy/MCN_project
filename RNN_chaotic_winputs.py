# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 15:33:58 2021

@author: Administrator
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
import torch.nn as nn
from random import sample, random
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy import signal
from scipy.io import loadmat, savemat
import skimage.io
import math

#%%
plt.close('all')



#%%

# stim_duration = 500;
# isi = 500;

# freqs_low = 2000;
# freqs_high = 90000;
# num_freqs = 10;
# #octaves = 1.5;
# octaves = (90/2)**(1/((num_freqs-1)))

# freqs = np.zeros((num_freqs));
# freqs[0] = freqs_low;
# for n_freq in range(num_freqs-1):
#     freqs[n_freq+1] = freqs[n_freq]*octaves

# Fs = 200000;

# num_stim = 9

# stim_types = sample(range(num_freqs), num_stim)

# stim_times = np.asarray(range(num_stim))*(stim_duration+isi)+isi+np.random.uniform(low=0, high=100, size=num_stim)

# sound_len = int(T*Fs/1000)

# sound = np.zeros((sound_len))
# sound_t = np.asarray(range(sound_len))/Fs*1000

# tone_len = int(stim_duration*Fs/1000)
# tone_t = np.asarray(range(tone_len))/Fs*1000

# for n_st in range(num_stim):
#     stim_time1 = stim_times[n_st]
#     stim_type1 = stim_types[n_st]
    
#     stim_idx = np.argmin((sound_t - stim_time1)**2)
    
#     temp_tone = np.sin(tone_t*2*math.pi*freqs[stim_type1]/1000);
    
#     sound[stim_idx:(stim_idx+tone_len)] = temp_tone

# sound_n = sound + np.random.randn(sound_len)*0.3

# plt.figure()
# plt.plot(sound_t,sound_n)

# windowsize1 = round(Fs * 0.0032);
# noverlap1 = round(Fs * 0.0032 * .50);
# nfft1 = round(Fs * 0.0032);

# spec1 = signal.spectrogram(sound_n, fs=Fs, nperseg=windowsize1, noverlap=noverlap1, nfft=nfft1)

# plt.figure()
# plt.imshow(spec1[2], extent = [spec1[1][0], spec1[1][-1], spec1[0][0], spec1[0][-1]], aspect=.0001,origin='lower')


#%%
fpath = 'C://Users//Administrator//Desktop//yuriy//RNN_project//data//'

#fname = 'sim_spec_1_stim_8_24_21.mat'

#fname = 'sim_spec_10complex_200rep_stim_8_25_21_1.mat'

fname = 'sim_spec_10tones_200rep_stim_8_25_21_1.mat'

data_mat = loadmat(fpath+fname)

data1 = data_mat['spec_data']
print(data1.dtype)

input_mat = data1[0, 0]['spec_cut'];
fr_cut = data1[0, 0]['fr_cut'];
ti = data1[0, 0]['ti'];
voc_seq = data1[0, 0]['voc_seq'];
num_voc = data1[0, 0]['num_voc'];
output_mat = data1[0, 0]['output_mat'];
output_mat_delayed = data1[0, 0]['output_mat_delayed'];

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
# plt.figure()
# plt.imshow(input_mat[:,0:5000],  aspect=10)

# plt.figure()
# plt.imshow(output_mat[:,0:5000],  aspect=100)


#%%

# plt.figure();
# plt.imshow(input_sig[:,0:300]);

#%%
#input_size = 50;

hidden_size = 250;

input_size = input_mat.shape[0];

output_size = output_mat.shape[0]

#T = 10000;

T = input_mat.shape[1]

g = 5;

dt = 1;
tau = 10;
alpha = dt/tau;


#%%

#input_sig = torch.zeros(input_size, T)

#input_mat = np.random.randn(input_size,T)

input_mat = input_mat - np.mean(input_mat)
input_mat = input_mat/np.std(input_mat)

input_sig = torch.tensor(input_mat[:,0:T]).float()

target = torch.tensor(output_mat[:,0:T]).float()

#%%
# W is inside i2h.weight

# i2h = nn.Linear(input_size, hidden_size)
# h2h = nn.Linear(hidden_size, hidden_size)

# h2o = nn.Linear(hidden_size, output_size)

# #softmax = nn.LogSoftmax(dim=1)

# tanh1 = nn.Tanh();

# softmax1 = nn.LogSoftmax(dim=1);

#%% fix weights


# wh2h = torch.empty(hidden_size, hidden_size)
# nn.init.normal_(wh2h, mean=0.0, std = 1)

# std1 = g/np.sqrt(hidden_size);

# wh2h = wh2h - np.mean(wh2h.detach().numpy());
# wh2h = wh2h * std1;

# h2h.weight.data = wh2h;

# wi2h = torch.empty(hidden_size, input_size)
# nn.init.normal_(wi2h, mean=0.0, std = 1)

# std1 = g/np.sqrt(hidden_size);

# wi2h = wi2h - np.mean(wi2h.detach().numpy());
# wi2h = wi2h * std1;

# i2h.weight.data = wi2h;

#%%

rnn = RNN_chaotic(input_size, hidden_size, output_size, alpha)

#%%
loss = nn.NLLLoss()
learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

#%%

rate = rnn.init_rate();

rnn.init_weights(g)

#rnn.h2h.bias.data  = rnn.h2h.bias.data -2

#np.std(np.asarray(rnn.h2h.weight ).flatten())

loss_all_all = [];
iteration1 = [];
iteration1.append(0);

#%%
num_cycles = 1;

rates_all = np.zeros((hidden_size, T, num_cycles));
outputs_all = np.zeros((output_size, T, num_cycles));
loss_all = np.zeros((T, num_cycles));

#%%

train_it = 1

for n_cyc in range(num_cycles):
    
    print('cycle ' + str(n_cyc+1) + ' of ' + str(num_cycles))
    
    for n_t in range(T-1):
        
        optimizer.zero_grad()
        
        output, rate_new = rnn.forward(input_sig[:,n_t], rate)
        
        rates_all[:,n_t+1,n_cyc] = rate_new.detach().numpy()[0,:];
        
        rate = rate_new.detach();
    
        outputs_all[:,n_t+1,n_cyc] = output.detach().numpy()[0,:];
        
        if train_it:
            target2 = torch.argmax(target[:,n_t]) * torch.ones(1) # torch.tensor()
            
            loss2 = loss(output, target2.long())
            
            loss2.backward() # retain_graph=True
            optimizer.step()
            
            loss_all[n_t,n_cyc] = loss2.item()
            
            loss_all_all.append(loss2.item())
        
        iteration1.append(iteration1[-1]+1);

print('Done')
    

#%%

sm_bin = 500;

kernel = np.ones(sm_bin)/sm_bin

loss_all_smooth = np.convolve(loss_all_all, kernel, mode='same')


#%%

plt.figure()
plt.plot(loss_all_all)
plt.plot(loss_all_smooth)

plt.figure()
plt.plot(loss_all_smooth)

#%%
# rates_all = np.zeros((hidden_size, T));
# outputs_all = np.zeros((output_size, T));

# rate = torch.empty(1, hidden_size);

# nn.init.uniform_(rate, a=0, b=1)

# rates_all[:,0] = rate.detach().numpy()[0,:];

# for n_t in range(T-1):
    
#     rate_new = tanh1(i2h(input_sig[:,n_t]) + h2h(rate))
    
#     rate_new = (1-alpha)*rate + alpha*rate_new
    
#     rates_all[:,n_t+1] = rate_new.detach().numpy()[0,:];
    
#     rate = rate_new;
    
#     output = softmax1(h2o(rate_new))
    
#     outputs_all[:,n_t+1] = output.detach().numpy()[0,:];
    
#     target2 = torch.argmax(target[:,n_t]) * torch.ones(1) # torch.tensor()
    
#     loss2 = loss(output, target2.long())
    
    
    
    
    

#%% testing

# print(np.std(np.asarray(i2h.weight.data).flatten()))
# print(np.std(np.asarray(h2h.weight.data).flatten()))

# print(np.mean(np.asarray(i2h(input_sig[:,n_t]).data).flatten()))
# print(np.std(np.asarray(i2h(input_sig[:,n_t]).data).flatten()))

# x1 = rate.data;
# x1 = i2h(input_sig[:,n_t]).data;
# print(np.mean(np.asarray(x1).flatten()))
# print(np.std(np.asarray(x1).flatten()))

#%%

num_plots = 10;

plot_cells = np.sort(sample(range(hidden_size), num_plots));

spec = gridspec.GridSpec(ncols=1, nrows=6, height_ratios=[4, 1, 3, 1, 1, 1])

plt.figure()
ax1 = plt.subplot(spec[0])
for n_plt in range(num_plots):  
    shift = n_plt*2.5    
    ax1.plot(rates_all[plot_cells[n_plt],:,-1]+shift)
plt.title('example cells')
plt.subplot(spec[1], sharex=ax1)
plt.plot(np.mean(rates_all[:,:,-1], axis=0))
plt.title('population average')
plt.subplot(spec[2], sharex=ax1)
plt.imshow(input_sig.data) #  , aspect=5
plt.title('inputs')
plt.subplot(spec[3], sharex=ax1)
plt.imshow(target.data) # , aspect=100
plt.title('target')
plt.subplot(spec[4], sharex=ax1)
plt.imshow(outputs_all[:,:,-1]) # , aspect=100
plt.title('outputs')
plt.subplot(spec[5], sharex=ax1)
plt.plot(loss_all[:,-1]) # , aspect=100
plt.title('outputs')
#%%
pca = PCA();
pca.fit(rates_all[:,:,0])

#%%
plt.figure()
plt.subplot(1,2,1);
plt.plot(pca.explained_variance_, 'o')
plt.title('Explained Variance'); plt.xlabel('component')

plt.subplot(1,2,2);
plt.plot(pca.components_[0,:], pca.components_[1,:])
plt.title('PCA components'); plt.xlabel('PC1'); plt.ylabel('PC2')

#%%

# dim = 256;

# radius = 4
# pat1 = np.zeros((radius*2+1,radius*2+1));

# for n_m in range(radius*2+1):
#     for n_n in range(radius*2+1):
#         if np.sqrt((radius-n_m)**2 + (radius-n_n)**2)<radius:
#             pat1[n_m,n_n] = 1;
        

# plt.figure()
# plt.imshow(pat1)

# coords = np.round(np.random.uniform(low=0.0, high=(dim-1), size=(hidden_size,2)))

# frame1 = np.zeros((dim,dim, hidden_size))

# for n_frame in range(hidden_size):
#     temp_frame = frame1[:,:,n_frame]
#     temp_frame[int(coords[n_frame,0]), int(coords[n_frame,1])] = 1;
#     temp_frame2 = signal.convolve2d(temp_frame, pat1, mode='same')
#     frame1[:,:,n_frame] = temp_frame2

# plt.figure()
# plt.imshow(frame1[:,:,1])



#%%% make movie if want

# rates_all2 = rates_all - np.min(rates_all)
# rates_all2 = rates_all2/np.max(rates_all2)

# frame2 = frame1.reshape(256*256,250)

# frame2 = np.dot(frame2, rates_all2).T

# frame2 = frame2.reshape(10000,256,256)

# skimage.io.imsave('test2.tif', frame2)


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
flat_dist_met = pdist(rates_all[:,:,0], metric='cosine');
cs = 1- squareform(flat_dist_met);
res_linkage = linkage(flat_dist_met, method='average')

N = len(cs)
res_ord = seriation(res_linkage,N, N + N -2)
    
#%%

cs_ord = 1- squareform(pdist(rates_all[res_ord,:,0], metric='cosine'));

plt.figure()
plt.imshow(cs_ord)
plt.title('cosine similarity sorted')

#%% cell tuning


data_save = {"rates_all": rates_all, "loss_all_smooth": loss_all_smooth,
             "input_sig": np.asarray(input_sig.data), "target": np.asarray(target.data),
             "output": outputs_all, "g": g, "dt": dt, "tau": tau, "hidden_size": hidden_size,
             'h2h_weight': np.asarray(rnn.h2h.weight.data), 'train_it': train_it,
             'i2h_weight': np.asarray(rnn.i2h.weight.data),
             'h2o_weight': np.asarray(rnn.h2o.weight.data)}

#save_fname = 'rnn_out_8_25_21_1_complex_g_tau10_5cycles.mat'

save_fname = 'rnn_out_8_25_21_1_tones.mat'

savemat(fpath+ save_fname, data_save)

#%%

# plot_cells = np.sort(sample(range(hidden_size), num_plots));

# spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[4, 1])

# plt.figure()
# ax1 = plt.subplot(spec[0])
# for n_plt in range(num_plots):  
#     shift = n_plt*2.5    
#     ax1.plot(rates_all[plot_cells[n_plt],output_mat[1,:],-1]+shift)
# plt.title('example cells')
# plt.subplot(spec[1], sharex=ax1)
# plt.plot(output_mat[1,output_mat[1,:]]) # , aspect=100
# plt.title('target')
