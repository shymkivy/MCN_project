# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 15:33:58 2021

@author: Administrator
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from random import sample, random
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy import signal
import skimage.io
import math

#%%
plt.close('all')

#%%
input_size = 50;

hidden_size = 250;

T = 10000;

g = 5;

dt = 1;
tau = 10;
alpha = dt/tau;


#%%

# W is inside i2h.weight

i2h = nn.Linear(input_size, hidden_size)
h2h = nn.Linear(hidden_size, hidden_size)

#softmax = nn.LogSoftmax(dim=1)

tanh1 = nn.Tanh();

#%% fix weights


w = torch.empty(hidden_size, hidden_size)
nn.init.normal_(w, mean=0.0, std = 1)

std1 = g/np.sqrt(hidden_size);

w = w - np.mean(w.detach().numpy());
w = w * std1;


h2h.weight.data = w;

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

input_sig = torch.zeros(input_size, T)


#%%

# plt.figure();
# plt.imshow(input_sig[:,0:300]);

#%%

rates_all = np.zeros((hidden_size, T));
rate = torch.empty(1, hidden_size);

nn.init.uniform_(rate, a=0, b=1)

rates_all[:,0] = rate.detach().numpy()[0,:];

for n_t in range(T-1):
    
    rate_new = tanh1(i2h(input_sig[:,n_t]) + h2h(rate))
    
    rate_new = (1-alpha)*rate + alpha*rate_new
    
    rates_all[:,n_t+1] = rate_new.detach().numpy()[0,:];
    
    rate = rate_new;


#%%

num_plots = 10;

plot_cells = np.sort(sample(range(hidden_size), num_plots));

plt.figure()
for n_plt in range(num_plots):  
    shift = n_plt*2.5    
    plt.plot(rates_all[plot_cells[n_plt],:]+shift)
plt.title('example cells')

#%%
pca = PCA();
pca.fit(rates_all)

#%%
plt.figure()
plt.subplot(1,2,1);
plt.plot(pca.explained_variance_, 'o')
plt.title('Explained Variance'); plt.xlabel('component')

plt.subplot(1,2,2);
plt.plot(pca.components_[0,:], pca.components_[1,:])
plt.title('PCA components'); plt.xlabel('PC1'); plt.ylabel('PC2')

#%%

dim = 256;

radius = 4
pat1 = np.zeros((radius*2+1,radius*2+1));

for n_m in range(radius*2+1):
    for n_n in range(radius*2+1):
        if np.sqrt((radius-n_m)**2 + (radius-n_n)**2)<radius:
            pat1[n_m,n_n] = 1;
        

plt.figure()
plt.imshow(pat1)

coords = np.round(np.random.uniform(low=0.0, high=(dim-1), size=(hidden_size,2)))

frame1 = np.zeros((dim,dim, hidden_size))

for n_frame in range(hidden_size):
    temp_frame = frame1[:,:,n_frame]
    temp_frame[int(coords[n_frame,0]), int(coords[n_frame,1])] = 1;
    temp_frame2 = signal.convolve2d(temp_frame, pat1, mode='same')
    frame1[:,:,n_frame] = temp_frame2

plt.figure()
plt.imshow(frame1[:,:,1])



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
flat_dist_met = pdist(rates_all, metric='cosine');
cs = 1- squareform(flat_dist_met);
res_linkage = linkage(flat_dist_met, method='average')

N = len(cs)
res_ord = seriation(res_linkage,N, N + N -2)
    
#%%

cs_ord = 1- squareform(pdist(rates_all[res_ord], metric='cosine'));

plt.figure()
plt.imshow(cs_ord)
plt.title('cosine similarity sorted')
