# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 15:01:28 2021

@author: ys2605
"""


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
