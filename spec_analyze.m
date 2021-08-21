clear;
close all;


fpath =  'C:\Users\Administrator\Desktop\yuriy\vocalizations';

fname = '3BB89473_FECompStim2014-01-15_0000001.WAV';

[y,Fs] = audioread([fpath '\' fname]);

windowsize = round(Fs * 0.0032); % 0.0032 s
noverlap = round(Fs * 0.0032 * .50); % 50 % 
nfft = round(Fs * 0.0032); %0.0032 s

[s, fr, ti, p] = spectrogram(y,windowsize,noverlap,nfft,Fs,'yaxis');


fig1 = figure;
imagesc(ti,fr/1000,abs(s));
fig1.CurrentAxes.YDir = 'normal';
