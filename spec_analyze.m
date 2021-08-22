clear;
close all;


fpath =  'C:\Users\Administrator\Desktop\yuriy\RNN_project\data\';

fname_voc = '3BB89473_FECompStim2014-01-15_0000001.WAV';
fname_det = '3BB89473_FECompStim2014-01-15_0000001 2021-08-18  6_06 PM.mat';
fname_clust = 'Extracted Contours_8_22_FEComp_stim_1.mat';

%% laod audio
[y,Fs] = audioread([fpath '\' fname_voc]);

windowsize = round(Fs * 0.0032); % 0.0032 s
noverlap = round(Fs * 0.0032 * .50); % 50 % 
nfft = round(Fs * 0.0032); %0.0032 s

[s, fr, ti] = spectrogram(y,windowsize,noverlap,nfft,Fs,'yaxis');




%%
data_det = load([fpath '\' fname_det]);

data_clust = load([fpath '\' fname_clust]);


%%

call_time = data_det.Calls(1,:).Box(1);


%% plots

fig1 = figure; hold on; axis tight;
imagesc(ti,fr/1000,abs(s));
fig1.CurrentAxes.YDir = 'normal';
fig1.Children.CLim = [0 10];
plot(data_det.Calls(2,:).Box(1), data_det.Calls(2,:).Box(2), 'o')




