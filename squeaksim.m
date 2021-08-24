%% simulating squeaks

clear;
close all;

%%
%voc_data.voc_trace_duration

%voc_data.voc_trace_bandwidth
% duration 1-3 sec
% bandwidth 20Khz
% 40 - 65 kHz

%%

Fs = 250000;

voc_times = 1:10;
duration = 2;

freq_low = 40000;
freq_high = 60000;

%%
squeak_type =7;
y = f_squeak_generator(squeak_type, Fs, duration, freq_low, freq_high);
%%

all_types = (3:8)';

duraion = 2;

freq_low_high = [30000, 50000; 40000, 60000];

%%

num_types = numel(all_types);
num_durations = numel(duraion);
num_freqs = size(freq_low_high,1);

voc_seq = [repmat(all_types, [num_durations, 1]), repmat(duraion, [num_types*num_durations,1])];

voc_seq3 = [];

for n_fr = 1:num_freqs
    voc_seq2 = [voc_seq, repmat(freq_low_high(n_fr,:), [num_types*num_durations, 1])];
    voc_seq3 = [voc_seq3; voc_seq2];
end

voc_seq_fn = [(1:size(voc_seq3,1))', voc_seq3];

% unique type, stim_type, duration, low freq, high freq
%%
repeats = 10;
isi = 2;
rand_pad = 1;

voc_seq_fn2 = repmat(voc_seq_fn, [repeats , 1]);

num_voc = size(voc_seq_fn2,1);

new_ord = randsample(num_voc,num_voc);

voc_seq_fn2r = voc_seq_fn2(new_ord,:);

%%

stim_times = zeros(num_voc,1);

dt = 1/Fs;

y_all = [];
for n_stim = 1:num_voc
    y_all = [y_all, dt:dt:(isi + rand(1)*rand_pad)];
    
    stim_times(n_stim) = numel(y_all)+1;

    y_sq = f_squeak_generator(voc_seq_fn2r(n_stim,2), Fs, voc_seq_fn2r(n_stim,3), voc_seq_fn2r(n_stim,4), voc_seq_fn2r(n_stim,5));
    
    y_all = [y_all, y_sq];
end

y_all = [y_all, dt:dt:(isi + rand(1)*rand_pad)];

siz_y_all = numel(y_all);

%%
y_alln = y_all + randn(1, siz_y_all);

figure; plot(y_alln)


%%
windowsize = round(Fs * 0.0032); % 0.0032 s
noverlap = round(Fs * 0.0032 * .50); % 50 % 
nfft = round(Fs * 0.0032); %0.0032 s

[s, fr, ti] = spectrogram(y_alln,windowsize,noverlap,nfft,Fs,'yaxis');

spec = abs(s);

fig1 = figure; 
imagesc(ti,fr/1000, spec)
fig1.CurrentAxes.YDir = 'normal';

%%
output_mat = zeros(num_voc, numel(ti));



