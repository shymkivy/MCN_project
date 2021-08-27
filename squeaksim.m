%% simulating squeaks

clear;
close all;

%%

fpath =  'C:\Users\Administrator\Desktop\yuriy\RNN_project\data\';

%%
%voc_data.voc_trace_duration

%voc_data.voc_trace_bandwidth
% duration 1-3 sec
% bandwidth 20Khz
% 40 - 65 kHz

%%

Fs = 250000;

voc_times = 1:10;

freq_low = 40000;
freq_high = 60000;

%%
% squeak_type =7;
% y = f_squeak_generator(squeak_type, Fs, duration, freq_low, freq_high);
%%
%(type, duration, freq_low, freq_high);
duration = 0.04;

% all_types = [1, duration, 40, 55;...
%              2, duration, 45, 60;...
%              3, duration, 40, 55;...
%              4, duration, 45, 60;...
%              5, duration, 40, 55;...
%              6, duration, 40, 55;...
%              7, duration, 40, 60;...
%              8, duration, 40, 60;...
%              5, duration, 45, 60;...
%              6, duration, 45, 60];%(3:8)'; ; 7

all_types = [0, duration, 40, 60;...
             0, duration, 42, 60;...
             0, duration, 44, 60;...
             0, duration, 46, 60;...
             0, duration, 48, 60;...
             0, duration, 50, 60;...
             0, duration, 52, 60;...
             0, duration, 54, 60;...
             0, duration, 56, 60;...
             0, duration, 58, 60];%(3:8)'; ; 7

%%

num_types = size(all_types,1);

all_types_idx = [(1:num_types)', all_types];

all_types_gen = cell(num_types,1);

for n_type = 1:num_types
    all_types_gen{n_type} = f_squeak_generator(all_types_idx(n_type,2), Fs, all_types_idx(n_type,3), all_types_idx(n_type,4), all_types_idx(n_type,5));
end


% unique_stim_type, stim_type, duration, low freq, high freq
%%
repeats = 200;
isi = .03;
rand_pad = 0.02;

%%

voc_seq_fn = repmat(all_types_idx, [repeats , 1]);

num_voc = size(voc_seq_fn,1);

new_ord = randsample(num_voc,num_voc);

voc_seq_fnr = voc_seq_fn(new_ord,:);

%%

stim_times = zeros(num_voc,1);

dt = 1/Fs;

y_all = [];
w1 = waitbar(0,'generating stim');
for n_stim = 1:num_voc
    y_all = [y_all, dt:dt:(isi + rand(1)*rand_pad)];
    
    stim_times(n_stim) = numel(y_all)+1;
    
    y_sq = all_types_gen{voc_seq_fnr(n_stim,1)};
    
    y_all = [y_all, y_sq];
    waitbar(n_stim/num_voc, w1);
end
delete(w1);

y_all = [y_all, dt:dt:(isi + rand(1)*rand_pad)];

siz_y_all = numel(y_all);

%%
y_alln = y_all + randn(1, siz_y_all);

%figure; plot(y_alln)


%%
windowsize = round(Fs * 0.0032); % 0.0032 s
noverlap = round(Fs * 0.0032 * .50); % 50 % 
nfft = round(Fs * 0.0032); %0.0032 s

[s, fr, ti] = spectrogram(y_alln,windowsize,noverlap,nfft,Fs,'yaxis');

spec = abs(s);

[~, fr_start] = min((fr - 35000).^2);
[~, fr_end] = min((fr - 65000).^2);

spec_cut = spec(fr_start:fr_end,:);
fr_cut = fr(fr_start:fr_end);

%%
output_mat = zeros(num_types, numel(ti));
output_mat_delayed = zeros(num_types, numel(ti));
output_delay = .010;

for n_stim = 1:num_voc
    
    stim_time1 = stim_times(n_stim)/Fs;
    
    onset_idx = find(ti>stim_time1);
    onset_idx = onset_idx(1);
    
    offset_idx = find(ti>(stim_time1+voc_seq_fnr(n_stim,3)));
    offset_idx = offset_idx(1);
    
    output_mat(voc_seq_fnr(n_stim,1),onset_idx:offset_idx) = 1;
    
    onset_idx = find(ti>(stim_time1+output_delay));
    onset_idx = onset_idx(1);
    
    offset_idx = find(ti>(stim_time1+voc_seq_fnr(n_stim,3)+output_delay));
    offset_idx = offset_idx(1);
    
    output_mat_delayed(voc_seq_fnr(n_stim,1),onset_idx:offset_idx) = 1;
    
end

output_mat = [~logical(sum(output_mat,1)); output_mat];
output_mat_delayed = [~logical(sum(output_mat_delayed,1)); output_mat_delayed];

figure; 
ax1 = subplot(3,1,1);
imagesc(ti,fr_cut/1000, spec_cut)
ax1.YDir = 'normal';
ax2 = subplot(3,1,2);
imagesc(ti, 1:num_voc, output_mat)
ax3 = subplot(3,1,3);
imagesc(ti, 1:num_voc, output_mat_delayed)
linkaxes([ax1 ax2 ax3], 'x');

%%
spec_data.spec_cut = spec_cut;
spec_data.fr_cut = fr_cut;
spec_data.ti = ti;
spec_data.output_mat = output_mat;
spec_data.output_mat_delayed = output_mat_delayed;
spec_data.voc_seq = voc_seq_fnr;
spec_data.num_voc = num_voc;

save([fpath 'sim_spec_10complex_200rep_stim_8_25_21_1'], 'spec_data');
