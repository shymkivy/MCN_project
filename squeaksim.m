%% simulating squeaks

clear;
close all;

%%

fpath =  'C:\Users\ys2605\Desktop\stuff\RNN_stuff\RNN_data\';

%%
%voc_data.voc_trace_duration

%voc_data.voc_trace_bandwidth
% duration 1-3 sec
% bandwidth 20Khz
% 40 - 65 kHz

%%

stim_type = 'tones'; % 'tones', 'complex'

Fs = 250000;

duration = 0.5;
isi = 1;

repeats = 10;
rand_pad = 0.025;


freq_low = 40; % kHz
freq_high = 60;

num_freqs = 10;
increase_fac = (freq_high/freq_low).^(1/(num_freqs-1));

%%
% squeak_type =7;
% y = f_squeak_generator(squeak_type, Fs, duration, freq_low, freq_high);

% for complex tone

%%
%(type, duration, freq_low, freq_high);

all_types_idx = (1:num_freqs)';
all_types_gen = cell(num_freqs,1);

if strcmpi(stim_type, 'tones')
    
    freqs_all = zeros(num_freqs,1);
    freqs_all(1) = freq_low;
    for n_fr = 2:num_freqs
        freqs_all(n_fr) = freqs_all(n_fr-1)*increase_fac;
    end
    
    for n_type = 1:num_freqs
        all_types_gen{n_type} = f_squeak_generator(0, Fs, duration, freqs_all(n_type));
    end
    
elseif strcmpi(stim_type, 'complex')
    cpx_type = [1 2 3 4 5 6 7 8 5 6];
    cpx_low =  [40 45 40 45 40 40 40 40 45 45];
    cpx_high = [55 60 55 60 55 55 60 60 60 60];
    
    for n_type = 1:num_freqs
        all_types_gen{n_type} = f_squeak_generator(cpx_type(n_type), Fs, duration, cpx_low(n_type), cpx_high(n_type));
    end
    
end


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


%%

voc_seq_fn = repmat(all_types_idx, [repeats , 1]);
num_voc = size(voc_seq_fn,1);
new_ord = randsample(num_voc,num_voc);
voc_seq_fnr = voc_seq_fn(new_ord,:);

%%
dt = 1/Fs;

stim_size = duration/dt;
stim_times = zeros(num_voc,1);


sig1 = cell(num_voc,1);

last_time = 0;
for n_stim = 1:num_voc
    new_time = last_time + isi + rand(1)*rand_pad;
    stim_times(n_stim) = new_time;
    
    last_time = new_time + duration;
end
sig_size = round((last_time+isi)/dt);

sig_all = zeros(sig_size,1);
for n_stim = 1:num_voc
    temp_idx = round(stim_times(n_stim)/dt);
    sig_all(temp_idx:(temp_idx+stim_size-1)) = all_types_gen{voc_seq_fnr(n_stim)};
end



%%
sig_alln = sig_all + randn(sig_size,1);

figure; plot(sig_alln)

%%

windowsize = round(Fs * 0.0032); % 0.0032 s
noverlap = round(Fs * 0.0032 * .50); % 50 % 
nfft = round(Fs * 0.0032); %0.0032 s

[s, fr, ti] = spectrogram(sig_alln',windowsize,noverlap,nfft,Fs,'yaxis');

spec = abs(s);

[~, fr_start] = min((fr - 35000).^2);
[~, fr_end] = min((fr - 65000).^2);

spec_cut = spec(fr_start:fr_end,:);
fr_cut = fr(fr_start:fr_end);


figure; imagesc(fr, ti, spec_cut)
%%
output_mat = zeros(num_freqs, numel(ti));
output_mat_delayed = zeros(num_freqs, numel(ti));
output_delay = .010;

for n_stim = 1:num_voc
    
    stim_time1 = stim_times(n_stim)/Fs;
    
    onset_idx = find(ti>stim_time1);
    onset_idx = onset_idx(1);
    
    offset_idx = find(ti>(stim_time1+duration));
    offset_idx = offset_idx(1);
    
    output_mat(voc_seq_fnr(n_stim),onset_idx:offset_idx) = 1;
    
    onset_idx = find(ti>(stim_time1+output_delay));
    onset_idx = onset_idx(1);
    
    offset_idx = find(ti>(stim_time1+duration+output_delay));
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
