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
isi = 0.5;

repeats = 100;
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

sig_t = (1:sig_size)*dt;

% figure; 
% plot(sig_t, sig_alln);
% axis tight

%%

window_size = 0.0032;

windowsize = round(Fs * window_size); % Fs * 0.0032s     3.2ms windows
noverlap = round(Fs * window_size * .50); % Fs * 0.0032 * .50
nfft = []; %round(Fs * 0.020); %0.0032 s % related to num freqs in spectrogram, depends on size of window

% windowsize = round(Fs * dt*250); % Fs * 0.0032s
% noverlap = round(Fs * dt*250 * .50); % Fs * 0.0032 * .50
% nfft = round(Fs * 0.0032); %0.0032 s % related to num freqs in spectrogram

[s, fr, ti] = spectrogram(sig_alln',windowsize,noverlap,nfft,Fs,'yaxis');
ti = ti - ti(1);

spec = abs(s);

fr_cut_min = 38000;
fr_cut_max = 62000;

fr_min_idx = find(fr<fr_cut_min, 1,'last');
fr_max_idx = find(fr>fr_cut_max, 1, 'first');

spec_cut = spec(fr_min_idx:fr_max_idx,:);
fr_cut = fr(fr_min_idx:fr_max_idx);

figure; imagesc(ti, fr_cut/1000, spec_cut);
ylabel('Freq (kHz)'); xlabel('Time (sec)');

dt_spec = ti(1);
df_spec = fr(2) - fr(1);
%% interpolate input spectrogram
dt_target = 0.05;
df_target = 500;

ti_ip = ti(1):dt_target:ti(end);
%fr_ip = fr_cut_min:df_target:fr_cut_max;
fr_ip = fr_cut;

[X, Y] = meshgrid(ti_ip, fr_ip);

spec_ip = interp2(ti, fr_cut, spec_cut, X, Y);

figure; imagesc(ti_ip, fr_ip/1000, spec_ip);
ylabel('Freq (kHz)'); xlabel('Time (sec)');

%%

output_mat = zeros(num_freqs, numel(ti_ip));
output_mat_delayed = zeros(num_freqs, numel(ti_ip));
output_delay = .010;

for n_stim = 1:num_voc
    
    stim_time1 = stim_times(n_stim);
    
    onset_idx = find(ti_ip>stim_time1);
    onset_idx = onset_idx(1);
    
    offset_idx = find(ti_ip>(stim_time1+duration));
    offset_idx = offset_idx(1);
    
    output_mat(voc_seq_fnr(n_stim),onset_idx:offset_idx) = 1;
    
    onset_idx = find(ti_ip>(stim_time1+output_delay));
    onset_idx = onset_idx(1);
    
    offset_idx = find(ti_ip>(stim_time1+duration+output_delay));
    offset_idx = offset_idx(1);
    
    output_mat_delayed(voc_seq_fnr(n_stim,1),onset_idx:offset_idx) = 1;
    
end

output_mat = [~logical(sum(output_mat,1)); output_mat];
output_mat_delayed = [~logical(sum(output_mat_delayed,1)); output_mat_delayed];

figure; 
ax1 = subplot(3,1,1);
imagesc(ti_ip,fr_ip/1000, spec_ip)
ax1.YDir = 'normal';
ax2 = subplot(3,1,2);
imagesc(ti, 1:num_voc, output_mat)
ax3 = subplot(3,1,3);
imagesc(ti, 1:num_voc, output_mat_delayed)
linkaxes([ax1 ax2 ax3], 'x');

%%
stim_params.Fs = Fs;
stim_params.stim_type = stim_type;
stim_params.duration = duration;
stim_params.isi = isi;
stim_params.repeats = repeats;
stim_params.rand_pad = rand_pad;
stim_params.freq_low = freq_low;
stim_params.freq_high = freq_high;
stim_params.num_freqs = num_freqs;
stim_params.increase_fac = increase_fac;

spec_data.stim_params = stim_params;
spec_data.spec_cut = spec_ip;
spec_data.fr_cut = fr_ip;
spec_data.ti = ti_ip;
spec_data.stim_times = stim_times;
spec_data.output_mat = output_mat;
spec_data.output_mat_delayed = output_mat_delayed;
spec_data.output_delay = output_delay;
spec_data.voc_seq = voc_seq_fnr;
spec_data.num_voc = num_voc;

temp_time = clock;
fname = sprintf('sim_spec_%d%s_%dreps_%.1fisi_%.0fdt_%d_%d_%d_%dh_%dm.mat',num_freqs, stim_type,...
    repeats, isi,(ti_ip(2) - ti_ip(1))*1000, temp_time(2), temp_time(3), temp_time(1)-2000, temp_time(4), temp_time(5));
disp(fname);
save([fpath fname], 'spec_data');
