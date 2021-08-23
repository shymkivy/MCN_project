clear;
close all;

%%
fpath =  'C:\Users\Administrator\Desktop\yuriy\RNN_project\data\';

fname_voc = '3BB89473_FECompStim2014-01-15_0000002.WAV';
fname_det = '3BB89473_FECompStim2014-01-15_0000002_08_22_21_manual';
%fname_clust = 'Extracted Contours_8_22_FEComp_stim_1_two.mat';
%fname_clust2 = 'K-Means Model_8_22_FEComp_stim_1_two.mat';


%% laod audio
[y,Fs] = audioread([fpath '\' fname_voc]);

windowsize = round(Fs * 0.0032); % 0.0032 s
noverlap = round(Fs * 0.0032 * .50); % 50 % 
nfft = round(Fs * 0.0032); %0.0032 s

[s, fr, ti] = spectrogram(y,windowsize,noverlap,nfft,Fs,'yaxis');

spec = abs(s);

%%
data_det = load([fpath '\' fname_det]);

%data_clust = load([fpath '\' fname_clust]);

%data_clust2 = load([fpath '\' fname_clust2]);


%%
call_time = data_det.Calls(1,:).Box(1);

%% plots

clusters1 = unique(double(string(data_det.Calls.Type)));
clusters1(isnan(clusters1)) = [];
num_voc = size(data_det.Calls,1);

% generate outputs
output_call = zeros(numel(clusters1), numel(ti));
for n_clust = 1:numel(clusters1)
    for n_sig = 1:num_voc
        if data_det.Calls(n_sig,:).Accept
            if double(string(data_det.Calls(n_sig,:).Type)) == n_clust 
                box_data = data_det.Calls(n_sig,:).Box;
                [~, start_idx] = min((box_data(1) - ti).^2);
                [~, end_idx] = min((box_data(1)+box_data(3) - ti).^2);
                output_call(n_clust, start_idx:end_idx) = 1;
            end
        end
    end
end
%

%% plot 
colors1 = jet(numel(clusters1));
figure;
ax1 = subplot(2,1,1); hold on; axis tight;
imagesc(ti,fr/1000,abs(s));
ax1.YDir = 'normal';
ax1.CLim = [0 5];

for n_clust = 1:numel(clusters1)
    for n_sig = 1:num_voc
        if data_det.Calls(n_sig,:).Accept
            if double(string(data_det.Calls(n_sig,:).Type)) == n_clust
                box_data = data_det.Calls(n_sig,:).Box;
                rectangle('Position',box_data,'EdgeColor',colors1(n_clust,:))
            end
        end
    end
end

ax2 = subplot(2,1,2);
imagesc(ti, clusters1, output_call)
linkaxes([ax1 ax2],'x')
%%

for n_clust = 1:numel(clusters1)
    fig1 = figure;
    imagesc(abs(s(:,logical(output_call(n_clust,:)))))
    fig1.CurrentAxes.YDir = 'normal';
    fig1.CurrentAxes.CLim = [0 5];
    title(sprintf('Cluster %d', n_clust))
end

%%

spec_n = spec - mean(spec(:));
spec_n = spec_n/std(spec_n(:));

voc_all = cell(num_voc,1);
for n_voc = 1:num_voc
    box1 = data_det.Calls(n_voc,:).Box;

    [~, start_idx] = min((box1(1) - ti).^2);
    [~, end_idx] = min((box1(1)+box1(3) - ti).^2);
    voc1 = spec_n(:,start_idx:end_idx);
    voc_all{n_voc} = voc1;
end

xcorr_all = zeros(num_voc,1);
for n_voc = 1:num_voc
    C = xcorr2(voc_all{1},voc_all{n_voc});
    xcorr_all(n_voc) = max(mean(C,1));
end

figure; imagesc(C)

figure; plot(mean(C,1))

voc1 = voc_all{1};
voc2 = voc_all{2};

fig1 = figure;
imagesc(voc_all{1})
fig1.CurrentAxes.YDir = 'normal';
fig1.CurrentAxes.CLim = [0 5];
fig1 = figure;
imagesc(voc_all{2})
fig1.CurrentAxes.YDir = 'normal';
fig1.CurrentAxes.CLim = [0 5];
