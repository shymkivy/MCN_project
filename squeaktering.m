%% aka clustering squeaks

clear;
close all;

%%
fpath =  'C:\Users\Administrator\Desktop\yuriy\RNN_project\data\';

fname_voc = '3BB89473_FECompStim2014-01-15_0000002.WAV';
fname_det = '3BB89473_FECompStim2014-01-15_0000002_08_22_21_manual';
% fname_clust = 'Extracted Contours_2_8_83_21_new.mat';
% fname_clust2 = 'K-Means Model_2_8_23_21_new.mat';


%% laod audio
[y,Fs] = audioread([fpath '\' fname_voc]);

windowsize = round(Fs * 0.0032); % 0.0032 s
noverlap = round(Fs * 0.0032 * .50); % 50 % 
nfft = round(Fs * 0.0032); %0.0032 s

[s, fr, ti] = spectrogram(y,windowsize,noverlap,nfft,Fs,'yaxis');

spec = abs(s);

%%
data_det = load([fpath '\' fname_det]);

% data_clust = load([fpath '\' fname_clust]);
% 
% data_clust2 = load([fpath '\' fname_clust2]);


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
% 
% spec_n = spec - mean(spec(:));
% spec_n = spec_n/std(spec_n(:));
% 
% voc_all = cell(num_voc,1);
% for n_voc = 1:num_voc
%     box1 = data_det.Calls(n_voc,:).Box;
% 
%     [~, start_idx] = min((box1(1) - ti).^2);
%     [~, end_idx] = min((box1(1)+box1(3) - ti).^2);
%     voc1 = spec_n(:,start_idx:end_idx);
%     voc1 = voc1 - mean(voc1(:));
%     voc1 = voc1/std(voc1(:));
%     [~, min_idx1] = min((fr - box1(2)*1000).^2);
%     [~, min_idx2] = min((fr - (box1(2)+box1(4))*1000).^2);
%     voc1(1:min_idx1,:) = 0;
%     voc1(min_idx2:end,:) = 0;
%     voc_all{n_voc} = voc1;
% end
% 
% xcorr_all = zeros(num_voc,1);
% for n_voc = 1:num_voc
%     C = xcorr2(voc_all{1},voc_all{n_voc});
%     xcorr_all(n_voc) = max(mean(C,1));
% end
% 
% [max_val, max_idx]=max(xcorr_all)
% 
% C = xcorr2(voc_all{1},voc_all{2});
% 
% figure; imagesc(voc_all{1})
% figure; imagesc(voc_all{260})
% figure; imagesc(voc_all{9})
% 
% figure; plot(mean(C,1))
% 
% voc1 = voc_all{1};
% voc2 = voc_all{458};
% 
% fig1 = figure;
% imagesc(voc1)
% fig1.CurrentAxes.YDir = 'normal';
% fig1.CurrentAxes.CLim = [0 5];
% fig1 = figure;
% imagesc(voc2)
% fig1.CurrentAxes.YDir = 'normal';
% fig1.CurrentAxes.CLim = [0 5];

%%
% 
% clust_type = str2double(string(data_det.Calls.Type));
% 
% n_clust = 12;
% data_cut = data_det.Calls(clust_type == n_clust,:);
% data_clust_cut = data_clust.ClusteringData(clust_type == n_clust,:);
% 
% figure; hold on; axis tight
% current_t = 0;
% num_voc2 = size(data_cut,1);
% for n_voc = 1:num_voc2
%     im1 = data_clust_cut(n_voc,:).Spectrogram{1};
%     im_num_time = size(im1,2);
%     
%     imagesc(ti(1:im_num_time)+current_t,fr, flipud(im1));
%     plot(data_clust_cut(n_voc,:).xTime{1}+current_t, data_clust_cut(n_voc,:).xFreq{1}*1000, 'LineWidth', 2, 'color', 'r')
%     
%     current_t = current_t + ti(im_num_time);
% end
% title(sprintf('Clust %d',n_clust ))
% 

%%

figure; histogram(data_det.Calls.Type)

%%

voc_data = data_det.Calls;
num_voc = size(voc_data,1);
voc_data.voc_box_duration = zeros(num_voc,1);
voc_data.voc_box_bandwidth = zeros(num_voc,1);
voc_data.voc_trace_time = cell(num_voc,1);
voc_data.voc_trace_freqs = cell(num_voc,1);
voc_data.voc_trace_duration = zeros(num_voc,1);
voc_data.voc_trace_bandwidth = zeros(num_voc,1);
voc_data.voc_trace_slope = zeros(num_voc,3);

% for smoothing
sigma_pixels = 1;
kernel_half_size = ceil(sqrt(-log(0.1)*2*sigma_pixels^2));
[X_gaus,Y_gaus] = meshgrid((-kernel_half_size):kernel_half_size, (-kernel_half_size):.5:(kernel_half_size));
conv_kernel = exp(-(X_gaus.^2 + Y_gaus.^2)/(2*sigma_pixels^2));
conv_kernel = conv_kernel/sum(conv_kernel(:));

spec_n = spec - mean(spec(:));
spec_n = spec_n/std(spec_n(:));

z_thresh = 1;

for n_voc = 1:num_voc
    box1 = voc_data(n_voc,:).Box;
    [~, start_idx] = min((box1(1) - ti).^2);
    [~, end_idx] = min((box1(1)+box1(3) - ti).^2);
    [~, min_fr1] = min((fr - box1(2)*1000).^2);
    [~, max_fr1] = min((fr - (box1(2)+box1(4))*1000).^2);
    
    voc_data.voc_box_duration(n_voc) = ti(end_idx) - ti(start_idx);
    voc_data.voc_box_bandwidth(n_voc) = (fr(max_fr1) - fr(min_fr1))/1000;
    
    voc1 = spec_n(:,start_idx:end_idx);
    voc2 = voc1;
    
    voc1(1:min_fr1,:) = 0;
    voc1(max_fr1:end,:) = 0;
    
    z_fac = std(voc1(:));
    thresh1 = (z_thresh*z_fac);
    
    %voc1(thresh1>voc1) = 0;
    
    im_sm2 = conv2(voc1,conv_kernel, 'same');
    [max_vals, max_idx] = max(im_sm2);
    time1 = 1:size(voc1,2);
    
    time2 = ti(start_idx:end_idx);
    time3 = time2(max_vals>thresh1);
    voc_data.voc_trace_time{n_voc} = time3';
    
    freqs2 = fr(max_idx);
    freqs3 = freqs2(max_vals>thresh1);
    voc_data.voc_trace_freqs{n_voc} = freqs3;
    
    voc_data.voc_trace_duration(n_voc) = time3(end) - time3(1);
    voc_data.voc_trace_bandwidth(n_voc) = (freqs3(end) - freqs3(1))/1000;
    
    diff_all = diff(freqs3/1000)*mean(diff(time3));
    
    step1 = floor(numel(diff_all)/3);
   
    voc_data.voc_trace_slope(n_voc,1) = mean(diff_all(1:step1));
    voc_data.voc_trace_slope(n_voc,2) = mean(diff_all((step1+1):(2*step1)));
    voc_data.voc_trace_slope(n_voc,3) = mean(diff_all((2*step1+1):(3*step1)));
end


%%

fig1 = figure; hold on; axis tight;
imagesc(ti,fr/1000,abs(s));
fig1.CurrentAxes.YDir = 'normal';
fig1.CurrentAxes.CLim = [0 5];

for n_sig = 1:num_voc

    box_data = voc_data(n_sig,:).Box;
    rectangle('Position',box_data,'EdgeColor',colors1(n_clust,:))

    plot(voc_data(n_sig,:).voc_trace_time{1}, voc_data(n_sig,:).voc_trace_freqs{1}/1000, 'r')

end

%%

feature_vec = zeros(num_voc,6);

feature_vec(:,1) = voc_data.voc_trace_duration;
feature_vec(:,2) = voc_data.voc_trace_bandwidth;
feature_vec(:,3) = voc_data.voc_trace_slope(:,1);
feature_vec(:,4) = voc_data.voc_trace_slope(:,2);
feature_vec(:,5) = voc_data.voc_trace_slope(:,3);
feature_vec(:,6) = (voc_data.Box(:,2) + voc_data.Box(:,4)/2)*2;

for n_row = 1:size(feature_vec,2)
    temp = feature_vec(:,n_row);
    temp = temp - mean(temp);
    temp = temp/std(temp);
    feature_vec(:,n_row) = temp;
end

%%

Y = tsne(feature_vec);

idx = kmeans(feature_vec,10, 'Distance', 'sqeuclidean'); % cosine

figure;
gscatter(Y(:,1),Y(:,2),idx)

voc_data.clust_label = idx;

%%

clusters1 = unique(idx);
colors1 = jet(numel(clusters1));
figure;
ax1 = subplot(2,1,1); hold on; axis tight;
imagesc(ti,fr/1000,abs(s));
ax1.YDir = 'normal';
ax1.CLim = [0 5];

for n_clust = 1:numel(clusters1)
    for n_sig = 1:num_voc
        if data_det.Calls(n_sig,:).Accept
            if voc_data.clust_label(n_sig) == n_clust
                box_data = voc_data(n_sig,:).Box;
                rectangle('Position',box_data,'EdgeColor',colors1(n_clust,:))
                
                plot(voc_data(n_sig,:).voc_trace_time{1}, voc_data(n_sig,:).voc_trace_freqs{1}/1000, 'r')
            end
        end
    end
end

ax2 = subplot(2,1,2);
imagesc(ti, clusters1, output_call)
linkaxes([ax1 ax2],'x')


%%

for n_clust = 1:10
    fig1 = figure; hold on;
    fig1.CurrentAxes.YDir = 'normal';
    fig1.CurrentAxes.CLim = [0 5];

    current_t = 0;
    for n_sig = 1:num_voc
        if voc_data.clust_label(n_sig) == n_clust

            box1 = voc_data(n_sig,:).Box;

            [~, start_idx] = min((box1(1) - ti).^2);
            [~, end_idx] = min((box1(1)+box1(3) - ti).^2);

            siz1 = numel(start_idx:end_idx);
            time1 = ti(1:siz1)+current_t;
            imagesc(time1,fr/1000, spec(:,start_idx:end_idx));

            %rectangle('Position',box_data,'EdgeColor',colors1(n_clust,:))

            time2 = voc_data(n_sig,:).voc_trace_time{1} - box1(1) + current_t;

            plot(time2, voc_data(n_sig,:).voc_trace_freqs{1}/1000, 'r')

            current_t = current_t + ti(siz1);
        end
    end
    title(sprintf('Clsut %d', n_clust))
end

%%

% clust 5 ok

figure; histogram(voc_data.voc_trace_duration)
title('voc duration')

figure; histogram(voc_data.voc_trace_bandwidth)
title('voc bandwidth')

figure; histogram(voc_data.Box(:,2))
title('voc lower')


