clear;
close all;

%%

fpath =  'C:\Users\Administrator\Desktop\yuriy\RNN_project\data\';

fname = 'rnn_out_8_25_21_1';

%fname = 'rnn_out_8_25_21_1_tones_g_tau10.mat';

%fname = 'rnn_out_8_25_21_1_tones_g_tau10_bias-2.mat';

%fname = 'rnn_out_8_25_21_1_tones_g_tau10_notrain.mat';

%fname = 'rnn_out_8_25_21_1_tones_g_tau10.mat';

%fname = 'rnn_out_8_25_21_1_tones_g_tau10_trained_noise_resp.mat';

%fname = 'rnn_out_8_25_21_1_tones_g_tau10_notrain_noise_resp.mat';
%%

data = load([fpath fname]);

%%
figure;
imagesc(data.rates_all)
%%

num_plot_cells = 10;

plt_cells = randsample(data.hidden_size, num_plot_cells);

figure; 
subplot(6,1,1:5)
hold on; axis tight;
for n_cell = 1:num_plot_cells
    shift = (n_cell - 1) * 2;
    plot(data.rates_all(plt_cells(1),end-1000:end) + shift)
end
subplot(6,1,6);
imagesc(data.target(:,end-1000:end))

%%

x = corr(data.target(2,:)', data.rates_all');

figure; histogram(x)

[~, max_idx] = max(x);

[~, min_idx] = min(x);

figure; 
ax1 = subplot(2,1,1);
hold on; axis tight;
plot(data.rates_all(max_idx,:) + shift)
ax2 = subplot(2,1,2);
imagesc(data.target(:,:));
linkaxes([ax1, ax2], 'x');

%%


% 
% figure; hold on
% plot(data.target(2,:))
% plot(diff(data.target(2,:)))
% 
% figure; imagesc(squeeze(stim_data_sort)')
% figure; imagesc(squeeze(trial_data_sort(max_idx,:,:))')
% figure; imagesc(squeeze(trial_data_sort(min_idx,:,:))')
% 
% figure; hold on;
% for n_tr = 1:size(trial_data_sort,3)
%     plot(squeeze(trial_data_sort(max_idx,:,n_tr)), 'color', [.4 .4 .4])
% end
% plot(mean(trial_data_sort(max_idx,:,:),3), 'r', 'Linewidth', 2)
% title(sprintf('cell %d', max_idx))
% 
% 
% figure; hold on;
% for n_tr = 1:size(trial_data_sort,3)
%     plot(squeeze(trial_data_sort(min_idx,:,n_tr)), 'color', [.4 .4 .4])
% end
% plot(mean(trial_data_sort(min_idx,:,:),3), 'r', 'Linewidth', 2)
% title(sprintf('cell %d', min_idx))
% 
% 
% 
% for n_cell = 1:50
%     figure; hold on;
%     for n_tr = 1:size(trial_data_sort,3)
%         plot(squeeze(trial_data_sort(n_cell,:,n_tr)), 'color', [.4 .4 .4])
%     end
%     plot(mean(trial_data_sort(n_cell,:,:),3), 'r', 'Linewidth', 2)
%     title(sprintf('cell %d', n_cell))
% end
% 
% n_cell = 242;
% for n_tr = 2:11
%     stim_frame_index = find(diff(data.target(n_tr,:))>0);
%     stim_data_sort = f_get_stim_trig_resp(data.target(n_tr,:), stim_frame_index, [5 30]);
% 
%     trial_data_sort = f_get_stim_trig_resp(data.rates_all, stim_frame_index, [5 30]);
%     
%     figure; subplot(); hold on;
%     for n_tr2 = 1:size(trial_data_sort,3)
%         plot(squeeze(trial_data_sort(n_cell,:,n_tr2)), 'color', [.4 .4 .4])
%     end
%     plot(mean(trial_data_sort(n_cell,:,:),3), 'r', 'Linewidth', 2)
%     title(sprintf('cell %d', n_cell))
% end
% 
% 
% for n_tr = 2:11
%     stim_frame_index = find(diff(data.target(n_tr,:))>0);
%     stim_data_sort = f_get_stim_trig_resp(data.target(n_tr,:), stim_frame_index, [5 30]);
% 
%     trial_data_sort = f_get_stim_trig_resp(data.rates_all, stim_frame_index, [5 30]);
%     
%     figure; subplot()
%     imagesc(squeeze(stim_data_sort)')
%     imagesc(squeeze(trial_data_sort(41,:,:))')
%     
% end

%%


stim_frame_index = find(diff(data.target(2,:))>0);

stim_data_sort = f_get_stim_trig_resp(data.target(2,:), stim_frame_index, [5 30]);
trial_data_sort = f_get_stim_trig_resp(data.rates_all, stim_frame_index, [5 30]);

n_cell = max_idx;
x = squeeze(trial_data_sort(n_cell,:,:));

si = f_pdist_YS(x', 'cosine');

hparams.plot_dist_mat = 1;
hclust = f_hcluster_wrap(x',hparams);

[coeff,score,latent,tsquared,explained,mu] = pca(x');

figure; plot(explained)


%%


n_tr = 3;

stim_frame_index = find(diff(data.target(n_tr,:))>0);
stim_data_sort = f_get_stim_trig_resp(data.target(n_tr,:), stim_frame_index, [5 30]);

trial_data_sort = f_get_stim_trig_resp(data.rates_all, stim_frame_index, [5 30]);

trial_data_sort_2d = reshape(trial_data_sort, 250, []);



hparams.plot_dist_mat = 1;
hparams.plot_clusters = 0;
hparams.num_clust = 10;
hclust = f_hcluster_wrap(trial_data_sort_2d,hparams);


% x = squeeze(mean(trial_data_sort(:,10:25,:),2));
% hclust2 = f_hcluster_wrap(x',hparams);


x2 = reshape(trial_data_sort, [], 200);
hclust3 = f_hcluster_wrap(x2',hparams);

trial_data_sort_sort = trial_data_sort(hclust.dend_order,:,hclust3.dend_order);

trial_data_sort_sort_2d = reshape(trial_data_sort_sort, 250, []);


figure; imagesc(trial_data_sort_sort_2d)



%%

stim_frame_index = find(diff(data.target(n_tr,:))>0);
stim_data_sort2 = f_get_stim_trig_resp(data.target(:,:), stim_frame_index, [50 30]);

x = stim_data_sort2(:,:, hclust3.clust_ident == 2);

figure; imagesc(reshape(x, 11, []))

%%



[~,~,~,~,explained_h2h,~] = pca(data.h2h_weight');

[~,~,~,~,explained_i2h,~] = pca(data.i2h_weight');

[~,~,~,~,explained_h20,~] = pca(data.h2o_weight');

figure; hold on;
plot(explained_h2h, 'o', 'linewidth', 2)
plot(explained_i2h, 'o', 'linewidth', 2)
plot(explained_h20, 'o', 'linewidth', 2)
title('weigth mat');
legend('h2h', 'i2h', 'h2o');

%%
[coeff,scores,~,~,explained_rates,~] = pca(data.rates_all');

figure; hold on;
plot(explained_rates, 'o', 'linewidth', 2)
title('rates eig')

figure; 
plot(scores(:,1), scores(:,2))
xlabel('pc1')
xlabel('pc2')

hparams.plot_dist_mat = 1;
hparams.plot_clusters = 0;
hparams.num_clust = 10;
hclust = f_hcluster_wrap(data.rates_all,hparams);

