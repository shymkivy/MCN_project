close all;
clear;
%%

num_neurons = 250;

g = 4;

dt = 1;
tau = 10;
alpha = dt/tau;

T = 10000;
pl_time = (dt:dt:T);
num_T = numel(pl_time);

%%
std1 = g/sqrt(num_neurons); % the larger the network, the weaker each connection is

W = randn(num_neurons,num_neurons);
W = (W - mean(W(:)));
W = W*std1;

rate_0 = rand(num_neurons,1);

%%

rate_all = zeros(num_neurons, num_T);
rate_all(:,1) = rate_0;

for n_t = 2:num_T
    
    rate_t = f_tanh(W*rate_all(:,n_t-1));
    
    rate_t = (1 - alpha)*rate_all(:,n_t-1) + alpha*rate_t;
    
    rate_all(:,n_t) = rate_t;
end


%%
num_pl = 10;
figure; hold on;
for n_pl = 1:num_pl
    shift = (n_pl-1)*2.5;
    plot(pl_time, rate_all(n_pl,:)+shift, 'Linewidth', 2)
end
xlabel('sec')
%%

[coeff,score,~,~,explained,~] = pca(rate_all);

figure; plot(explained, 'o')
figure; plot(coeff(:,1), coeff(:,2))

e = eig(rate_all*rate_all')
figure; plot(e)


