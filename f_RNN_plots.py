# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:06:53 2024

@author: ys2605
"""

import numpy as np

from scipy.spatial.distance import pdist, squareform #, cdist

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec

from f_RNN_process import f_get_rdc_trav, f_trial_ave_ctx_rd

#%%
def f_plot_dred_rates(trials_oddball_ctx_cut, comp_out4d, red_dd_seq, pl_params, params, title_tag=''):
    num_runs_plot = pl_params['num_runs_plot']
    plot_trials = pl_params['plot_trials'] #800
    color_ctx = pl_params['color_ctx']  # 0 = red; 1 = dd
    mark_red = pl_params['mark_red']
    mark_dd = pl_params['mark_dd']
    
    colors1 = cm.jet(np.linspace(0,1,params['num_freq_stim']))
    
    trial_len, num_tr, num_batch, num_cells = comp_out4d.shape
    
    comp_out3d = np.reshape(comp_out4d, (trial_len*num_tr, num_batch, num_cells), order='F')
    
    plot_runs = range(num_runs_plot)#[0, 1, 5]
    
    plot_T = plot_trials*trial_len
    
    plot_pc = pl_params['plot_pc']
    for n_pcpl in range(len(plot_pc)):
        plot_pc2 = plot_pc[n_pcpl]
        plt.figure()
        #plt.subplot(1,2,2);
        for n_run in plot_runs: #num_bouts
            temp_ob_tr = trials_oddball_ctx_cut[:,n_run]
            
            red_idx = temp_ob_tr == round(params['num_ctx']-1)
            dd_idx = temp_ob_tr == params['num_ctx']
            
            temp_comp4d = comp_out4d[:,:plot_trials,n_run,:]
            
            plt.plot(comp_out3d[:plot_T, n_run, plot_pc2[0]-1], comp_out3d[:plot_T, n_run, plot_pc2[1]-1], color=colors1[red_dd_seq[color_ctx,n_run]-1,:])
            
            if mark_red:
                plt.plot(temp_comp4d[4:15,:,plot_pc2[0]-1][:,red_idx[:plot_trials]], temp_comp4d[4:15,:,plot_pc2[1]-1][:,red_idx[:plot_trials]], '.b')
                plt.plot(temp_comp4d[4,:,plot_pc2[0]-1][red_idx[:plot_trials]], temp_comp4d[4,:,plot_pc2[1]-1][red_idx[:plot_trials]], 'ob')
        
            if mark_dd: 
                plt.plot(temp_comp4d[4:15,:,plot_pc2[0]-1][:,dd_idx[:plot_trials]], temp_comp4d[4:15,:,plot_pc2[1]-1][:,dd_idx[:plot_trials]], '.r')
                plt.plot(temp_comp4d[4,:,plot_pc2[0]-1][dd_idx[:plot_trials]], temp_comp4d[4,:,plot_pc2[1]-1][dd_idx[:plot_trials]], 'or')
  
        plt.title('PCA components; %s' % title_tag); plt.xlabel('PC%d' % plot_pc2[0]); plt.ylabel('PC%d' % plot_pc2[1])
        
    
    if 0:
        n_run  = 0
        
        plt.figure()
        plt.plot(comp_out3d[:plot_T, n_run, 0], comp_out3d[:plot_T, n_run, 1])
        plt.plot(comp_out3d[0, n_run, 0], comp_out3d[0, n_run, 1], '*')
        plt.title('PCA components; bout %d' % n_run); plt.xlabel('PC1'); plt.ylabel('PC2')
        
        plot_T = 800
        idx3 = np.linspace(0, plot_T-trial_len, round(plot_T/trial_len)).astype(int)
        
        plt.figure()
        plt.plot(comp_out3d[:plot_T, n_run, 0], comp_out3d[:plot_T, n_run, 1])
        plt.plot(comp_out3d[idx3, n_run, 0], comp_out3d[idx3, n_run, 1], 'o')
        plt.plot(comp_out3d[0, n_run, 0], comp_out3d[0, n_run, 1], '*')
        plt.title('PCA components; bout %d' % n_run); plt.xlabel('PC1'); plt.ylabel('PC2')


#%%
def f_plot_dred_rates2(trials_oddball_ctx_cut, comp_out4d, plot_pcs=[[1,2],[3,4]], num_runs_plot=int(1e10), num_trials_plot=int(1e10), run_labels = [], run_colors=[], trial_stim_on = [], mark_red=False, mark_dev=False, title_tag=''):

    trial_len, num_tr, num_runs, num_cells = comp_out4d.shape
    
    num_trials_plot2 = np.min((num_trials_plot, num_tr))
    num_runs_plot2 = np.min((num_runs_plot, num_runs))
    
    if not len(trial_stim_on):
        trial_stim_on = np.zeros((trial_len), dtype=bool)
        trial_stim_on[round(trial_len/4):round(trial_len/4*3)] = 1
    stim_on_bin = np.where(trial_stim_on)[0][0]
    
    if not len(run_labels):
        run_labels = np.arange(num_runs_plot2)
    
    if not len(run_colors):
        labels_all = np.unique(run_labels)
        num_tt = labels_all.shape[0]
        run_colors = cm.jet(np.linspace(0,1,num_tt))

    comp_out3d = np.reshape(comp_out4d, (trial_len*num_tr, num_runs, num_cells), order='F')
    
    plot_runs = range(num_runs_plot2)#[0, 1, 5]
    
    plot_T = num_trials_plot2*trial_len
    
    for n_pcpl in range(len(plot_pcs)):
        plot_pc2 = plot_pcs[n_pcpl]
        plt.figure()
        #plt.subplot(1,2,2);
        for n_run in plot_runs: #num_bouts
            temp_ob_tr = trials_oddball_ctx_cut[:,n_run]
            
            temp_comp4d = comp_out4d[:,:num_trials_plot2,n_run,:]
            
            col_idx = run_labels[n_run] == labels_all
            plt.plot(comp_out3d[:plot_T, n_run, plot_pc2[0]-1], comp_out3d[:plot_T, n_run, plot_pc2[1]-1], color=run_colors[col_idx,:])
            
            if mark_red:
                red_idx = temp_ob_tr == 0
                data_xy = temp_comp4d[:,red_idx[:num_trials_plot2],:]
                x_data = data_xy[:,:,plot_pc2[0]-1]
                y_data = data_xy[:,:,plot_pc2[1]-1]
                
                plt.plot(x_data[trial_stim_on,:], y_data[trial_stim_on,:], '.b')
                plt.plot(x_data[stim_on_bin,:], y_data[stim_on_bin,:], 'ob')
        
            if mark_dev:
                dd_idx = temp_ob_tr == 1
                data_xy = temp_comp4d[:,dd_idx[:num_trials_plot2],:]
                x_data = data_xy[:,:,plot_pc2[0]-1]
                y_data = data_xy[:,:,plot_pc2[1]-1]
                
                plt.plot(x_data[trial_stim_on,:], y_data[trial_stim_on,:], '.r')
                plt.plot(x_data[stim_on_bin,:], y_data[stim_on_bin,:], 'or')
  
        plt.title('PCA components; %s' % title_tag); plt.xlabel('PC%d' % plot_pc2[0]); plt.ylabel('PC%d' % plot_pc2[1])

#%%
def f_plot_dred_rates3(trials_freq, comp_out4d, plot_pcs=[[1,2],[3,4]], num_runs_plot=int(1e10), num_trials_plot=int(1e10), run_colors=[], trial_stim_on = [], title_tag='', rescale_colors=True):
    # trying different color scheme
    
    trial_len, num_tr, num_runs, num_cells = comp_out4d.shape
    
    num_trials_plot2 = np.min((num_trials_plot, num_tr))
    num_runs_plot2 = np.min((num_runs_plot, num_runs))
    
    if not len(trial_stim_on):
        trial_stim_on = np.zeros((trial_len), dtype=bool)
        trial_stim_on[round(trial_len/4):round(trial_len/4*3)] = 1
    stim_on_bin = np.where(trial_stim_on)[0][0]
    
    if not len(run_colors):
        if rescale_colors:
            num_tt = np.max(trials_freq[:,:num_runs_plot2])
        else:
            num_tt = np.max(trials_freq)
        run_colors = cm.jet(np.linspace(0,1,num_tt))

    comp_out3d = np.reshape(comp_out4d, (trial_len*num_tr, num_runs, num_cells), order='F')
    
    plot_runs = range(num_runs_plot2)#[0, 1, 5]
    plot_T = num_trials_plot2*trial_len
    
    for n_pcpl in range(len(plot_pcs)):
        plot_pc2 = plot_pcs[n_pcpl]
        plt.figure()
        #plt.subplot(1,2,2);
        for n_run in plot_runs: #num_bouts
            trials_freq2 = trials_freq[:,n_run]
            
            temp_comp4d = comp_out4d[:,:num_trials_plot2,n_run,:]
            
            plt.plot(comp_out3d[:plot_T, n_run, plot_pc2[0]-1], comp_out3d[:plot_T, n_run, plot_pc2[1]-1], color='grey') #run_colors[col_idx,:]
            
            for n_tr in range(num_trials_plot2):
                if trials_freq2[n_tr]:
                    plt.plot(temp_comp4d[trial_stim_on,n_tr,plot_pc2[0]-1], temp_comp4d[trial_stim_on,n_tr,plot_pc2[1]-1], color=run_colors[trials_freq2[n_tr]-1,:])
                    plt.plot(temp_comp4d[stim_on_bin,n_tr,plot_pc2[0]-1], temp_comp4d[stim_on_bin,n_tr,plot_pc2[1]-1], '.', color=run_colors[trials_freq2[n_tr]-1,:])
                    
        plt.title('PCA components; %s' % title_tag); plt.xlabel('PC%d' % plot_pc2[0]); plt.ylabel('PC%d' % plot_pc2[1])


#%%
def f_plot_dred_rates3d(trials_oddball_ctx_cut, comp_out4d, plot_pcs=[[1, 2, 3]], num_runs_plot=int(1e10), num_trials_plot=int(1e10), run_labels = [], run_colors=[], trial_stim_on = [], mark_red=False, mark_dev=False, el_az_ro=[30, -60, 0], title_tag=''):

    trial_len, num_tr, num_runs, num_cells = comp_out4d.shape
    
    num_trials_plot2 = np.min((num_trials_plot, num_tr))
    num_runs_plot2 = np.min((num_runs_plot, num_runs))
    
    if not len(trial_stim_on):
        trial_stim_on = np.zeros((trial_len), dtype=bool)
        trial_stim_on[5:15] = 1
    stim_on_bin = np.where(trial_stim_on)[0][0]
    
    if not len(run_labels):
        run_labels = np.arange(num_runs_plot2)
    
    if not len(run_colors):
        labels_all = np.unique(run_labels)
        num_tt = labels_all.shape[0]
        run_colors = cm.jet(np.linspace(0,1,num_tt))

    comp_out3d = np.reshape(comp_out4d, (trial_len*num_tr, num_runs, num_cells), order='F')
    
    plot_patches = range(num_runs_plot2)#[0, 1, 5]
    
    plot_T = num_trials_plot2*trial_len
    
    for n_pcpl in range(len(plot_pcs)):
        plot_pc2 = plot_pcs[n_pcpl]
        ax = plt.figure().add_subplot(projection='3d')
        #plt.subplot(1,2,2);
        for n_bt in plot_patches: #num_bouts
            temp_ob_tr = trials_oddball_ctx_cut[:,n_bt]
            
            temp_comp4d = comp_out4d[:,:num_trials_plot2,n_bt,:]
            
            ax.plot(comp_out3d[:plot_T, n_bt, plot_pc2[0]-1], comp_out3d[:plot_T, n_bt, plot_pc2[1]-1], comp_out3d[:plot_T, n_bt, plot_pc2[2]-1], color=run_colors[run_labels[n_bt]-1,:])

            if mark_red:
                red_idx = temp_ob_tr == 0
                data_xy = temp_comp4d[:,red_idx[:num_trials_plot2],:]
                x_data = data_xy[:,:,plot_pc2[0]-1]
                y_data = data_xy[:,:,plot_pc2[1]-1]
                z_data = data_xy[:,:,plot_pc2[2]-1]
                
                ax.plot(x_data[trial_stim_on,:], y_data[trial_stim_on,:], z_data[trial_stim_on,:], '.b')
                ax.plot(x_data[stim_on_bin,:], y_data[stim_on_bin,:], z_data[stim_on_bin,:], 'ob')
        
            if mark_dev:
                dd_idx = temp_ob_tr == 1
                data_xy = temp_comp4d[:,dd_idx[:num_trials_plot2],:]
                x_data = data_xy[:,:,plot_pc2[0]-1]
                y_data = data_xy[:,:,plot_pc2[1]-1]
                z_data = data_xy[:,:,plot_pc2[2]-1]
                
                ax.plot(x_data[trial_stim_on,:], y_data[trial_stim_on,:], z_data[trial_stim_on,:], '.r')
                ax.plot(x_data[stim_on_bin,:], y_data[stim_on_bin,:], z_data[stim_on_bin,:], 'or')
                
            ax.view_init(elev=el_az_ro[0], azim=el_az_ro[1], roll=el_az_ro[2])  # 30, -60, 0
  
        plt.title('PCA components; %s' % title_tag)
        ax.set_xlabel('PC%d' % plot_pc2[0])
        ax.set_ylabel('PC%d' % plot_pc2[1])
        ax.set_zlabel('PC%d' % plot_pc2[2])

#%%
def f_plot_traj_speed(rates, ob_data, n_run, start_idx = 0, title_tag = ''):
    dist1 = squareform(pdist(rates[start_idx:,n_run,:], metric='euclidean'))
    dist2 = np.diag(dist1, 1)
    
    spec2 = gridspec.GridSpec(ncols=1, nrows=3, height_ratios=[3, 1, 2])
    
    plt.figure()
    ax1 = plt.subplot(spec2[0])
    plt.imshow(ob_data['input_oddball'][start_idx:-1,n_run,:].T, aspect="auto", interpolation='none')
    plt.title(title_tag)
    plt.subplot(spec2[1], sharex=ax1)
    plt.imshow(ob_data['taret_oddball_ctx'][start_idx:-1,n_run,:].T, aspect="auto", interpolation='none')
    plt.subplot(spec2[2], sharex=ax1)
    plt.plot(dist2)
    plt.ylabel('euclidean dist')
    plt.xlabel('trials')    

def f_plot_resp_distances(rates4d_cut, trials_oddball_ctx_cut, ob_data1, params, choose_idx = 'center', variab_tr_idx = 0, plot_tr_idx = 0, title_tag=''):

    if variab_tr_idx:
        var_seq = ob_data1['dev_stim']
    else:
        var_seq = ob_data1['red_stim']
      
    trial_len, num_tr, num_batch, num_cells = rates4d_cut.shape
    
    #
    trial_ave_rd = f_trial_ave_ctx_rd(rates4d_cut, trials_oddball_ctx_cut, params)
    
    if choose_idx == 'center':
        cur_tr = var_seq[round(len(var_seq)/2)]
    elif choose_idx == 'sample':
        cur_tr = np.random.choice(var_seq, size=1)[0]
        
    idx_cur = ob_data1['red_dd_seq'][variab_tr_idx,:] == cur_tr
    base_resp = np.mean(trial_ave_rd[plot_tr_idx,:,idx_cur,:], axis=0)
    base_resp1d = np.reshape(base_resp, (trial_len*num_cells), order='F')
    
    num_var = len(var_seq)
    
    dist_all = np.zeros((num_var))
    dist_all_cos = np.zeros((num_var))
    has_data = np.zeros((num_var), dtype=bool)
    
    for n_tr in range(num_var):
        idx1 = ob_data1['red_dd_seq'][variab_tr_idx,:] == var_seq[n_tr]
        if np.sum(idx1):
            temp1 = np.mean(trial_ave_rd[plot_tr_idx,:,idx1,:], axis=0)
            temp1_1d = np.reshape(temp1, (trial_len*num_cells), order='F')
            
            dist_all[n_tr] = pdist(np.vstack((base_resp1d,temp1_1d)), metric='euclidean')[0]
            dist_all_cos[n_tr] = pdist(np.vstack((base_resp1d,temp1_1d)), metric='cosine')[0]
            has_data[n_tr] = 1
        
    
    plt.figure()
    plt.plot(var_seq[has_data], dist_all[has_data])
    plt.ylabel('euclidean dist')
    if variab_tr_idx:
        plt.title('const red, var dd; ref tr = %d; %s' % (cur_tr, title_tag))
        plt.xlabel('dd stim')
    else:
        plt.title('const dd, var red; ref tr = %d; %s' % (cur_tr, title_tag))
        plt.xlabel('red stim')
    
    plt.figure()
    plt.plot(var_seq[has_data], dist_all_cos[has_data])
    plt.ylabel('cosine dist')
    plt.xlabel('red stim')
    if variab_tr_idx:
        plt.title('const red, var dd; ref tr = %d; %s' % (cur_tr, title_tag))
        plt.xlabel('dd stim')
    else:
        plt.title('const dd, var red; ref tr = %d; %s' % (cur_tr, title_tag))
        plt.xlabel('red stim')


#%%
def f_plot_dred_pcs(data3d, comp_list, color_idx, color_ctx, colors, title_tag=''):
    
    num_t, num_run, num_pcs = data3d.shape
    
    for n_pl in range(len(comp_list)):
        plt.figure()
        for n_run in range(num_run):
            plt.plot(data3d[:,n_run, comp_list[n_pl][0]], data3d[:,n_run, comp_list[n_pl][1]], color=colors[color_idx[color_ctx,n_run]-1,:])
            plt.plot(data3d[0,n_run, comp_list[n_pl][0]], data3d[0,n_run, comp_list[n_pl][1]], 'o', color=colors[color_idx[color_ctx,n_run]-1,:])
            plt.xlabel('comp %d' % comp_list[n_pl][0])
            plt.ylabel('comp %d' % comp_list[n_pl][1])
        plt.title('%s; pl %d' % (title_tag, n_pl))

#%%

def f_plot_rnn_weights(rnn, rnn0, rnn0c=[]):
    
    alpha1 = 0.3
    density1 = False
    
    wr1 = rnn.h2h.weight.detach().cpu().numpy().flatten()
    wi1 = rnn.i2h.weight.detach().cpu().numpy().flatten()
    wo1 = rnn.h2o_ctx.weight.detach().cpu().numpy().flatten()
    
    br1 = rnn.h2h.bias.detach().cpu().numpy().flatten()
    bi1 = rnn.i2h.bias.detach().cpu().numpy().flatten()
    bo1 = rnn.h2o_ctx.bias.detach().cpu().numpy().flatten()
    
    
    wr0 = rnn0.h2h.weight.detach().cpu().numpy().flatten()
    wi0 = rnn0.i2h.weight.detach().cpu().numpy().flatten()
    wo0 = rnn0.h2o_ctx.weight.detach().cpu().numpy().flatten()
    
    br0 = rnn0.h2h.bias.detach().cpu().numpy().flatten()
    bi0 = rnn0.i2h.bias.detach().cpu().numpy().flatten()
    bo0 = rnn0.h2o_ctx.bias.detach().cpu().numpy().flatten()
    
    if rnn0c != []:
        wr0c = rnn0c.h2h.weight.detach().cpu().numpy().flatten()
        wi0c = rnn0c.i2h.weight.detach().cpu().numpy().flatten()
        wo0c = rnn0c.h2o_ctx.weight.detach().cpu().numpy().flatten()
        
        br0c = rnn0c.h2h.bias.detach().cpu().numpy().flatten()
        bi0c = rnn0c.i2h.bias.detach().cpu().numpy().flatten()
        bo0c = rnn0c.h2o_ctx.bias.detach().cpu().numpy().flatten()
    
        # compensate
        rnn0c.h2h.weight.data = rnn0c.h2h.weight.data/np.std(wr0c)*np.std(wr1)
        rnn0c.i2h.weight.data = rnn0c.i2h.weight.data/np.std(wi0c)*np.std(wi1)
        rnn0c.h2o_ctx.weight.data = rnn0c.h2o_ctx.weight.data/np.std(wo0c)*np.std(wo1)
        
        rnn0c.h2h.bias.data = rnn0c.h2h.bias.data/np.std(br0c)*np.std(br1)
        rnn0c.i2h.bias.data = rnn0c.i2h.bias.data/np.std(bi0c)*np.std(bi1)
        rnn0c.h2o_ctx.bias.data = rnn0c.h2o_ctx.bias.data/np.std(bo0c)*np.std(bo1)
    
        # get again
        wr0c = rnn0c.h2h.weight.detach().cpu().numpy().flatten()
        wi0c = rnn0c.i2h.weight.detach().cpu().numpy().flatten()
        wo0c = rnn0c.h2o_ctx.weight.detach().cpu().numpy().flatten()
        
        br0c = rnn0c.h2h.bias.detach().cpu().numpy().flatten()
        bi0c = rnn0c.i2h.bias.detach().cpu().numpy().flatten()
        bo0c = rnn0c.h2o_ctx.bias.detach().cpu().numpy().flatten()
    
    
    plt.figure()
    plt.subplot(311)
    plt.hist(wr1, density=density1, alpha=alpha1)
    plt.hist(wr0, density=density1, alpha=alpha1)
    if rnn0c != []:
        plt.hist(wr0c, density=density1, alpha=alpha1)
        plt.legend(('trained; std=%.2f' % np.std(wr1), 'untrained; std=%.2f' % np.std(wr0), 'untrained comp; std=%.2f' % np.std(wr0c)))
    else:
        plt.legend(('trained; std=%.2f' % np.std(wr1), 'untrained; std=%.2f' % np.std(wr0)))
    plt.title('Recurrent W')
    
    plt.subplot(312)
    plt.hist(wi1, density=density1, alpha=alpha1)
    plt.hist(wi0, density=density1, alpha=alpha1)
    if rnn0c != []:
        plt.hist(wi0c, density=density1, alpha=alpha1)
        plt.legend(('trained; std=%.2f' % np.std(wi1), 'untrained; std=%.2f' % np.std(wi0), 'untrained comp; std=%.2f' % np.std(wi0c)))
    else:
        plt.legend(('trained; std=%.2f' % np.std(wi1), 'untrained; std=%.2f' % np.std(wi0)))
    plt.title('Input W')
    
    plt.subplot(313)
    plt.hist(wo1, density=density1, alpha=alpha1)
    plt.hist(wo0, density=density1, alpha=alpha1)
    if rnn0c != []:
        plt.hist(wo0c, density=density1, alpha=alpha1)
        plt.legend(('trained; std=%.2f' % np.std(wo1), 'untrained; std=%.2f' % np.std(wo0), 'untrained comp; std=%.2f' % np.std(wo0c)))
    else:
        plt.legend(('trained; std=%.2f' % np.std(wo1), 'untrained; std=%.2f' % np.std(wo0)))
    plt.title('Output W')
    
    
    plt.figure()
    plt.subplot(311)
    plt.hist(br1, density=density1, alpha=alpha1)
    plt.hist(br0, density=density1, alpha=alpha1)
    if rnn0c != []:
        plt.hist(br0c, density=density1, alpha=alpha1)
        plt.legend(('trained; std=%.2f' % np.std(br1), 'untrained; std=%.2f' % np.std(br0), 'untrained comp; std=%.2f' % np.std(br0c)))
    else:
        plt.legend(('trained; std=%.2f' % np.std(br1), 'untrained; std=%.2f' % np.std(br0)))
    plt.title('Recurrent bias')
    
    plt.subplot(312)
    plt.hist(bi1, density=density1, alpha=alpha1)
    plt.hist(bi0, density=density1, alpha=alpha1)
    if rnn0c != []:
        plt.hist(bi0c, density=density1, alpha=alpha1)
        plt.legend(('trained; std=%.2f' % np.std(bi1), 'untrained; std=%.2f' % np.std(bi0), 'untrained comp; std=%.2f' % np.std(bi0c)))
    else:
        plt.legend(('trained; std=%.2f' % np.std(bi1), 'untrained; std=%.2f' % np.std(bi0)))
    plt.title('Input bias')
    
    plt.subplot(313)
    plt.hist(bo1, density=density1, alpha=alpha1)
    plt.hist(bo0, density=density1, alpha=alpha1)
    if rnn0c != []:
        plt.hist(bo0c, density=density1, alpha=alpha1)
        plt.legend(('trained; std=%.2f' % np.std(bo1), 'untrained; std=%.2f' % np.std(bo0), 'untrained comp; std=%.2f' % np.std(bo0c)))
    else:
        plt.legend(('trained; std=%.2f' % np.std(bo1), 'untrained; std=%.2f' % np.std(bo0)))
    plt.title('Output bias')


def f_plot_rnn_weights2(rnn_list, legend_list, alpha = 0.3, density = False):

    w_rec = []
    w_in = []
    w_out_ctx = []
    w_out_freq = []
    
    b_rec = []
    b_in = []
    b_out_ctx = []
    b_out_freq = []
    
    num_rnn = len(rnn_list)
    
    for n_rnn in range(num_rnn):
        w_rec.append(rnn_list[n_rnn].h2h.weight.detach().cpu().numpy().flatten())
        w_in.append(rnn_list[n_rnn].i2h.weight.detach().cpu().numpy().flatten())
        w_out_ctx.append(rnn_list[n_rnn].h2o_ctx.weight.detach().cpu().numpy().flatten())
        w_out_freq.append(rnn_list[n_rnn].h2o.weight.detach().cpu().numpy().flatten())
 
        
        b_rec.append(rnn_list[n_rnn].h2h.bias.detach().cpu().numpy().flatten())
        b_in.append(rnn_list[n_rnn].i2h.bias.detach().cpu().numpy().flatten())
        b_out_ctx.append(rnn_list[n_rnn].h2o_ctx.bias.detach().cpu().numpy().flatten())
        b_out_freq.append(rnn_list[n_rnn].h2o.bias.detach().cpu().numpy().flatten())
                
                
    # weights
    plt.figure()
    plt.subplot(411)
    leg2 = []
    for n_rnn in range(num_rnn):
        plt.hist(w_rec[n_rnn], density=density, alpha=alpha)
        leg2.append('%s; std=%.2f' % (legend_list[n_rnn], np.std(w_rec[n_rnn])))
    plt.legend(leg2)
    plt.title('Recurrent W')
    
    plt.subplot(412)
    leg2 = []
    for n_rnn in range(num_rnn):
        plt.hist(w_in[n_rnn], density=density, alpha=alpha)
        leg2.append('%s; std=%.2f' % (legend_list[n_rnn], np.std(w_in[n_rnn])))
    plt.legend(leg2)
    plt.title('Input W')
    
    plt.subplot(413)
    leg2 = []
    for n_rnn in range(num_rnn):
        plt.hist(w_out_ctx[n_rnn], density=density, alpha=alpha)
        leg2.append('%s; std=%.2f' % (legend_list[n_rnn], np.std(w_out_ctx[n_rnn])))
    plt.legend(leg2)
    plt.title('Output W ctx')
        
    plt.subplot(414)
    leg2 = []
    for n_rnn in range(num_rnn):
        plt.hist(w_out_freq[n_rnn], density=density, alpha=alpha)
        leg2.append('%s; std=%.2f' % (legend_list[n_rnn], np.std(w_out_freq[n_rnn])))
    plt.legend(leg2)
    plt.title('Output W freq')
    
    # bias
    plt.figure()
    plt.subplot(411)
    leg2 = []
    for n_rnn in range(num_rnn):
        plt.hist(b_rec[n_rnn], density=density, alpha=alpha)
        leg2.append('%s; std=%.2f' % (legend_list[n_rnn], np.std(b_rec[n_rnn])))
    plt.legend(leg2)
    plt.title('Recurrent bias')
    
    plt.subplot(412)
    leg2 = []
    for n_rnn in range(num_rnn):
        plt.hist(b_in[n_rnn], density=density, alpha=alpha)
        leg2.append('%s; std=%.2f' % (legend_list[n_rnn], np.std(b_in[n_rnn])))
    plt.legend(leg2)
    plt.title('Input bias')
    
    plt.subplot(413)
    leg2 = []
    for n_rnn in range(num_rnn):
        plt.hist(b_out_ctx[n_rnn], density=density, alpha=alpha)
        leg2.append('%s; std=%.2f' % (legend_list[n_rnn], np.std(b_out_ctx[n_rnn])))
    plt.legend(leg2)
    plt.title('Output bias ctx')
    
    plt.subplot(414)
    leg2 = []
    for n_rnn in range(num_rnn):
        plt.hist(b_out_freq[n_rnn], density=density, alpha=alpha)
        leg2.append('%s; std=%.2f' % (legend_list[n_rnn], np.std(b_out_freq[n_rnn])))
    plt.legend(leg2)
    plt.title('Output bias freq')

#%%
def f_plot_mmn_dist(trials_oddball_ctx_cut, rates4d_cut, trials_test_cont_cut, rates_cont_freq4d_cut, plot_t1, red_dd_seq, title_tag='', baseline_subtract=True):
    
    trial_len, num_trials, num_runs, num_cells = rates4d_cut.shape
    freqs_all = np.unique(red_dd_seq)
    num_freqs = freqs_all.shape[0]
    
    trial_ave_rdc = f_get_rdc_trav(trials_oddball_ctx_cut, rates4d_cut, trials_test_cont_cut, rates_cont_freq4d_cut, plot_t1, red_dd_seq, baseline_subtract=baseline_subtract)
    
    mmn_dist = np.zeros((trial_len, 3, num_freqs))
    
    base_rdc = np.mean(trial_ave_rdc[:5,:,:,:], axis=0)
    base_rdc2 = np.mean(base_rdc, axis=1)
    
    if baseline_subtract:
        title_tag2 = '; base sub'
    else:
        title_tag2 = ''
    
    for n_ctx in range(3):
        for n_freq in range(num_freqs):
            resp1 = trial_ave_rdc[:,n_ctx,n_freq,:]
            
            base1 = base_rdc[n_ctx,n_freq,:]
            dist1 = pdist(np.vstack((base1, resp1)), 'euclidean')
            mmn_dist[:,n_ctx,n_freq] = dist1[:trial_len]

    # colors_ctx = ['blue', 'red', 'black']
    # plt.figure()
    # for n_ctx in range(3):
    #     plt.plot(mmn_dist[:,n_ctx,:], color=colors_ctx[n_ctx])
    # plt.title(title_tag)

    mmn_mean = np.mean(mmn_dist,axis=2)
    mmn_sem = np.std(mmn_dist,axis=2)/np.sqrt(num_freqs-1)

    colors_ctx = ['blue', 'red', 'black']
    plt.figure()
    for n_ctx in range(3):
        plt.plot(plot_t1, mmn_mean[:,n_ctx], color=colors_ctx[n_ctx])
        plt.fill_between(plot_t1, mmn_mean[:,n_ctx]-mmn_sem[:,n_ctx], mmn_mean[:,n_ctx]+mmn_sem[:,n_ctx], color=colors_ctx[n_ctx], alpha=0.2)
    plt.title('%s%s' %(title_tag, title_tag2))

#%%
def f_plot_mmn(rates4d_cut, trials_oddball_ctx_cut, plot_t1, params, title_tag):
    
    trial_len, num_tr, num_batch, num_cells = rates4d_cut.shape

    if params['num_ctx'] == 1:
        ctx_pad1 = 0
    elif params['num_ctx'] == 2:
        ctx_pad1 = 1
    
    trial_ave_ctx = np.zeros((trial_len, 2, num_batch, num_cells))
    for n_run in range(num_batch):
        for n_ctx in range(2):
            idx1 = trials_oddball_ctx_cut[:,n_run] == n_ctx+ctx_pad1
            trial_ave_ctx[:, n_ctx, n_run,:] = np.mean(rates4d_cut[:,idx1,n_run,:], axis=1)
    
    trial_ave_ctxn = trial_ave_ctx - np.mean(trial_ave_ctx[:5,:,:,:], axis=0)
    
    n_run = 2
    plt.figure(); 
    for n_run in range(num_batch):
        plt.plot(plot_t1, np.mean(trial_ave_ctxn[:,0,n_run,:], axis=1), 'b')
        plt.plot(plot_t1, np.mean(trial_ave_ctxn[:,1,n_run,:], axis=1), 'r')
    plt.title(title_tag)

#%%
def f_plot_mmn2(trials_oddball_ctx_cut, rates4d_cut, trials_test_cont_cut, rates_cont_freq4d_cut, plot_t1, red_dd_seq, baseline_subtract=True, split_pos_cells=True, title_tag=''):

    trial_len, num_trials, num_runs, num_cells = rates4d_cut.shape
    freqs_all = np.unique(red_dd_seq)
    num_freqs = freqs_all.shape[0]
    
    trial_ave_rdc = f_get_rdc_trav(trials_oddball_ctx_cut, rates4d_cut, trials_test_cont_cut, rates_cont_freq4d_cut, plot_t1, red_dd_seq, baseline_subtract=baseline_subtract)
    
    tr_ave_rdc_gr = []
    title_tag_all = []
    
    if baseline_subtract:
        title_tag2 = '; base sub'
    else:
        title_tag2 = ''
    
    if split_pos_cells:
        tr_ave2 = np.mean(np.mean(trial_ave_rdc, axis=1),axis=1)
        base1 = np.mean(tr_ave2[:5,:], axis=0)
        pos_cells = np.mean(tr_ave2[10:15,:], axis=0) > base1
        neg_cells = ~pos_cells

        tr_ave_rdc_gr.append(trial_ave_rdc[:,:,:,pos_cells])
        title_tag_all.append('positive cells')
        
        tr_ave_rdc_gr.append(trial_ave_rdc[:,:,:,neg_cells])
        title_tag_all.append('negative cells')
    else:
        tr_ave_rdc_gr.append(trial_ave_rdc)
        title_tag_all.append('all cells')    
    
    
    for n_gr in range(len(tr_ave_rdc_gr)):
        tr_ave3 = tr_ave_rdc_gr[n_gr]
        
        num_cells = tr_ave3.shape[3]
        
        tr_ave4 = np.reshape(tr_ave3, (trial_len, 3, num_freqs*num_cells), order='F')
        
        mmn_mean = np.mean(tr_ave4, axis=2)
        mmn_sem = np.std(tr_ave4, axis=2)/np.sqrt(num_freqs*num_cells-1)
        
        colors_ctx = ['blue', 'red', 'black']
        plt.figure()
        for n_ctx in range(3):
            plt.plot(plot_t1, mmn_mean[:,n_ctx], color=colors_ctx[n_ctx])
            plt.fill_between(plot_t1, mmn_mean[:,n_ctx]-mmn_sem[:,n_ctx], mmn_mean[:,n_ctx]+mmn_sem[:,n_ctx], color=colors_ctx[n_ctx], alpha=0.2)
        plt.title('%s; %s%s' % (title_tag, title_tag_all[n_gr], title_tag2))
        

    
#%%

def f_plot_mmn_freq(trials_oddball_ctx_cut, rates4d_cut, trials_test_cont_cut, rates_cont_freq4d_cut, plot_t1, red_dd_seq, baseline_subtract=True, title_tag=''):

    trial_len, num_trials, num_runs, num_cells = rates4d_cut.shape
    freqs_all = np.unique(red_dd_seq)
    num_freqs = freqs_all.shape[0]
    
    trial_ave_rdc = f_get_rdc_trav(trials_oddball_ctx_cut, rates4d_cut, trials_test_cont_cut, rates_cont_freq4d_cut, plot_t1, red_dd_seq, baseline_subtract=baseline_subtract)
    
    pairs = [[1, 0], [1, 2], [2, 0]]
    labels2 = ['red', 'dev', 'cont']
    
    if baseline_subtract:
        title_tag2 = '; base sub'
    else:
        title_tag2 = ''
    
    dist_ave = np.zeros((len(pairs), num_freqs))

    
    pair_titles = []

    for n_pair in range(len(pairs)):
        dist1 = np.zeros((num_freqs, num_freqs))
        
        tr1 = pairs[n_pair][0]
        tr2 = pairs[n_pair][1]
        
        for n_freq1 in range(num_freqs):
            for n_freq2 in range(num_freqs):
                trav1 = np.mean(trial_ave_rdc[10:15,tr1,n_freq1,:], axis=0)
                trav2 = np.mean(trial_ave_rdc[10:15,tr2,n_freq2,:], axis=0)
                dist1[n_freq1, n_freq2] = pdist((trav1, trav2))[0]
        
        dist2 = (np.triu(dist1) + np.tril(dist1).T)/2
        
        for n_freq in range(num_freqs):
            dist_ave[n_pair, n_freq] = np.mean(np.diagonal(dist2, offset=n_freq))
        
        pair_titles.append('%s - %s' % (labels2[tr1], labels2[tr2]))
        
        plt.figure()
        plt.imshow(dist1)
        plt.colorbar()
        plt.xlabel(labels2[tr1])
        plt.ylabel(labels2[tr2])
        plt.title('%s - %s distances; %s%s' % (labels2[tr1], labels2[tr2], title_tag, title_tag2))
        
    plt.figure()
    for n_pair in range(len(pairs)):
        plt.plot(dist_ave[n_pair,:])
    plt.legend(pair_titles)
    plt.xlabel('Freq difference')
    plt.ylabel('distance')
    plt.title('mean distances; %s%s' % (title_tag, title_tag2))


#%%

def f_plot_run_dist(rates_in, plot_runs=20, plot_trials=100, zero_trials=10, stim_ave_win=[], run_labels = [], run_colors = 50, ymax = [], title_tag=''):
    
    trial_len, num_trials, num_runs, num_cells = rates_in.shape
    
    if not len(run_labels):
        run_labels = np.arange(num_runs)
    
    if type(run_colors) is int:
        run_colors = cm.jet(np.linspace(0,1,run_colors+1))
    else:
        if not len(run_colors):
            run_colors = cm.jet(np.linspace(0,1,np.max(run_labels)+1))
        
    if len(stim_ave_win):    
        rates_ave1 = np.mean(rates_in[stim_ave_win,:,:,:], axis=0)
    else:
        rates_ave1 = np.mean(rates_in, axis=0)
    
    zero_trials2 = np.min((num_trials, zero_trials)).astype(int)
    plot_trials2 = np.min((num_trials, plot_trials)).astype(int)
    plot_runs2 = np.min((num_runs, plot_runs)).astype(int)
    
    start_loc = np.mean(np.mean(rates_ave1[:zero_trials2,:,:], axis=0), axis=0)
    dist_all = np.zeros((plot_trials2, plot_runs2))
    for n_run in range(plot_runs2):
        dist_all[:,n_run] = np.sqrt(np.sum((rates_ave1[:plot_trials2,n_run,:] - start_loc)**2, axis=1))
        #dist1 = pdist(np.vstack((start_loc, rates3d_cut[:,0,:])))
    
    plt.figure()
    for n_run in range(plot_runs2):
        if run_labels[n_run]:
            color1 = run_colors[run_labels[n_run],:]
        else:
            color1 = 'grey'
        plt.plot(dist_all[:,n_run], color=color1)
    plt.title(title_tag)
    plt.ylabel('euclidean distance')
    plt.xlabel('trials')
    if type(ymax) is int:
        plt.ylim(top=ymax)

#%%
def f_plot_shadederrorbar(x, y, alpha=0.2, legend=[], color=[]):
    
    if type(y) is list:
        y1 = y
        color1 = color
    else:
        y1 = [y]
        color1 = [color]
    line_all = []
    for n_pl in range(len(y1)):
        y2 = y1[n_pl]
        mean1 = np.mean(y2, axis=1)
        std1 = np.std(y2, axis=1)
        if len(color):
            l1 = plt.plot(x, mean1, color=color1[n_pl])
        else:
            l1 = plt.plot(x, mean1)
        plt.fill_between(x, mean1-std1, mean1+std1, color=l1[0].get_color(), alpha=alpha)
        line_all.append(l1[0])
    if len(legend):
        plt.legend(line_all, legend)