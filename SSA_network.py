# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:33:38 2024

@author: ys2605
"""

import numpy as np

import matplotlib.pyplot as plt

#%%

ring_net = 0


P = 21                      # number of columns
NE = 100                    # excitatory cells in each column
NI = 100                    # inhibitory cells in each column


# Controlling connection strengths:
factor   = 1                # multiplies all intra-column connections (controls spontaneous population spikes)
factor_1 = 1                # multiplies nearest-neighbor inter-column connections (aids spread of population spikes)
factor_2 = 1                # multiplies 2nd-nearest-neighbor inter-column connections (aids spread of population spikes)


# Background input drawn from a uniform distribution (neurons are indexed by input strength):
bg_low_E  = -9.9            # % lowest background input (in Hz) to excitatory population
bg_high_E = 9.9             # highest background input (in Hz) to excitatory population
bg_low_I  = bg_low_E        # lowest background input (in Hz) to inhibitory population
bg_high_I = bg_high_E       # highest background input (in Hz) to inhibitory population
# In Loebel & Tsodyks 2002, the above values were reached by requiring a spontaneous activity of a few Hz. 

tau_E     = 0.001           # excitatory neurons' time constant (in seconds)
tau_I     = 0.001           # inhibitory neurons' time constant (in seconds) 
tau_ref_E = 0.003           # tau refractory of excitatory neurons (in seconds)
tau_ref_I = 0.003           # tau refractory of inhibitory neurons (in seconds)
tau_rec   = 0.800           # recovery time constant of intracortical synapses (in seconds)
tau_rec_s = 0.300           # recovery time constant of sensory input synapses (in seconds)


U   = 0.5                   # Portion of available fraction of resources that is utilized in response to an action potential
U_s = 0.7                   # Same as U, only for the thalamo-cotical synapses that convey the sensory input

# Connection strengths:
Jee = 6*factor/NE           # exc2exc
Jei = -4*factor/NI          # inh2exc
Jie = 0.5*factor/NE         # exc2inh
Jii = -0.5*factor/NI        # inh2inh


Jee_1 = 0.045*factor_1/NE   # exc2exc, between neighboring columns
Jie_1 = 0.0035*factor_1/NE  # exc2inh, between neighboring columns
Jee_2 = 0.015*factor_2/NE   # exc2exc, from one column to its 2nd neighbor
Jie_2 = 0.0015*factor_2/NE  # exc2inh, from one column to its 2nd neighbor

# Simulation conditions:
t_eq      = 5               # The time given to reach equilibrium (in seconds). It is important to allow enough time, or else the response to the first stimulus is distorted.
dt        = 0.001           #%0.0001; % Time-step (in seconds)
post_stim = 0.100           # This is defined here so the simulations keeps runnng for 2*post_stim after the last stimulus offset


#%% Initialize

e_step = (bg_high_E - bg_low_E)/(NE - 1)        # *) How about input from a non-uniform (e.g. Gaussian) distribution?
i_step = (bg_high_I - bg_low_I)/(NI - 1)
Inp_E  = np.ones((P,1))*np.arange(bg_low_E, bg_high_E+e_step, e_step)  # These are the inputs to all the excitatory neurons, hence the P columns and NE rows
Inp_I  = np.ones((P,1))*np.arange(bg_low_I, bg_high_I+e_step, i_step)  # These are the inputs to all the inhibitory neurons

tmax         = t_eq                             # + t_prot*stim; % Maximum time that the simulation will reach (in seconds) 
num_steps_eq = np.floor(tmax/dt).astype(int)    # Total number of steps in the simulation


E = np.zeros((P,NE))        # Acitivity of all excitatory neurons (in Hz)
I = np.zeros((P,NI))        # Activity of all inhibitory neurons (in Hz)
x = np.zeros((P,NE))        # The fractions of resources available for synaptic transmission in all excitatory neurons
y = np.zeros((P,NI))        # Same as x, only for inhibitory neurons
z = 0                       # Only one synapse is needed here; zeros(P,NE,M_aug); % Same as x, only for the thalamo-cortical synapses


EUx = np.zeros((P,1))


E_act  = np.zeros((P,NE,num_steps_eq))      # Activity of all excitatory neurons during the time allowed for reaching equilibrium, for identification of the active neurons.
E_mean = np.zeros((P,num_steps_eq))         # Mean excitatory neuron activities in all columns, in all time-steps (in Hz)
I_mean = np.zeros((P,num_steps_eq))         # Mean inhibitory neuron activities in all columns, in all time-steps (in Hz)


#%%


for i in range(num_steps_eq):

    EUx = np.diagonal(np.dot(E,(U*x).T))[:,None]        # Intra-column input from excitatory synapses, taking synaptic depression into account
    EUx_1 = np.vstack((EUx[1:], ring_net*EUx[:1])) + np.vstack((ring_net*EUx[-1:], EUx[:-1]))          # Excitatory input from neighboring column
    EUx_2 = np.vstack((EUx[2:], ring_net*EUx[:2])) + np.vstack((ring_net*EUx[-2:], EUx[:-2]))      # Excitatory input from the neighboring column once removed
    
    
    
    IUy = np.diagonal(np.dot(I,(U*y).T))  [:,None]       # Intra-column input from inhibitory synapses, taking synaptic depression into account
    
    
    # Pre-calculation for the inter-column exc2inh gain:
    E_sum = np.sum(E, axis=1)[:,None]           # Total intra-column input in each column
    E_sum_1 = np.vstack((E_sum[1:], ring_net*E_sum[:1])) + np.vstack((ring_net*E_sum[-1], E_sum[:-1]))      # Total input to each column from its nearest neighbors
    E_sum_2 = np.vstack((E_sum[2:], ring_net*E_sum[:2])) + np.vstack((ring_net*E_sum[-2:], E_sum[:-2]))     # Same, for 2nd-nearest neighbors.
    
    
    I_sum = np.sum(I, axis=1)[:,None]
    
    
    Gain_E = Inp_E + (Jee*EUx + Jei*IUy + Jee_1*EUx_1 + Jee_2*EUx_2)
    Gain_I = Inp_I + (Jie*E_sum + Jii*I_sum + Jie_1*E_sum_1 + Jie_2*E_sum_2)
    
    
    # Implementing the non-linearity:
    Gain_E[Gain_E < 0]   = 0;
    Gain_E[Gain_E > 300] = 300;
    Gain_I[Gain_I < 0]   = 0;
    Gain_I[Gain_I > 300] = 300;
    
    
    # The variables' dynamics:
    E = E + (dt/tau_E)*(-E + Gain_E*(1 - tau_ref_E*E))
    x = x + dt*((1 - x)/tau_rec - U*E*x)
    I = I + (dt/tau_I)*(-I + Gain_I*(1 - tau_ref_I*I))
    y = y + dt*((1 - y)/tau_rec - U*I*y);
    z = z + dt*((1 - z)/tau_rec_s)
    
    
    E_act[:,:,i]  = E                       # Tracking the activity of all neurons
    E_mean[:,i]   = np.mean(E, axis=1)      # Tracking mean excitatory activity
    I_mean[:,i]   = np.mean(I, axis=1)      # Tracking mean inhibitory activity



#%%

plt.figure()
plt.plot(E_mean.T)



plt.figure()
plt.plot(I_mean.T)




