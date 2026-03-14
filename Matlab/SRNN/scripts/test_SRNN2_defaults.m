% test_SRNN2_defaults.m - Run SRNNModel2 with all defaults and plot
%
% Constructs SRNNModel2 with n_a_E=3 (SFA) and n_b_E=1 (STD), all other
% parameters at defaults (n=300, T_range=[0,50], lya_method='benettin').

close all; clear; clc;

% Add paths
setup_paths();

%% Create model
% n = 100
% model = SRNNModel2('level_of_chaos',1.5,'f',0.58,'n',n, 'indegree',50,'n_a_E', 3, 'n_b_E', 1, 'c_0_E', 0/sqrt(n), 'rng_seeds', [123, 456]+3, 'c_E', 0.15/3);

model = SRNNModel2('n',500,'n_a_E', 3, 'n_b_E', 1, 'n_a_I', 3, 'n_b_I');

%% Build, run, and plot
model.build();
model.run();
model.plot();
