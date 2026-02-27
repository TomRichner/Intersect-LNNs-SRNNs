% test_LNN.m - Run LNN with defaults and plot
%
% Constructs an LNN with n=50, n_in=2 (circular trajectory input),
% all other parameters at defaults (activation='tanh', tau_init=1.0).

close all; clear; clc;

% Add paths
setup_paths();

%% Create model
model = LNN('n', 50, 'n_in', 2);

%% Build, run, and plot
model.build();
model.run();
model.plot();
