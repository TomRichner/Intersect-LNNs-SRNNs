% export_matlab_data.m — Export SRNN and LNN simulation data for Julia cross-validation
%
% Runs both models with small n=20 and fixed seeds, saves all parameters,
% stimulus, trajectories, and initial conditions to .mat files that
% Julia can reload for numerical comparison.
%
% Output files:
%   CrossValidation/data/srnn_cross_val.mat
%   CrossValidation/data/ltc_cross_val.mat

close all; clear; clc;

%% ── Setup paths ────────────────────────────────────────────────────────
script_dir = fileparts(mfilename('fullpath'));
cross_val_dir = fileparts(script_dir);  % CrossValidation/
repo_dir = fileparts(cross_val_dir);    % Intersect-LNNs-SRNNs/
matlab_dir = fullfile(repo_dir, 'Matlab');

addpath(genpath(fullfile(matlab_dir, 'shared', 'src')));
addpath(genpath(fullfile(matlab_dir, 'SRNN', 'src')));
addpath(genpath(fullfile(matlab_dir, 'LNN', 'src')));
fprintf('Paths added.\n');

% Create data output directory
data_dir = fullfile(cross_val_dir, 'data');
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end

%% ════════════════════════════════════════════════════════════════════════
%  SRNN EXPORT
%  ════════════════════════════════════════════════════════════════════════
fprintf('\n=== SRNN Export ===\n');

srnn = SRNNModel2('n', 20, 'indegree', 10, ...
                   'n_a_E', 3, 'n_b_E', 1, ...
                   'T_range', [0, 5], ...
                   'rng_seeds', [42 42], ...
                   'lya_method', 'none', ...
                   'store_full_state', true);

srnn.build();
srnn.run();

% Extract params
p = srnn.get_params();

% Package for export
srnn_data = struct();
srnn_data.W       = p.W;
srnn_data.W_in    = p.W_in;
srnn_data.f       = p.f;
srnn_data.n       = p.n;
srnn_data.n_E     = p.n_E;
srnn_data.n_I     = p.n_I;
srnn_data.E_indices = p.E_indices;
srnn_data.I_indices = p.I_indices;
srnn_data.tau_d   = p.tau_d;
srnn_data.c_E     = p.c_E;
srnn_data.c_I     = p.c_I;
srnn_data.tau_a_E = p.tau_a_E;
srnn_data.n_a_E   = p.n_a_E;
srnn_data.n_a_I   = p.n_a_I;
srnn_data.n_b_E   = p.n_b_E;
srnn_data.n_b_I   = p.n_b_I;
srnn_data.tau_b_E_rec = p.tau_b_E_rec;
srnn_data.tau_b_E_rel = p.tau_b_E_rel;
srnn_data.tau_b_I_rec = p.tau_b_I_rec;
srnn_data.tau_b_I_rel = p.tau_b_I_rel;

% Trajectory
srnn_data.t_out     = srnn.t_out;
srnn_data.state_out = srnn.state_out;
srnn_data.S0        = srnn.S0;

% Stimulus (full time series)
srnn_data.t_ex = srnn.stimulus.t_ex;
srnn_data.u_ex = srnn.stimulus.u_ex;

% Activation parameters (piecewise sigmoid defaults)
srnn_data.activation_type = 'piecewise_sigmoid';
srnn_data.S_a = 0.9;
srnn_data.S_c = 0.35;

save(fullfile(data_dir, 'srnn_cross_val.mat'), '-struct', 'srnn_data', '-v7');
fprintf('Saved: %s\n', fullfile(data_dir, 'srnn_cross_val.mat'));
fprintf('  n=%d, state_dim=%d, T=[0, 5], %d time steps\n', ...
    p.n, p.N_sys_eqs, length(srnn.t_out));

%% ════════════════════════════════════════════════════════════════════════
%  LTC EXPORT
%  ════════════════════════════════════════════════════════════════════════
fprintf('\n=== LTC Export ===\n');

lnn = LNN('n', 20, 'n_in', 2, ...
           'T_range', [0, 5], ...
           'rng_seeds', [42 42], ...
           'lya_method', 'none');

lnn.build();
lnn.run();

% Extract params
p = lnn.get_params();

% Package for export
ltc_data = struct();
ltc_data.W     = p.W;
ltc_data.W_in  = p.W_in;
ltc_data.mu    = p.mu;
ltc_data.tau   = p.tau;
ltc_data.A     = p.A;
ltc_data.n     = p.n;
ltc_data.n_in  = p.n_in;

% Trajectory
ltc_data.t_out     = lnn.t_out;
ltc_data.state_out = lnn.state_out;
ltc_data.S0        = lnn.S0;

% Stimulus
ltc_data.t_ex = lnn.stimulus.t_ex;
ltc_data.u_ex = lnn.stimulus.u_ex;

% Activation
ltc_data.activation_type = 'tanh';

save(fullfile(data_dir, 'ltc_cross_val.mat'), '-struct', 'ltc_data', '-v7');
fprintf('Saved: %s\n', fullfile(data_dir, 'ltc_cross_val.mat'));
fprintf('  n=%d, n_in=%d, T=[0, 5], %d time steps\n', ...
    p.n, p.n_in, length(lnn.t_out));

%% ════════════════════════════════════════════════════════════════════════
%  LNN2 EXPORT (Hasani MATLAB-faithful)
%  ════════════════════════════════════════════════════════════════════════
fprintf('\n=== LNN2 Export ===\n');

lnn2 = LNN2('n', 20, 'n_in', 2, 'k', 20, ...
             'T_range', [0, 5], ...
             'rng_seeds', [42 42], ...
             'activation_name', 'tanh', ...
             'lya_method', 'none');

lnn2.build();
lnn2.run();

p2 = lnn2.get_params();

ltc2_data = struct();
ltc2_data.n            = p2.n;
ltc2_data.n_in         = p2.n_in;
ltc2_data.k            = p2.k;
ltc2_data.n_layers     = p2.n_layers;
ltc2_data.activation_name = p2.activation_name;
ltc2_data.W_ff         = p2.W_ff;
ltc2_data.b_ff         = p2.b_ff;
ltc2_data.E_ff         = p2.E_ff;
ltc2_data.W_rec        = p2.W_rec;
ltc2_data.b_rec        = p2.b_rec;
ltc2_data.E_rec        = p2.E_rec;
ltc2_data.tau          = p2.tau;

ltc2_data.t_out     = lnn2.t_out;
ltc2_data.state_out = lnn2.state_out;
ltc2_data.S0        = lnn2.S0;
ltc2_data.t_ex      = lnn2.stimulus.t_ex;
ltc2_data.u_ex      = lnn2.stimulus.u_ex;

save(fullfile(data_dir, 'ltc2_cross_val.mat'), '-struct', 'ltc2_data', '-v7');
fprintf('Saved: %s\n', fullfile(data_dir, 'ltc2_cross_val.mat'));
fprintf('  n=%d, k=%d, n_layers=%d, T=[0, 5], %d time steps\n', ...
    p2.n, p2.k, p2.n_layers, length(lnn2.t_out));

%% ════════════════════════════════════════════════════════════════════════
%  LNN1 EXPORT (Hasani Python-faithful)
%  ════════════════════════════════════════════════════════════════════════
fprintf('\n=== LNN1 Export ===\n');

lnn1 = LNN1('n', 20, 'n_in', 2, ...
             'T_range', [0, 5], ...
             'rng_seeds', [42 42], ...
             'lya_method', 'none');

lnn1.build();
lnn1.run();

p1 = lnn1.get_params();

ltc1_data = struct();
ltc1_data.n             = p1.n;
ltc1_data.n_in          = p1.n_in;
ltc1_data.W_syn         = p1.W_syn;
ltc1_data.mu_syn        = p1.mu_syn;
ltc1_data.sigma_syn     = p1.sigma_syn;
ltc1_data.erev          = p1.erev;
ltc1_data.sensory_W     = p1.sensory_W;
ltc1_data.sensory_mu    = p1.sensory_mu;
ltc1_data.sensory_sigma = p1.sensory_sigma;
ltc1_data.sensory_erev  = p1.sensory_erev;
ltc1_data.vleak         = p1.vleak;
ltc1_data.gleak         = p1.gleak;
ltc1_data.cm            = p1.cm;
ltc1_data.input_w       = p1.input_w;
ltc1_data.input_b       = p1.input_b;

ltc1_data.t_out     = lnn1.t_out;
ltc1_data.state_out = lnn1.state_out;
ltc1_data.S0        = lnn1.S0;
ltc1_data.t_ex      = lnn1.stimulus.t_ex;
ltc1_data.u_ex      = lnn1.stimulus.u_ex;

save(fullfile(data_dir, 'ltc1_cross_val.mat'), '-struct', 'ltc1_data', '-v7');
fprintf('Saved: %s\n', fullfile(data_dir, 'ltc1_cross_val.mat'));
fprintf('  n=%d, n_in=%d, T=[0, 5], %d time steps\n', ...
    p1.n, p1.n_in, length(lnn1.t_out));

fprintf('\n=== Export complete ===\n');
fprintf('Files saved to: %s\n', data_dir);

