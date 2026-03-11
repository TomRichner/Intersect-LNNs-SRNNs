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

fprintf('\n=== Export complete ===\n');
fprintf('Files saved to: %s\n', data_dir);
