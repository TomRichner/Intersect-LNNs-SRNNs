% compare_solvers_lyapunov.m
%
% Compare ODE (ode45) and fused semi-implicit Euler solvers for SRNNModel2.
% Uses Benettin's algorithm to compute the largest Lyapunov exponent (LLE)
% for each solver, then overlays trajectories and Lyapunov time series.
%
% The fused solver treats linear decay terms implicitly (Hasani et al. 2021).
% Both solvers share identical W, W_in, and stimulus (same rng_seeds).

close all; clear; clc;
setup_paths();

%% ========================================================================
%  Configuration
%  ========================================================================
n = 300;
T_total = 50;                       % Simulation length (s)
substep_values = [4, 6, 10];       % Fused sub-step counts to compare

% Common model arguments
common_args = { ...
    'n', n, ...
    'n_a_E', 3, ...
    'n_b_E', 1, ...
    'T_range', [0, T_total], ...
    'lya_method', 'benettin', ...
    'store_full_state', true, ...
    'store_decimated_state', true ...
};

%% ========================================================================
%  Build comparison object
%  ========================================================================
comp = SRNNComparison();

% ODE reference
comp.add(SRNNModel2(common_args{:}, 'solver_mode', 'ode'), 'ode45');

% Fused at multiple substep counts
for i = 1:length(substep_values)
    s = substep_values(i);
    comp.add(SRNNModel2(common_args{:}, ...
        'solver_mode', 'fused', ...
        'fused_substeps', s), ...
        sprintf('fused(%d)', s));
end

%% ========================================================================
%  Build and run all models
%  ========================================================================
comp.build_all();
comp.run_all();

%% ========================================================================
%  Results
%  ========================================================================
comp.summary();
comp.param_diff();

%% ========================================================================
%  Plots
%  ========================================================================

% Collect all figure handles and names for saving
all_figs = [];
all_names = {};

% Individual time series for each model
figs_ts = comp.plot_all();
all_figs = [all_figs; figs_ts(:)];
for i = 1:comp.n_models
    all_names{end+1} = sprintf('tseries_%s', comp.labels{i}); %#ok<SAGROW>
end

% Trajectory overlay (first 5 neurons)
[fig_tx, ~] = comp.compare_traces(1:5);
all_figs(end+1) = fig_tx;
all_names{end+1} = 'traces_x';

[fig_tr, ~] = comp.compare_traces(1:5, 'r');
all_figs(end+1) = fig_tr;
all_names{end+1} = 'traces_r';

% Lyapunov comparison
[fig_lya, ~] = comp.compare_lyapunov();
all_figs(end+1) = fig_lya;
all_names{end+1} = 'lyapunov_comparison';

% LLE bar chart
[fig_lle, ~] = comp.compare_LLE();
all_figs(end+1) = fig_lle;
all_names{end+1} = 'LLE_comparison';

fprintf('\nPlots generated.\n');

%% ========================================================================
%  Save figures
%  ========================================================================
out_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'figs', 'comparisons');
comp.save_figures(all_figs, all_names, out_dir);
