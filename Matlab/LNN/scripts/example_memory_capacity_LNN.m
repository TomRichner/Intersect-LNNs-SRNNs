%% Example: Memory Capacity Measurement with LNN_ESN_reservoir
% Demonstrates how to use LNN_ESN_reservoir to measure memory capacity
% of a Liquid Time-Constant (LTC) reservoir under different spectral
% radius conditions.
%
% Compares: level_of_chaos = {0.5, 1.0, 1.5}
%
% Protocol:
%   1. Drive LTC reservoir with scalar random input u(t)
%   2. Train linear readouts for each delay d to reconstruct u(t-d)
%   3. Compute R^2_d and sum to get total memory capacity
%
% See also: LNN_ESN_reservoir, LNN

clear; clc; close all;

%% Add paths
setup_paths();

%% Common parameters
n = 100;                        % Number of neurons
rng_seed = 42;                  % Network seed

% Sampling frequency
fs = 200;

% MC protocol (seconds -> samples)
T_wash_sec = 10;
T_train_sec = 25;
T_test_sec = 25;

T_wash = T_wash_sec * fs;
T_train = T_train_sec * fs;
T_test = T_test_sec * fs;
d_max = 2 * fs;                 % Max delay = 2 seconds worth of samples

% Input type
input_type = 'white';

%% Base ESN configuration (shared across all conditions)
base_args = { ...
    'n', n, ...
    'n_in', 1, ...
    'fs', fs, ...
    'rng_seed', rng_seed, ...
    'activation', 'tanh', ...
    'tau_init', 1.0, ...
    'sigma_in', 0.25, ...           % Input weight scale (strong drive)
    'f_in', 1.0, ...               % All neurons receive input
    'u_scale', 2, ...              % Input amplitude
    'input_type', input_type, ...
    'T_wash', T_wash, ...
    'T_train', T_train, ...
    'T_test', T_test, ...
    'd_max', d_max};

%% Condition-specific overrides (vary spectral radius)
condition_names = {'R = 0.5', 'R = 1.0', 'R = 1.5'};
condition_args = { ...
    {'level_of_chaos', 0.5}, ...
    {'level_of_chaos', 1.0}, ...
    {'level_of_chaos', 1.5}};
n_cond = numel(condition_names);

%% Build all conditions
esn = cell(1, n_cond);
for i = 1:n_cond
    fprintf('\n==============================\n');
    fprintf('Building %s...\n', condition_names{i});
    fprintf('==============================\n');
    esn{i} = LNN_ESN_reservoir(base_args{:}, condition_args{i}{:});
    esn{i}.build();
end

%% Verify shared structure
LNN_ESN_reservoir.verify_shared_build(esn, ...
    {'level_of_chaos', 'W_r'}, ...
    {'W_in_esn', 'u_scalar', 'u_ex', 't_ex'});

%% Run all conditions
MC = zeros(1, n_cond);
R2 = cell(1, n_cond);
results = cell(1, n_cond);
for i = 1:n_cond
    fprintf('\n==============================\n');
    fprintf('CONDITION %d: %s\n', i, condition_names{i});
    fprintf('==============================\n');
    [MC(i), R2{i}, results{i}] = esn{i}.run_memory_capacity();
end

%% Summary
fprintf('\n==============================\n');
fprintf('SUMMARY\n');
fprintf('==============================\n');
fprintf('Memory Capacity Results:\n');
for i = 1:n_cond
    fprintf('  %s: MC = %.2f\n', condition_names{i}, MC(i));
end

%% Comparison Plot
colors = [0.3, 0.7, 0.3;   % Green for R=0.5
    0.3, 0.6, 0.9;   % Blue for R=1.0
    0.9, 0.4, 0.3];  % Red for R=1.5

figure();

% Plot 1: R^2 vs delay
subplot(1, 3, 1);
hold on;
bar_data = zeros(d_max, n_cond);
for i = 1:n_cond
    bar_data(:, i) = R2{i}(:);
end
b = bar(1:d_max, bar_data);
for i = 1:n_cond
    b(i).FaceColor = colors(i, :);
end
xlabel('Delay d (samples)');
ylabel('R^2_d');
title('Memory Capacity by Delay');
legend(condition_names, 'Location', 'northeast');
grid on;
hold off;

% Plot 2: Cumulative MC
subplot(1, 3, 2);
hold on;
line_styles = {'-', '-', '-'};
for i = 1:n_cond
    plot(1:d_max, cumsum(R2{i}), line_styles{i}, 'Color', colors(i,:), ...
        'LineWidth', 2, 'DisplayName', condition_names{i});
end
xlabel('Delay d (samples)');
ylabel('Cumulative MC');
title('Cumulative Memory Capacity');
legend('Location', 'southeast');
grid on;
hold off;

% Plot 3: Bar chart of total MC
subplot(1, 3, 3);
b = bar(1:n_cond, MC);
b.FaceColor = 'flat';
for i = 1:n_cond
    b.CData(i, :) = colors(i, :);
end
set(gca, 'XTickLabel', condition_names);
ylabel('Total Memory Capacity');
title('Total MC Comparison');
grid on;

for i = 1:n_cond
    text(i, MC(i) + 0.5, sprintf('%.1f', MC(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

sgtitle('LNN Memory Capacity: Effect of Spectral Radius');

%% Time Series Plots for Each Condition
delays_to_plot = [1, 10, 50, 100, 200, 400];

fprintf('\nGenerating time series plots...\n');
for i = 1:n_cond
    esn{i}.plot_esn_timeseries(delays_to_plot, 'title', condition_names{i});
end
fprintf('Time series plots generated.\n');

%% Save results
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
output_dir = fullfile(project_root, 'data', 'memory_capacity');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

results_all = struct();
results_all.conditions = {condition_names};
results_all.MC = MC;
results_all.R2 = {R2};
for i = 1:n_cond
    results_all.(sprintf('cond%d', i)) = results{i};
end

save_file = fullfile(output_dir, 'memory_capacity_LNN_results.mat');
save(save_file, 'results_all');
fprintf('\nResults saved to %s\n', save_file);
