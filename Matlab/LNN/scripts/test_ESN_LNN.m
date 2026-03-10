%% Example: Memory Capacity with ESN_reservoir + LNN
% Demonstrates the unified ESN_reservoir class with LNN (Liquid Time-Constant
% network) under different spectral radius conditions.
%
% Uses composition: ESN_reservoir wraps LNN.
% Network params go to LNN(), ESN params go to ESN_reservoir().
%
% See also: ESN_reservoir, LNN, ESNStimulus

clear; clc; close all;

%% Add paths
setup_paths();

%% Common parameters
n = 300;                      % Number of neurons
rng_seed = 41;                % Network seed
fs = 200;                     % Sampling frequency (Hz)

% MC protocol (seconds -> samples)
T_wash_sec = 10;
T_train_sec = 25;
T_test_sec = 25;

T_wash = T_wash_sec * fs;
T_train = T_train_sec * fs;
T_test = T_test_sec * fs;
d_max = 2 * fs;               % Max delay = 2 seconds

%% Network params (shared — go to LNN)
net_args = { ...
    'n', n, ...
    'n_in', 1, ...
    'fs', fs, ...
    'rng_seeds', [rng_seed, rng_seed], ...
    'activation_name', 'tanh', ...
    'tau_init', 1.0};

%% ESN params (shared — go to ESN_reservoir)
esn_args = { ...
    'T_wash', T_wash, ...
    'T_train', T_train, ...
    'T_test', T_test, ...
    'd_max', d_max, ...
    'input_type', 'white', ...
    'sigma_in', 0.25, ...
    'f_in', 1.0, ...
    'u_scale', 2};

%% Conditions: vary spectral radius
condition_names = {'R = 0.5', 'R = 1.0', 'R = 1.5'};
condition_overrides = { ...
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

    model = LNN(net_args{:}, condition_overrides{i}{:});
    model.lya_method = 'benettin';
    esn{i} = ESN_reservoir(model, esn_args{:});
    esn{i}.build();
end

%% Run all conditions
MC = zeros(1, n_cond);
R2 = cell(1, n_cond);
for i = 1:n_cond
    fprintf('\n==============================\n');
    fprintf('CONDITION %d: %s\n', i, condition_names{i});
    fprintf('==============================\n');
    [MC(i), R2{i}] = esn{i}.run_memory_capacity();
end

%% Summary
fprintf('\n==============================\n');
fprintf('SUMMARY\n');
fprintf('==============================\n');
for i = 1:n_cond
    fprintf('  %s: MC = %.2f\n', condition_names{i}, MC(i));
end

%% Comparison Plot
colors = [0.3, 0.8, 0.3;     % Green
          0.3, 0.6, 0.9;     % Blue
          0.9, 0.4, 0.3];    % Red

figure('Name', 'LNN MC Comparison');

% R² vs delay
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

% Cumulative MC
subplot(1, 3, 2);
hold on;
for i = 1:n_cond
    plot(1:d_max, cumsum(R2{i}), '-', 'Color', colors(i,:), ...
        'LineWidth', 2, 'DisplayName', condition_names{i});
end
xlabel('Delay d (samples)');
ylabel('Cumulative MC');
title('Cumulative Memory Capacity');
legend('Location', 'southeast');
grid on;

% Total MC bar chart
subplot(1, 3, 3);
b = bar(1:n_cond, MC);
b.FaceColor = 'flat';
for i = 1:n_cond
    b.CData(i, :) = colors(i, :);
end
set(gca, 'XTickLabel', condition_names);
ylabel('Total MC');
title('Total MC Comparison');
grid on;
for i = 1:n_cond
    text(i, MC(i) + 0.5, sprintf('%.1f', MC(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

sgtitle('LNN Memory Capacity: Effect of Spectral Radius');

%% Time Series Plots
delays_to_plot = [1, 50, 100, 200, 400];
for i = 1:n_cond
    esn{i}.plot_esn_timeseries(delays_to_plot, 'title', condition_names{i});
end

%% Model-specific plots (passthrough)
fprintf('\nGenerating model-specific plots for condition 2...\n');
esn{2}.model.plot();

fprintf('\nDone.\n');
