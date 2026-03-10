%% Example: Memory Capacity with ESN_reservoir + SRNNModel2
% Demonstrates the unified ESN_reservoir class with SRNN under different
% adaptation conditions (baseline, SFA, SFA+STD).
%
% Uses composition: ESN_reservoir wraps SRNNModel2.
% Network params go to SRNNModel2(), ESN params go to ESN_reservoir().
%
% See also: ESN_reservoir, SRNNModel2, ESNStimulus

clear; clc; close all;

%% Add paths
setup_paths();

%% Common parameters
n = 300;                      % Number of neurons
level_of_chaos = 1.0;         % Moderate chaos level
rng_seed_net = 41;            % Network seed
rng_seed_stim = 43;           % Stimulus seed
fs = 200;                     % Sampling frequency (Hz)

% MC protocol (seconds -> samples)
T_wash_sec = 10;
T_train_sec = 25;
T_test_sec = 25;

T_wash = T_wash_sec * fs;
T_train = T_train_sec * fs;
T_test = T_test_sec * fs;
d_max = 2 * fs;               % Max delay = 2 seconds

%% Network params (shared across conditions — go to SRNNModel2)
net_args = { ...
    'n', n, ...
    'fs', fs, ...
    'level_of_chaos', level_of_chaos, ...
    'rng_seeds', [rng_seed_net, rng_seed_stim], ...
    'tau_d', 0.1, ...
    'S_c', 0.4, ...
    'S_a', 0.9, ...
    'n_a_I', 0, ...
    'n_b_I', 0, ...
    'c_E', 0.15/3, ...
    'tau_a_E', [0.1, 1.0, 10], ...
    'tau_b_E_rec', 1.0, ...
    'tau_b_E_rel', 0.25};

%% ESN params (shared across conditions — go to ESN_reservoir)
esn_args = { ...
    'T_wash', T_wash, ...
    'T_train', T_train, ...
    'T_test', T_test, ...
    'd_max', d_max, ...
    'input_type', 'white'};

%% Conditions: vary adaptation
condition_names = {'Baseline', 'SFA only', 'SFA + STD'};
condition_overrides = { ...
    {'n_a_E', 0, 'n_b_E', 0}, ...   % No adaptation
    {'n_a_E', 3, 'n_b_E', 0}, ...   % SFA: 3 timescales
    {'n_a_E', 3, 'n_b_E', 1}};      % SFA + STD
n_cond = numel(condition_names);

%% Build all conditions
esn = cell(1, n_cond);
for i = 1:n_cond
    fprintf('\n==============================\n');
    fprintf('Building %s...\n', condition_names{i});
    fprintf('==============================\n');

    model = SRNNModel2(net_args{:}, condition_overrides{i}{:});
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
colors = [0.7, 0.7, 0.7;     % Gray
          0.3, 0.6, 0.9;     % Blue
          0.9, 0.4, 0.3];    % Red

figure('Name', 'SRNN MC Comparison');

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

sgtitle('SRNN Memory Capacity: Effect of Adaptation');

%% Time Series Plots
delays_to_plot = [1, 50, 100, 200, 400];
for i = 1:n_cond
    esn{i}.plot_esn_timeseries(delays_to_plot, 'title', condition_names{i});
end

fprintf('\nDone.\n');
