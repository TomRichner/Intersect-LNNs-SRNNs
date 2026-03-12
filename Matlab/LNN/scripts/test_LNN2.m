% test_LNN2.m - Run LNN2 (Hasani MATLAB-faithful) with defaults and plot
%
% Tests single-layer and multi-layer modes with 4-panel figure:
% Panel 1: External input I(t)
% Panel 2: Neuron states x(t)
% Panel 3: Effective time constant τ_sys(t)
% Panel 4: Feedforward activation f_ff(t)

close all; clear; clc;

% Add paths
setup_paths();

%% ─── Single Layer Test ─────────────────────────────────────────────────
fprintf('=== LNN2 Single-Layer Test ===\n');
model = LNN2('n', 50, 'n_in', 2, 'k', 50, ...
             'T_range', [0, 10], ...
             'rng_seeds', [42 42], ...
             'activation_name', 'tanh');

model.build();
model.run();

% Extract plot data
pd = model.plot_data;
n = model.n;
cmap = LNN2.default_colormap(n);

fig1 = figure('Name', 'LNN2 Single-Layer', 'Position', [50 50 1000 800]);

% Panel 1: Input
ax1 = subplot(4, 1, 1);
plot(pd.t, pd.u);
title('External Input I(t)');
ylabel('Amplitude');
legend({'sin(2\pit)', 'cos(2\pit)'}, 'Location', 'northeast');

% Panel 2: States
ax2 = subplot(4, 1, 2);
hold on;
for i = 1:n
    plot(pd.t, pd.x(:, i), 'Color', cmap(i, :), 'LineWidth', 0.5);
end
hold off;
title('Neuron States x(t)');
ylabel('State');

% Panel 3: tau_sys
ax3 = subplot(4, 1, 3);
hold on;
for i = 1:n
    plot(pd.t, pd.tau_sys(:, i), 'Color', cmap(i, :), 'LineWidth', 0.5);
end
hold off;
title('\tau_{sys}(t) = \tau / (1 + \tau(|f_{ff}| + |f_{rec}|))');
ylabel('\tau_{sys}');

% Panel 4: f_ff
ax4 = subplot(4, 1, 4);
hold on;
for i = 1:n
    plot(pd.t, pd.f_ff(:, i), 'Color', cmap(i, :), 'LineWidth', 0.5);
end
hold off;
title('Feedforward Activation f_{ff}(t)');
ylabel('f_{ff}');
xlabel('Time (s)');

sgtitle('LNN2 (Hasani MATLAB-faithful) — Single Layer, n=50');
linkaxes([ax1, ax2, ax3, ax4], 'x');

%% ─── Multi-Layer Test ──────────────────────────────────────────────────
fprintf('\n=== LNN2 Multi-Layer Test (2 layers × 25 neurons) ===\n');
model2 = LNN2('n', 50, 'n_in', 2, 'k', 25, ...
              'T_range', [0, 10], ...
              'rng_seeds', [42 42], ...
              'activation_name', 'tanh');

model2.build();
model2.run();

pd2 = model2.plot_data;
k = model2.k;
cmap1 = LNN2.default_colormap(k);
cmap2 = LNN2.default_colormap(k);

fig2 = figure('Name', 'LNN2 Multi-Layer', 'Position', [100 50 1000 900]);

% Panel 1: Layer 1 states
ax_ml1 = subplot(4, 1, 1);
hold on;
for i = 1:k
    plot(pd2.t, pd2.x(:, i), 'Color', cmap1(i, :), 'LineWidth', 0.5);
end
hold off;
title('Layer 1 States x_{1:k}(t)');
ylabel('State');

% Panel 2: Layer 2 states
ax_ml2 = subplot(4, 1, 2);
hold on;
for i = 1:k
    plot(pd2.t, pd2.x(:, k + i), 'Color', cmap2(i, :), 'LineWidth', 0.5);
end
hold off;
title('Layer 2 States x_{k+1:2k}(t)');
ylabel('State');

% Panel 3: Layer 1 f_ff
ax_ml3 = subplot(4, 1, 3);
hold on;
for i = 1:k
    plot(pd2.t, pd2.f_ff(:, i), 'Color', cmap1(i, :), 'LineWidth', 0.5);
end
hold off;
title('Layer 1: f_{ff} (sensory activation)');
ylabel('f_{ff}');

% Panel 4: Layer 2 f_ff
ax_ml4 = subplot(4, 1, 4);
hold on;
for i = 1:k
    plot(pd2.t, pd2.f_ff(:, k + i), 'Color', cmap2(i, :), 'LineWidth', 0.5);
end
hold off;
title('Layer 2: f_{ff} (feedforward from Layer 1)');
ylabel('f_{ff}');
xlabel('Time (s)');

sgtitle('LNN2 Multi-Layer — 2 layers × 25 neurons');
linkaxes([ax_ml1, ax_ml2, ax_ml3, ax_ml4], 'x');

fprintf('\n=== LNN2 Tests Complete ===\n');
