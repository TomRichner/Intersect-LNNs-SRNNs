% test_LNN1.m - Run LNN1 (Hasani Python-faithful) with defaults and plot
%
% Tests the per-synapse conductance model with 4-panel figure:
% Panel 1: External input I(t)
% Panel 2: Neuron states v(t)
% Panel 3: Mapped input
% Panel 4: State distribution over time

close all; clear; clc;

% Add paths
setup_paths();

%% ─── Default Test (semi-implicit solver) ───────────────────────────────
fprintf('=== LNN1 Test (semi-implicit solver) ===\n');
model = LNN1('n', 32, 'n_in', 2, ...
             'T_range', [0, 10], ...
             'rng_seeds', [42 42]);

model.build();
model.run();

pd = model.plot_data;
n = model.n;
n_show = min(n, 50);
cmap = LNN2.default_colormap(n_show);

fig = figure('Name', 'LNN1 Per-Synapse Model', 'Position', [50 50 1000 800]);

% Panel 1: Input
ax1 = subplot(4, 1, 1);
plot(pd.t, pd.u);
title('External Input I(t)');
ylabel('Amplitude');
legend({'sin(2\pit)', 'cos(2\pit)'}, 'Location', 'northeast');

% Panel 2: Neuron voltages
ax2 = subplot(4, 1, 2);
hold on;
for i = 1:n_show
    plot(pd.t, pd.x(:, i), 'Color', cmap(i, :), 'LineWidth', 0.5);
end
hold off;
title(sprintf('Neuron States v(t) — n=%d', n));
ylabel('v');

% Panel 3: State mean ± std
ax3 = subplot(4, 1, 3);
v_mean = mean(pd.x, 2);
v_std = std(pd.x, [], 2);
fill_x = [pd.t(:); flipud(pd.t(:))];
fill_y = [v_mean + v_std; flipud(v_mean - v_std)];
fill(fill_x, fill_y, [0.7 0.85 1], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
hold on;
plot(pd.t, v_mean, 'b', 'LineWidth', 1.5);
hold off;
title('State Mean ± Std');
ylabel('v');

% Panel 4: State histogram over time (last 20% of simulation)
ax4 = subplot(4, 1, 4);
nt_d = length(pd.t);
t_start_idx = round(0.8 * nt_d);
v_late = pd.x(t_start_idx:end, :);
histogram(v_late(:), 50, 'Normalization', 'pdf', 'FaceColor', [0.3 0.5 0.8]);
title(sprintf('State Distribution (t > %.1f s)', pd.t(t_start_idx)));
xlabel('v');
ylabel('PDF');

sgtitle(sprintf('LNN1 (Hasani Python-faithful) — n=%d, semi-implicit, %d unfolds', ...
    n, model.ode_solver_unfolds));

fprintf('State range: [%.4f, %.4f]\n', min(pd.x(:)), max(pd.x(:)));
fprintf('=== LNN1 Test Complete ===\n');
