classdef LNN_ESN_reservoir < LNN
    % LNN_ESN_RESERVOIR Echo State Network reservoir with LTC dynamics
    %
    % Extends LNN to provide ESN functionality with memory capacity
    % measurement, mirroring SRNN_ESN_reservoir's relationship to SRNNModel2.
    %
    % The LTC reservoir is driven with a scalar random input, and linear
    % readouts are trained to reconstruct delayed versions of the input.
    % Memory capacity = sum of R^2_d for d = 1 to d_max.
    %
    % Usage:
    %   esn = LNN_ESN_reservoir('n', 100, 'level_of_chaos', 1.0);
    %   esn.build();
    %   [MC, R2_d] = esn.run_memory_capacity();
    %   esn.plot_memory_capacity();
    %
    % See also: LNN, SRNN_ESN_reservoir

    %% ESN Input Properties
    properties
        f_in = 0.1                  % Fraction of neurons receiving input
        sigma_in = 0.5              % Input weight scaling parameter
        rng_seed_input = 3          % RNG seed for input weight generation
        input_type = 'one_over_f'   % 'white', 'bandlimited', or 'one_over_f'
        u_f_cutoff = []             % Cutoff frequency for bandlimited (Hz)
        u_alpha = 1                 % Spectral exponent for 1/f^alpha noise
        u_scale = 1                 % Stimulus amplitude scaling
        u_offset = 0                % Stimulus DC offset
    end

    %% Memory Capacity Protocol Properties
    properties
        T_wash = 1000               % Washout samples (discard transients)
        T_train = 5000              % Training samples
        T_test = 5000               % Test samples
        d_max = 70                  % Maximum delay for memory capacity
        eta = 1e-7                  % Ridge regression regularization
        readout_mode = 'state'      % 'state' (x) or 'nonlinearity' (f)
    end

    %% Protected Build-Output Properties
    properties (SetAccess = protected)
        W_in_esn                    % Scalar input weight vector (n x 1)
        u_scalar                    % Scalar input sequence (T_total x 1)
    end

    %% Memory Capacity Results
    properties (SetAccess = private)
        mc_results                  % Struct with memory capacity results
    end

    %% Constructor
    methods
        function obj = LNN_ESN_reservoir(varargin)
            % LNN_ESN_RESERVOIR Constructor with name-value pairs
            %
            % Usage:
            %   esn = LNN_ESN_reservoir()
            %   esn = LNN_ESN_reservoir('n', 200, 'd_max', 100)

            % Call superclass constructor (LNN handles unknown props with warning)
            obj = obj@LNN(varargin{:});

            % Force n_in = 1 for ESN scalar input
            obj.n_in = 1;

            % Define ESN-specific property names (not in LNN)
            esn_props = {'f_in', 'sigma_in', 'rng_seed_input', ...
                'T_wash', 'T_train', 'T_test', 'd_max', 'eta', ...
                'input_type', 'u_f_cutoff', 'u_alpha', 'u_scale', 'u_offset', ...
                'readout_mode'};

            % Parse ESN-specific name-value pairs
            for i = 1:2:length(varargin)
                if ismember(varargin{i}, esn_props)
                    obj.(varargin{i}) = varargin{i+1};
                end
            end
        end
    end

    %% Public Methods
    methods
        function generate_input_weights(obj)
            % GENERATE_INPUT_WEIGHTS Create sparse input weight vector W_in_esn
            %
            % f_in fraction of neurons receive input with weights drawn
            % uniformly from [-sigma_in/2, sigma_in/2].

            rng(obj.rng_seed_input);

            obj.W_in_esn = zeros(obj.n, 1);
            n_input = round(obj.f_in * obj.n);
            input_neurons = randperm(obj.n, n_input);
            obj.W_in_esn(input_neurons) = obj.sigma_in * (rand(n_input, 1) - 0.5);

            fprintf('Input weights generated: %d neurons receive input (%.1f%%)\n', ...
                n_input, 100 * n_input / obj.n);
        end

        function [MC, R2_d, mc_results] = run_memory_capacity(obj, varargin)
            % RUN_MEMORY_CAPACITY Measure memory capacity of the LTC reservoir
            %
            % [MC, R2_d, results] = run_memory_capacity()
            % [MC, R2_d, results] = run_memory_capacity('verbose', true)
            %
            % Protocol:
            %   1. Run reservoir with pre-built scalar input
            %   2. Collect reservoir states (x or f depending on readout_mode)
            %   3. Discard washout, split train/test
            %   4. Train linear readouts for each delay via ridge regression
            %   5. Compute R^2_d on test set
            %   6. Memory capacity = sum(R^2_d)

            if ~obj.is_built
                error('LNN_ESN_reservoir:NotBuilt', ...
                    'Reservoir must be built first. Call build().');
            end

            % Parse optional arguments
            verbose = true;
            for i = 1:2:length(varargin)
                if strcmpi(varargin{i}, 'verbose')
                    verbose = varargin{i+1};
                end
            end

            if verbose
                fprintf('Running memory capacity measurement...\n');
                fprintf('  Washout: %d, Train: %d, Test: %d samples\n', ...
                    obj.T_wash, obj.T_train, obj.T_test);
                fprintf('  Max delay: %d, Readout mode: %s\n', obj.d_max, obj.readout_mode);
            end

            %% Step 1: Run reservoir
            if verbose
                fprintf('  Running reservoir simulation...\n');
            end
            obj.run_reservoir_esn();

            %% Step 2: Collect readout features
            T_total = size(obj.x_out, 1);
            params = obj.cached_params;

            if strcmpi(obj.readout_mode, 'nonlinearity')
                % Compute f(t) for each timestep
                if verbose
                    fprintf('  Computing nonlinearity features...\n');
                end
                R_all = zeros(obj.n, T_total);
                for k = 1:T_total
                    I_k = obj.u_interpolant(obj.t_out(k))';
                    z = params.W_r * obj.x_out(k,:)' + params.W_in * I_k + params.mu;
                    R_all(:, k) = LNN.apply_activation(z, params.activation, params);
                end
            else
                % Use raw state x(t)
                R_all = obj.x_out';  % n x T
            end

            %% Step 3: Discard washout, split train/test
            R_eff = R_all(:, (obj.T_wash + 1):end);
            u_eff = obj.u_scalar((obj.T_wash + 1):end);

            R_train = R_eff(:, 1:obj.T_train)';
            R_test = R_eff(:, (obj.T_train + 1):end)';
            u_train = u_eff(1:obj.T_train);
            u_test = u_eff((obj.T_train + 1):end);

            %% Step 4-5: Train readouts and compute R^2
            if verbose
                fprintf('  Training readouts for %d delays...\n', obj.d_max);
            end

            R2_d = zeros(1, obj.d_max);
            weights_all = zeros(obj.n, obj.d_max);
            predictions = struct();

            for d = 1:obj.d_max
                train_indices = (d + 1):obj.T_train;
                target_indices = 1:(obj.T_train - d);

                if length(train_indices) < 10
                    warning('LNN_ESN_reservoir:InsufficientData', ...
                        'Delay %d has insufficient training data.', d);
                    continue;
                end

                X_train = R_train(train_indices, :);
                y_train = u_train(target_indices);

                % Ridge regression
                w_d = LNN_ESN_reservoir.train_linear_readout(X_train, y_train, obj.eta);
                weights_all(:, d) = w_d;

                % Test
                test_indices = (d + 1):obj.T_test;
                test_target_indices = 1:(obj.T_test - d);

                if length(test_indices) < 10
                    continue;
                end

                X_test = R_test(test_indices, :);
                y_test_true = u_test(test_target_indices);
                y_test_pred = X_test * w_d;

                predictions(d).y_true = y_test_true;
                predictions(d).y_pred = y_test_pred;
                predictions(d).t_indices = test_indices;

                R2_d(d) = LNN_ESN_reservoir.compute_R2(y_test_true, y_test_pred);

                if verbose && mod(d, 10) == 0
                    fprintf('    Delay %d: R^2 = %.4f\n', d, R2_d(d));
                end
            end

            %% Step 6: Total memory capacity
            MC = sum(R2_d);

            if verbose
                fprintf('  Total Memory Capacity: %.4f\n', MC);
            end

            %% Store results
            mc_results = struct();
            mc_results.MC = MC;
            mc_results.R2_d = R2_d;
            mc_results.d = 1:obj.d_max;
            mc_results.weights = weights_all;
            mc_results.T_wash = obj.T_wash;
            mc_results.T_train = obj.T_train;
            mc_results.T_test = obj.T_test;
            mc_results.eta = obj.eta;
            mc_results.readout_mode = obj.readout_mode;
            mc_results.u_scalar = obj.u_scalar;
            mc_results.predictions = predictions;

            % Store time series for plotting
            test_start_idx = obj.T_wash + obj.T_train + 1;
            mc_results.t_test = obj.t_out(test_start_idx:end);
            mc_results.u_test = u_test;
            mc_results.x_out = obj.x_out;
            mc_results.R_all = R_all;

            obj.mc_results = mc_results;
        end

        function run_reservoir_esn(obj)
            % RUN_RESERVOIR_ESN Run the LTC reservoir (single ODE integration)

            params = obj.cached_params;
            dt = 1 / obj.fs;

            if isempty(obj.ode_opts)
                obj.ode_opts = odeset('RelTol', 1e-5, 'AbsTol', 1e-5, 'MaxStep', dt);
            end

            if strcmpi(obj.solver_mode, 'fused')
                fprintf('  Running LTC reservoir with fused solver (%d substeps)...\n', obj.fused_substeps);
                tic
                [obj.t_out, obj.x_out] = obj.run_fused(params);
                run_time = toc;
            else
                params.u_interpolant = obj.u_interpolant;
                rhs = @(t, x) LNN.dynamics_ltc(t, x, params);

                fprintf('  Integrating LTC ESN dynamics with %s...\n', func2str(obj.ode_solver));
                tic
                [obj.t_out, obj.x_out] = obj.ode_solver(rhs, obj.t_ex, obj.S0, obj.ode_opts);
                run_time = toc;
            end

            fprintf('  Integration complete in %.2f seconds.\n', run_time);
        end

        function [fig_handle, ax_handles] = plot_memory_capacity(obj, varargin)
            % PLOT_MEMORY_CAPACITY Plot R^2 vs delay and cumulative MC

            if isempty(obj.mc_results)
                error('LNN_ESN_reservoir:NoResults', ...
                    'No MC results. Run run_memory_capacity() first.');
            end

            fig_handle = figure();
            ax_handles = gobjects(2, 1);

            % R^2 vs delay
            ax_handles(1) = subplot(1, 2, 1);
            bar(obj.mc_results.d, obj.mc_results.R2_d, 'FaceColor', [0.3, 0.6, 0.9]);
            xlabel('Delay d (samples)');
            ylabel('R^2_d');
            title('Memory Capacity by Delay');
            grid on;

            % Cumulative MC
            ax_handles(2) = subplot(1, 2, 2);
            cumMC = cumsum(obj.mc_results.R2_d);
            plot(obj.mc_results.d, cumMC, 'b-', 'LineWidth', 2);
            xlabel('Delay d (samples)');
            ylabel('Cumulative MC');
            title(sprintf('Cumulative MC (Total: %.2f)', obj.mc_results.MC));
            grid on;

            sgtitle(sprintf('LNN Memory Capacity (MC = %.2f)', obj.mc_results.MC));
        end

        function [fig_handle, ax_handles] = plot_esn_timeseries(obj, delays_to_plot, varargin)
            % PLOT_ESN_TIMESERIES Plot time series for LTC ESN reservoir
            %
            % [fig, axes] = plot_esn_timeseries(delays_to_plot)
            %
            % Panels:
            %   1. u(t): Scalar input mapped through W_in_esn
            %   2. x(t): LTC neuron states
            %   3. f(t): Nonlinearity output
            %   4. tau_sys(t): Effective time constants
            %   5+: Delay reconstruction overlays

            if isempty(obj.mc_results)
                error('LNN_ESN_reservoir:NoResults', ...
                    'No time series data. Run run_memory_capacity() first.');
            end

            if nargin < 2 || isempty(delays_to_plot)
                delays_to_plot = [1, 10, 30, 50];
            end
            delays_to_plot = delays_to_plot(delays_to_plot <= obj.d_max);
            n_delays = length(delays_to_plot);

            % Parse optional args
            fig_title = '';
            for i = 1:2:length(varargin)
                if strcmpi(varargin{i}, 'title')
                    fig_title = varargin{i+1};
                end
            end

            % Slice to test period
            test_start_idx = obj.T_wash + obj.T_train + 1;
            test_end_idx = size(obj.mc_results.x_out, 1);
            test_indices = test_start_idx:test_end_idx;
            t = obj.t_out(test_indices);
            x_test = obj.mc_results.x_out(test_indices, :);
            params = obj.cached_params;

            % Compute f and tau_sys for test period
            nt_test = length(t);
            f_test = zeros(nt_test, obj.n);
            tau_sys_test = zeros(nt_test, obj.n);
            u_neural_test = zeros(nt_test, 1);

            for k = 1:nt_test
                I_k = obj.u_interpolant(t(k))';
                u_neural_test(k) = obj.u_scalar(test_indices(k));
                z = params.W_r * x_test(k, :)' + params.W_in * I_k + params.mu;
                f_k = LNN.apply_activation(z, params.activation, params);
                f_test(k, :) = f_k';
                tau_sys_test(k, :) = (params.tau ./ (1 + params.tau .* abs(f_k)))';
            end

            % Number of panels
            n_base = 4;  % u, x, f, tau_sys
            n_total = n_base + n_delays;

            fig_handle = figure();
            tiledlayout(n_total, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
            ax_handles = gobjects(n_total, 1);

            cmap = LNN.default_colormap(obj.n);

            %% Panel 1: Scalar input
            ax_handles(1) = nexttile;
            plot(t, u_neural_test, 'k-', 'LineWidth', 0.5);
            ylabel('u(t)');
            title('Scalar Input');
            set(gca, 'XTickLabel', []);
            grid on;

            %% Panel 2: Neuron states x(t)
            ax_handles(2) = nexttile;
            hold on;
            for i = 1:obj.n
                plot(t, x_test(:, i), 'Color', cmap(i, :), 'LineWidth', 0.5);
            end
            hold off;
            ylabel('x(t)');
            title('LTC Neuron States');
            set(gca, 'XTickLabel', []);
            grid on;

            %% Panel 3: Nonlinearity f(t)
            ax_handles(3) = nexttile;
            hold on;
            for i = 1:obj.n
                plot(t, f_test(:, i), 'Color', cmap(i, :), 'LineWidth', 0.5);
            end
            hold off;
            ylabel('f(t)');
            title('Nonlinearity Output');
            set(gca, 'XTickLabel', []);
            grid on;

            %% Panel 4: Effective tau_sys
            ax_handles(4) = nexttile;
            hold on;
            for i = 1:obj.n
                plot(t, tau_sys_test(:, i), 'Color', cmap(i, :), 'LineWidth', 0.5);
            end
            hold off;
            ylabel('\tau_{sys}');
            title('Effective Time Constants');
            set(gca, 'XTickLabel', []);
            grid on;

            %% Delay reconstruction panels
            colors_pred = lines(n_delays);
            for idx = 1:n_delays
                d = delays_to_plot(idx);
                ax_handles(n_base + idx) = nexttile;

                if d <= length(obj.mc_results.predictions) && ...
                        isfield(obj.mc_results.predictions(d), 'y_true') && ...
                        ~isempty(obj.mc_results.predictions(d).y_true)

                    y_true = obj.mc_results.predictions(d).y_true;
                    y_pred = obj.mc_results.predictions(d).y_pred;
                    t_pred = obj.mc_results.t_test(obj.mc_results.predictions(d).t_indices);

                    plot(t_pred, y_true, 'k-', 'LineWidth', 0.8, ...
                        'DisplayName', sprintf('u(t-%d)', d));
                    hold on;
                    plot(t_pred, y_pred, '-', 'Color', colors_pred(idx,:), ...
                        'LineWidth', 1.2, 'DisplayName', sprintf('y_%d(t)', d));
                    hold off;

                    R2 = obj.mc_results.R2_d(d);
                    title(sprintf('Delay d=%d: R^2=%.3f', d, R2), 'FontWeight', 'normal');
                    legend('Location', 'best', 'FontSize', 7);
                else
                    text(0.5, 0.5, sprintf('No data for delay %d', d), ...
                        'HorizontalAlignment', 'center');
                    title(sprintf('Delay d=%d', d), 'FontWeight', 'normal');
                end

                if idx < n_delays
                    set(gca, 'XTickLabel', []);
                else
                    xlabel('Time (s)');
                end
                grid on;
            end

            linkaxes(ax_handles, 'x');
            xlim([t(1), t(end)]);

            if ~isempty(fig_title)
                sgtitle(fig_title);
            else
                sgtitle(sprintf('LNN ESN Time Series (MC = %.2f)', obj.mc_results.MC));
            end
        end

        function reset(obj)
            % RESET Clear built state and MC results
            reset@LNN(obj);
            obj.W_in_esn = [];
            obj.u_scalar = [];
            obj.mc_results = [];
            fprintf('ESN reservoir reset.\n');
        end
    end

    %% Protected Build Sub-Methods (override LNN)
    methods (Access = protected)
        function build_stimulus(obj)
            % BUILD_STIMULUS Generate ESN-specific scalar input stimulus
            %
            % Overrides LNN.build_stimulus() to:
            %   1. Generate sparse input weight vector W_in_esn
            %   2. Generate scalar input (white/bandlimited/1_over_f)
            %   3. Map scalar to neural input via W_in_esn
            %   4. Set parent's W_in to W_in_esn (n x 1 for n_in=1)
            %   5. Build interpolant and initial state

            % 1. Generate input weights
            obj.generate_input_weights();

            % 2. Generate scalar input sequence
            T_total = obj.T_wash + obj.T_train + obj.T_test;
            rng(obj.rng_seed + 1);  % Different seed from network

            if strcmpi(obj.input_type, 'bandlimited')
                u_raw = rand(T_total, 1) - 0.5;

                if isempty(obj.u_f_cutoff)
                    f_cut = 1 / (2 * pi * mean(obj.tau));
                else
                    f_cut = obj.u_f_cutoff;
                end

                f_nyq = obj.fs / 2;
                [b_filt, a_filt] = butter(3, f_cut / f_nyq, 'low');
                u_filtered = filtfilt(b_filt, a_filt, u_raw);
                u_normalized = (u_filtered - min(u_filtered)) / (max(u_filtered) - min(u_filtered)) - 0.5;
                obj.u_scalar = obj.u_offset + obj.u_scale * u_normalized;
                fprintf('ESN stimulus: bandlimited input (f_cutoff = %.2f Hz)\n', f_cut);

            elseif strcmpi(obj.input_type, 'one_over_f')
                alpha_val = obj.u_alpha;
                N = T_total;
                X = randn(N, 1) + 1i * randn(N, 1);
                df = obj.fs / N;
                freq = (0:(N-1))' * df;
                freq(freq > obj.fs/2) = freq(freq > obj.fs/2) - obj.fs;
                freq = abs(freq);
                freq(freq < df) = df;
                scale_factor = freq.^(-alpha_val/2);
                X_shaped = X .* scale_factor;
                X_shaped(1) = real(X_shaped(1));
                if mod(N, 2) == 0
                    X_shaped(N/2 + 1) = real(X_shaped(N/2 + 1));
                end
                u_raw = real(ifft(X_shaped));
                u_normalized = (u_raw - min(u_raw)) / (max(u_raw) - min(u_raw)) - 0.5;
                obj.u_scalar = obj.u_offset + obj.u_scale * u_normalized;
                fprintf('ESN stimulus: 1/f^%.2f noise input\n', alpha_val);

            elseif strcmpi(obj.input_type, 'white')
                obj.u_scalar = obj.u_offset + obj.u_scale * (rand(T_total, 1) - 0.5);
                fprintf('ESN stimulus: white noise input\n');
            else
                error('LNN_ESN_reservoir:InvalidInputType', ...
                    'Unknown input_type ''%s''. Valid: ''white'', ''bandlimited'', ''one_over_f''', ...
                    obj.input_type);
            end

            % 3. Map scalar to time vector
            dt = 1 / obj.fs;
            obj.t_ex = (0:(T_total-1))' * dt;
            % Store full neural input for reference/plotting
            obj.u_ex = obj.W_in_esn * obj.u_scalar';  % n x T

            % 4. Override parent's W_in to match ESN scalar input
            obj.W_in = obj.W_in_esn;  % n x 1 (matches n_in = 1)

            % 5. Build interpolant for SCALAR input (1 x T)
            % dynamics_ltc computes W_in * I_t, so interpolant must return (1 x 1)
            obj.u_interpolant = griddedInterpolant(obj.t_ex, obj.u_scalar, 'linear', 'nearest');

            % 6. Initialize state
            obj.S0 = 0.01 * randn(obj.n, 1);

            fprintf('ESN stimulus built: %d samples, %d neurons receive input\n', ...
                T_total, sum(obj.W_in_esn ~= 0));
        end
    end

    %% Static Methods (internalized)
    methods (Static)
        function w = train_linear_readout(X, y, eta)
            % TRAIN_LINEAR_READOUT Ridge regression: w = (X'X + eta*I)^-1 * X'y
            n_features = size(X, 2);
            XTX = X' * X;
            XTy = X' * y;
            w = (XTX + eta * eye(n_features)) \ XTy;
        end

        function R2 = compute_R2(y_true, y_pred)
            % COMPUTE_R2 Squared correlation coefficient
            y_true = y_true(:);
            y_pred = y_pred(:);

            var_true = var(y_true);
            var_pred = var(y_pred);

            if var_true < 1e-12 || var_pred < 1e-12
                R2 = 0;
                return;
            end

            cov_matrix = cov(y_true, y_pred);
            cov_val = cov_matrix(1, 2);
            R2 = (cov_val^2) / (var_true * var_pred);
            R2 = max(0, min(1, R2));
        end

        function verify_shared_build(esn_array, expected_to_differ, also_check_protected)
            % VERIFY_SHARED_BUILD Verify built ESN objects share configuration
            %
            % LNN_ESN_reservoir.verify_shared_build(esn_array, expected_to_differ, also_check_protected)

            if numel(esn_array) < 2
                fprintf('verify_shared_build: only 1 object, nothing to compare.\n');
                return;
            end

            ref = esn_array{1};
            mc = metaclass(ref);
            n_obj = numel(esn_array);

            always_skip = {'S0', 'cached_params', 'mc_results', 'u_interpolant', ...
                'ode_opts', 't_out', 'x_out', 'plot_data'};

            n_checked = 0;
            checked_names = {};

            for p = 1:numel(mc.PropertyList)
                prop = mc.PropertyList(p);
                name = prop.Name;

                if prop.Dependent, continue; end
                if ismember(name, always_skip), continue; end
                if ismember(name, expected_to_differ), continue; end

                is_public_get = strcmp(prop.GetAccess, 'public');
                is_in_also_check = ismember(name, also_check_protected);

                if ~is_public_get && ~is_in_also_check
                    continue;
                end

                for i = 2:n_obj
                    val_ref = ref.(name);
                    val_obj = esn_array{i}.(name);

                    if isa(val_ref, 'function_handle') && isa(val_obj, 'function_handle')
                        match = strcmp(func2str(val_ref), func2str(val_obj));
                    else
                        match = isequaln(val_ref, val_obj);
                    end

                    if ~match
                        error('verify_shared_build:Mismatch', ...
                            'Property ''%s'' differs between condition 1 and %d.', name, i);
                    end
                end

                n_checked = n_checked + 1;
                checked_names{end+1} = name; %#ok<AGROW>
            end

            % Verify expected_to_differ actually differ
            for k = 1:numel(expected_to_differ)
                name = expected_to_differ{k};
                prop_meta = findobj(mc.PropertyList, 'Name', name);
                if isempty(prop_meta), continue; end

                any_differs = false;
                for i = 2:n_obj
                    if ~isequaln(ref.(name), esn_array{i}.(name))
                        any_differs = true;
                        break;
                    end
                end

                if ~any_differs
                    warning('verify_shared_build:NoDifference', ...
                        'Property ''%s'' is listed in expected_to_differ but is identical.', name);
                end
            end

            fprintf('verify_shared_build: %d properties matched across %d conditions.\n', ...
                n_checked, n_obj);
        end
    end
end
