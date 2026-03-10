classdef ESN_reservoir < handle
    % ESN_RESERVOIR Unified Echo State Network reservoir (composition-based).
    %
    % Wraps any cRNN subclass (SRNNModel2, LNN) for memory capacity
    % measurement. Uses ESNStimulus for scalar input generation and
    % model.get_readout_features() for model-agnostic readout.
    %
    % Usage:
    %   esn = ESN_reservoir(SRNNModel2('n', 100));
    %   esn = ESN_reservoir(LNN('n', 100, 'n_in', 1));
    %   esn.build();
    %   [MC, R2_d] = esn.run_memory_capacity();
    %   esn.plot_memory_capacity();
    %   esn.model.plot();  % model-specific time series
    %
    % See also: cRNN, SRNNModel2, LNN, ESNStimulus

    %% Core
    properties
        model                       % cRNN subclass handle (SRNNModel2 or LNN)
    end

    %% Memory Capacity Protocol
    properties
        T_wash = 1000               % Washout samples (discard transients)
        T_train = 5000              % Training samples
        T_test = 5000               % Test samples
        d_max = 70                  % Maximum delay
        eta = 1e-7                  % Ridge regression regularization
    end

    %% ESN Input Properties
    properties
        f_in = 0.1                  % Fraction of neurons receiving input
        sigma_in = 0.5              % Input weight scaling
        input_type = 'one_over_f'   % 'white', 'bandlimited', 'one_over_f'
        u_alpha = 1                 % Spectral exponent for 1/f^alpha
        u_scale = 1                 % Stimulus amplitude scaling
        u_offset = 0                % Stimulus DC offset
        u_f_cutoff = []             % Bandlimited cutoff (Hz); auto if empty
    end

    %% Results
    properties (SetAccess = protected)
        mc_results                  % Struct with memory capacity results
    end

    %% Constructor
    methods
        function obj = ESN_reservoir(model, varargin)
            % ESN_RESERVOIR Constructor.
            %
            % Usage:
            %   esn = ESN_reservoir(SRNNModel2('n', 100))
            %   esn = ESN_reservoir(LNN('n', 50), 'd_max', 100, 'T_wash', 2000)

            if ~isa(model, 'cRNN')
                error('ESN_reservoir:InvalidModel', ...
                    'model must be a cRNN subclass. Got: %s', class(model));
            end
            obj.model = model;

            % Parse ESN-specific name-value pairs
            for i = 1:2:length(varargin)
                if isprop(obj, varargin{i})
                    obj.(varargin{i}) = varargin{i+1};
                else
                    warning('ESN_reservoir:UnknownProperty', ...
                        'Unknown property: %s', varargin{i});
                end
            end
        end
    end

    %% Public Methods
    methods
        function build(obj)
            % BUILD Configure model for ESN mode and build.
            %
            % Creates an ESNStimulus from ESN properties, assigns it to
            % the model, and calls model.build().

            % Create ESNStimulus with current ESN parameters
            stim = ESNStimulus();
            stim.T_wash = obj.T_wash;
            stim.T_train = obj.T_train;
            stim.T_test = obj.T_test;
            stim.f_in = obj.f_in;
            stim.sigma_in = obj.sigma_in;
            stim.input_type = obj.input_type;
            stim.u_alpha = obj.u_alpha;
            stim.u_scale = obj.u_scale;
            stim.u_offset = obj.u_offset;
            stim.u_f_cutoff = obj.u_f_cutoff;

            % Assign to model and build
            obj.model.stimulus = stim;

            % Force store_full_state for readout extraction
            obj.model.store_full_state = true;

            obj.model.build();

            fprintf('ESN_reservoir built: model=%s, n=%d\n', class(obj.model), obj.model.n);
        end

        function [MC, R2_d, results] = run_memory_capacity(obj, varargin)
            % RUN_MEMORY_CAPACITY Measure memory capacity.
            %
            % [MC, R2_d, results] = esn.run_memory_capacity()
            % [MC, R2_d, results] = esn.run_memory_capacity('verbose', true)
            %
            % Protocol:
            %   1. Run reservoir via model.run()
            %   2. Extract readout features via model.get_readout_features()
            %   3. Washout, train/test split
            %   4. Ridge regression for each delay
            %   5. MC = sum(R²_d)

            if ~obj.model.is_built
                error('ESN_reservoir:NotBuilt', 'Call build() first.');
            end

            % Parse options
            verbose = true;
            for i = 1:2:length(varargin)
                if strcmpi(varargin{i}, 'verbose')
                    verbose = varargin{i+1};
                end
            end

            if verbose
                fprintf('Running memory capacity measurement...\n');
                fprintf('  Model: %s, n=%d, readout_mode=%s\n', ...
                    class(obj.model), obj.model.n, obj.model.readout_mode);
                fprintf('  Washout: %d, Train: %d, Test: %d\n', ...
                    obj.T_wash, obj.T_train, obj.T_test);
                fprintf('  Max delay: %d\n', obj.d_max);
            end

            %% Step 1: Run reservoir
            if verbose, fprintf('  Running reservoir...\n'); end
            obj.model.run();

            %% Step 2: Extract readout features (model-agnostic)
            R_all = obj.model.get_readout_features();  % n_features × T_total
            n_features = size(R_all, 1);

            % Get scalar input from stimulus
            stim = obj.model.stimulus;
            u_scalar = stim.u_scalar;

            %% Step 3: Washout + train/test split
            R_eff = R_all(:, (obj.T_wash + 1):end);
            u_eff = u_scalar((obj.T_wash + 1):end);

            R_train = R_eff(:, 1:obj.T_train)';                  % T_train × n_features
            R_test = R_eff(:, (obj.T_train + 1):end)';           % T_test × n_features
            u_train = u_eff(1:obj.T_train);
            u_test = u_eff((obj.T_train + 1):end);

            %% Step 4: Ridge regression for each delay
            if verbose, fprintf('  Training readouts for %d delays...\n', obj.d_max); end

            R2_d = zeros(1, obj.d_max);
            weights_all = zeros(n_features, obj.d_max);
            predictions = struct();

            for d = 1:obj.d_max
                train_indices = (d + 1):obj.T_train;
                target_indices = 1:(obj.T_train - d);

                if length(train_indices) < 10
                    continue;
                end

                X_train = R_train(train_indices, :);
                y_train = u_train(target_indices);
                w_d = ESN_reservoir.train_linear_readout(X_train, y_train, obj.eta);
                weights_all(:, d) = w_d;

                % Test
                test_indices = (d + 1):obj.T_test;
                test_target_indices = 1:(obj.T_test - d);
                if length(test_indices) < 10, continue; end

                X_test = R_test(test_indices, :);
                y_test_true = u_test(test_target_indices);
                y_test_pred = X_test * w_d;

                predictions(d).y_true = y_test_true;
                predictions(d).y_pred = y_test_pred;
                predictions(d).t_indices = test_indices;

                R2_d(d) = ESN_reservoir.compute_R2(y_test_true, y_test_pred);

                if verbose && mod(d, 10) == 0
                    fprintf('    Delay %d: R^2 = %.4f\n', d, R2_d(d));
                end
            end

            %% Step 5: Total MC
            MC = sum(R2_d);
            if verbose, fprintf('  Total Memory Capacity: %.4f\n', MC); end

            %% Store results
            results = struct();
            results.MC = MC;
            results.R2_d = R2_d;
            results.d = 1:obj.d_max;
            results.weights = weights_all;
            results.T_wash = obj.T_wash;
            results.T_train = obj.T_train;
            results.T_test = obj.T_test;
            results.eta = obj.eta;
            results.readout_mode = obj.model.readout_mode;
            results.model_class = class(obj.model);
            results.u_scalar = u_scalar;
            results.predictions = predictions;

            % Time info for plotting
            test_start_idx = obj.T_wash + obj.T_train + 1;
            results.t_test = obj.model.t_out(test_start_idx:end);
            results.u_test = u_test;

            %% Lyapunov (if configured)
            if ~strcmpi(obj.model.lya_method, 'none')
                t_all = obj.model.t_out;
                obj.model.lya_T_interval = [t_all(obj.T_wash + 1), t_all(end)];
                obj.model.compute_lyapunov();
                results.lya_results = obj.model.lya_results;
            end

            obj.mc_results = results;
        end

        function [fig_handle, ax_handles] = plot_memory_capacity(obj)
            % PLOT_MEMORY_CAPACITY Plot R²(d) and cumulative MC.

            if isempty(obj.mc_results)
                error('ESN_reservoir:NoResults', 'Run run_memory_capacity() first.');
            end

            r = obj.mc_results;
            fig_handle = figure('Name', 'Memory Capacity');
            ax_handles = gobjects(2, 1);

            % R² vs delay
            ax_handles(1) = subplot(1, 2, 1);
            bar(r.d, r.R2_d, 'FaceColor', [0.3, 0.6, 0.9]);
            xlabel('Delay d (samples)');
            ylabel('R^2_d');
            title('Memory Capacity by Delay');
            grid on;

            % Cumulative MC
            ax_handles(2) = subplot(1, 2, 2);
            plot(r.d, cumsum(r.R2_d), 'b-', 'LineWidth', 2);
            xlabel('Delay d (samples)');
            ylabel('Cumulative MC');
            title(sprintf('Cumulative MC (Total: %.2f)', r.MC));
            grid on;

            sgtitle(sprintf('%s Memory Capacity (MC = %.2f, readout: %s)', ...
                r.model_class, r.MC, r.readout_mode));
        end

        function [fig_handle, ax_handles] = plot_esn_timeseries(obj, delays_to_plot, varargin)
            % PLOT_ESN_TIMESERIES Plot delay reconstruction panels.
            %
            % [fig, axes] = esn.plot_esn_timeseries([1, 5, 10, 30])
            %
            % Generic panels:
            %   1. Scalar input u(t)
            %   2. Delay reconstruction overlays for selected delays

            if isempty(obj.mc_results)
                error('ESN_reservoir:NoResults', 'Run run_memory_capacity() first.');
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

            r = obj.mc_results;
            n_panels = 1 + n_delays;  % scalar input + delay panels

            fig_handle = figure('Name', 'ESN Time Series');
            tiledlayout(n_panels, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
            ax_handles = gobjects(n_panels, 1);

            % Panel 1: Scalar input
            ax_handles(1) = nexttile;
            t_test = r.t_test;
            u_test = r.u_test;
            plot(t_test, u_test, 'k-', 'LineWidth', 0.5);
            ylabel('u(t)');
            title('Scalar Input (test period)');
            set(gca, 'XTickLabel', []);
            grid on;

            % Delay reconstruction panels
            colors_pred = lines(n_delays);
            for idx = 1:n_delays
                d = delays_to_plot(idx);
                ax_handles(1 + idx) = nexttile;

                if d <= length(r.predictions) && ...
                        isfield(r.predictions(d), 'y_true') && ...
                        ~isempty(r.predictions(d).y_true)

                    y_true = r.predictions(d).y_true;
                    y_pred = r.predictions(d).y_pred;
                    t_pred = t_test(r.predictions(d).t_indices);

                    plot(t_pred, y_true, 'k-', 'LineWidth', 0.8, ...
                        'DisplayName', sprintf('u(t-%d)', d));
                    hold on;
                    plot(t_pred, y_pred, '-', 'Color', colors_pred(idx,:), ...
                        'LineWidth', 1.2, 'DisplayName', sprintf('y_%d(t)', d));
                    hold off;

                    R2 = r.R2_d(d);
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
            xlim([t_test(1), t_test(end)]);

            if ~isempty(fig_title)
                sgtitle(fig_title);
            else
                sgtitle(sprintf('%s ESN (MC = %.2f)', r.model_class, r.MC));
            end
        end

        function reset(obj)
            % RESET Clear MC results.
            obj.mc_results = [];
            fprintf('ESN results cleared.\n');
        end
    end

    %% Static Methods
    methods (Static)
        function w = train_linear_readout(X, y, eta)
            % TRAIN_LINEAR_READOUT Ridge regression: w = (X'X + eta*I)^-1 * X'y
            n_features = size(X, 2);
            w = (X' * X + eta * eye(n_features)) \ (X' * y);
        end

        function R2 = compute_R2(y_true, y_pred)
            % COMPUTE_R2 Squared correlation coefficient.
            y_true = y_true(:);
            y_pred = y_pred(:);

            var_true = var(y_true);
            var_pred = var(y_pred);

            if var_true < 1e-12 || var_pred < 1e-12
                R2 = 0;
                return;
            end

            cov_matrix = cov(y_true, y_pred);
            R2 = (cov_matrix(1,2)^2) / (var_true * var_pred);
            R2 = max(0, min(1, R2));
        end

        function verify_shared_build(esn_array, expected_to_differ, also_check_protected)
            % VERIFY_SHARED_BUILD Verify ESN objects share configuration.
            %
            % ESN_reservoir.verify_shared_build(esn_array, expected_to_differ, also_check_protected)

            if numel(esn_array) < 2
                fprintf('verify_shared_build: only 1 object, nothing to compare.\n');
                return;
            end

            ref = esn_array{1};
            mc = metaclass(ref);
            n_obj = numel(esn_array);

            always_skip = {'mc_results'};

            n_checked = 0;
            checked_names = {};

            for p = 1:numel(mc.PropertyList)
                prop = mc.PropertyList(p);
                name = prop.Name;

                if prop.Dependent, continue; end
                if ismember(name, always_skip), continue; end
                if ismember(name, expected_to_differ), continue; end

                is_public_get = strcmp(prop.GetAccess, 'public');
                is_in_also_check = nargin >= 3 && ismember(name, also_check_protected);

                if ~is_public_get && ~is_in_also_check
                    continue;
                end

                for i = 2:n_obj
                    val_ref = ref.(name);
                    val_obj = esn_array{i}.(name);

                    if isa(val_ref, 'function_handle') && isa(val_obj, 'function_handle')
                        match = strcmp(func2str(val_ref), func2str(val_obj));
                    elseif isa(val_ref, 'handle') && isa(val_obj, 'handle')
                        match = true;  % Skip handle comparison (model objects)
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

            fprintf('verify_shared_build: %d properties matched across %d conditions.\n', ...
                n_checked, n_obj);
        end
    end
end
