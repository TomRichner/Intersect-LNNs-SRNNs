classdef (Abstract) cRNN < handle
    % CRNN Abstract base class for continuous recurrent neural networks.
    %
    % Provides shared properties, build/run lifecycle, and Lyapunov
    % computation for continuous-time RNN models. Subclasses implement
    % model-specific dynamics via abstract methods.
    %
    % Uses strategy objects for pluggable components:
    %   - activation: Activation subclass (nonlinearity + derivative)
    %   - stimulus:   Stimulus subclass (input generation)
    %   - Connectivity via RMTMatrix (used in build_network)
    %
    % Abstract methods (subclasses MUST implement):
    %   get_rhs(params)           - Return @(t,S) ODE function handle
    %   get_readout_features()    - Return n × T feature matrix for ESN
    %   get_jacobian(S, params)   - Return Jacobian at state S
    %   initialize_state()        - Set obj.S0
    %   decimate_and_unpack()     - Model-specific decimation to plot_data
    %
    % Usage (via subclass):
    %   model = SRNNModel2('n', 300, 'n_a_E', 3);
    %   model.build(); model.run(); model.plot();
    %
    % See also: SRNNModel2, LNN, Activation, Stimulus, RMTMatrix

    %% ====================================================================
    %              NETWORK ARCHITECTURE PROPERTIES
    % =====================================================================
    properties
        n = 100                     % Total number of neurons
        f = 0.5                     % Fraction of excitatory neurons
        indegree                    % Expected in-degree (default: fully connected = n)

        % RMT tilde-notation parameters (Harris 2023)
        mu_E_tilde                  % Normalized excitatory mean
        mu_I_tilde                  % Normalized inhibitory mean
        sigma_E_tilde               % Normalized excitatory std dev
        sigma_I_tilde               % Normalized inhibitory std dev
        E_W = 0                     % Mean offset: added to both mu_E_tilde and mu_I_tilde
        zrs_mode = 'none'           % ZRS mode: 'none', 'ZRS', 'SZRS', 'Partial_SZRS'

        level_of_chaos = 1.0        % Scaling factor for W (spectral radius control)
    end

    %% ====================================================================
    %              SIMULATION SETTINGS
    % =====================================================================
    properties
        fs = 400                    % Sampling frequency (Hz)
        T_range = [0, 10]           % Simulation time interval [start, end]
        T_plot                      % Plotting time interval (defaults to T_range)
        ode_solver = @ode45         % ODE solver function handle
        ode_opts                    % ODE solver options struct
        rng_seeds = [1 2]           % RNG seeds: [network, stimulus]
    end

    %% ====================================================================
    %              STORAGE AND PLOTTING
    % =====================================================================
    properties
        store_full_state = false    % Whether to keep full state_out in memory
        store_decimated_state = true % Whether to keep decimated plot data
        plot_deci                   % Decimation factor for plotting
        plot_freq = 10              % Target plotting frequency (Hz)
    end

    %% ====================================================================
    %              STRATEGY OBJECTS
    % =====================================================================
    properties
        activation                  % Activation subclass (nonlinearity)
        stimulus                    % Stimulus subclass (input generation)
    end

    %% ====================================================================
    %              LYAPUNOV SETTINGS
    % =====================================================================
    properties
        lya_method = 'none'         % Lyapunov method: 'benettin', 'qr', or 'none'
        lya_T_interval              % Time interval for Lyapunov computation
        filter_local_lya = false    % Whether to filter local Lyapunov exponent
        lya_filter_order = 2        % Butterworth filter order
        lya_filter_cutoff = 0.25    % Normalized cutoff frequency
    end

    %% ====================================================================
    %              DEPENDENT PROPERTIES
    % =====================================================================
    properties (Dependent)
        alpha                       % Sparsity = indegree/n
        n_E                         % Number of excitatory neurons
        n_I                         % Number of inhibitory neurons
        E_indices                   % Indices of E neurons
        I_indices                   % Indices of I neurons
    end

    %% ====================================================================
    %              PROTECTED PROPERTIES (build outputs)
    % =====================================================================
    properties (SetAccess = protected)
        W                           % Recurrent weight matrix (n × n)
        W_in                        % Input weight matrix (n × n_in)
        t_ex                        % Time vector for stimulus
        u_ex                        % External input matrix
        u_interpolant               % griddedInterpolant for ODE solver
        S0                          % Initial state vector
        is_built = false            % Flag: network is built
        cached_params               % Cached params struct for fast ODE access
    end

    %% ====================================================================
    %              RESULTS PROPERTIES
    % =====================================================================
    properties (SetAccess = protected)
        t_out                       % Time vector from ODE solver
        state_out                   % State trajectory (nt × N_state)
        plot_data                   % Struct with decimated data for plotting
        lya_results                 % Lyapunov analysis results struct
        has_run = false             % Flag: simulation has completed
    end

    %% ====================================================================
    %              DEPENDENT PROPERTY GETTERS
    % =====================================================================
    methods
        function val = get.alpha(obj)
            if isempty(obj.indegree)
                val = 1.0;  % Fully connected
            else
                val = obj.indegree / obj.n;
            end
        end

        function val = get.n_E(obj)
            val = round(obj.f * obj.n);
        end

        function val = get.n_I(obj)
            val = obj.n - obj.n_E;
        end

        function val = get.E_indices(obj)
            val = 1:obj.n_E;
        end

        function val = get.I_indices(obj)
            val = (obj.n_E + 1):obj.n;
        end
    end

    %% ====================================================================
    %              ABSTRACT METHODS
    % =====================================================================
    methods (Abstract)
        rhs = get_rhs(obj, params)
        % GET_RHS Return @(t, S) function handle for ODE integration.
        %
        % The returned function must capture all needed data (params,
        % interpolant, etc.) in its closure. It is called millions of
        % times by the ODE solver — minimize overhead.

        features = get_readout_features(obj)
        % GET_READOUT_FEATURES Return n × T matrix of readout features.
        %
        % For ESN memory capacity: the features used by linear readout.
        % E.g. firing rates [r_E; r_I] for SRNN, raw states x' for LNN.

        J = get_jacobian(obj, S, params)
        % GET_JACOBIAN Return Jacobian matrix at state S.
        %
        % Used by Lyapunov computation (Benettin and QR methods).
        % S is the state vector (N_state × 1).

        initialize_state(obj)
        % INITIALIZE_STATE Set obj.S0 for initial conditions.
    end

    methods (Abstract, Access = protected)
        decimate_and_unpack(obj)
        % DECIMATE_AND_UNPACK Model-specific decimation into plot_data.
    end

    %% ====================================================================
    %              ABSTRACT METHODS (with defaults)
    % =====================================================================
    methods
        function n_state = get_n_state(obj)
            % GET_N_STATE Return total state dimension.
            % Default: n (flat state). Override for packed states (e.g., SRNN).
            n_state = obj.n;
        end

        function bounds = get_state_bounds(obj)
            % GET_STATE_BOUNDS Return [min, max] bounds per state variable.
            % Default: NaN (no bounds). Override for model-specific constraints.
            bounds = nan(obj.get_n_state(), 2);
        end
    end

    %% ====================================================================
    %              PUBLIC LIFECYCLE METHODS
    % =====================================================================
    methods
        function build(obj)
            % BUILD Initialize the network: connectivity, stimulus, state.
            %
            % Delegates to three protected sub-methods:
            %   1. build_network()   - Create W via RMTMatrix
            %   2. build_stimulus()  - Delegate to stimulus strategy
            %   3. finalize_build()  - Validate and cache params

            obj.build_network();
            obj.build_stimulus();
            obj.initialize_state();
            obj.finalize_build();
        end

        function run(obj)
            % RUN Execute the ODE simulation.
            %
            % Integrates the model equations, optionally computes Lyapunov
            % exponents, and decimates results for plotting.

            if ~obj.is_built
                error('cRNN:NotBuilt', 'Model must be built before running. Call build() first.');
            end

            params = obj.cached_params;

            % Set up ODE options
            dt = 1 / obj.fs;
            if isempty(obj.ode_opts)
                obj.ode_opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-8, 'MaxStep', dt);
            end

            % Get RHS via polymorphic dispatch — captures everything in closure
            params.u_interpolant = obj.u_interpolant;
            rhs = obj.get_rhs(params);

            % Integrate
            fprintf('Integrating equations\n');
            tic
            [t_raw, S_raw] = obj.ode_solver(rhs, obj.t_ex, obj.S0, obj.ode_opts);
            integration_time = toc;
            fprintf('Integration complete in %.2f seconds.\n', integration_time);

            % Store results
            obj.t_out = t_raw;
            obj.state_out = S_raw;

            % Compute Lyapunov exponents
            if ~strcmpi(obj.lya_method, 'none')
                obj.compute_lyapunov();
                if obj.filter_local_lya
                    obj.filter_lyapunov();
                end
            end

            % Decimate and unpack for plotting
            if obj.store_decimated_state
                obj.decimate_and_unpack();
            end

            % Clear full state if not storing (free memory)
            if ~obj.store_full_state
                obj.state_out = [];
            end

            obj.has_run = true;
            fprintf('Simulation complete.\n');
        end

        function compute_lyapunov(obj)
            % COMPUTE_LYAPUNOV Compute Lyapunov exponents based on lya_method.

            if isempty(obj.state_out)
                error('cRNN:NoStateData', 'State data not available. Set store_full_state=true or call before clearing.');
            end

            dt = 1 / obj.fs;
            params = obj.cached_params;

            % Set Lyapunov time interval
            if isempty(obj.lya_T_interval)
                if obj.T_range(2) >= 15
                    obj.lya_T_interval = [obj.T_range(1) + 15, obj.T_range(2)];
                else
                    obj.lya_T_interval = [obj.T_range(1), obj.T_range(2)];
                end
            end

            % Get RHS and Jacobian via polymorphic dispatch
            params.u_interpolant = obj.u_interpolant;
            rhs = obj.get_rhs(params);
            jac_wrapper = @(tt, S, p) obj.get_jacobian(S, p);

            fprintf('Computing Lyapunov exponents using %s method\n', obj.lya_method);
            n_state = obj.get_n_state();
            obj.lya_results = cRNN.compute_lyapunov_exponents_internal( ...
                obj.lya_method, obj.state_out, obj.t_out, dt, obj.fs, ...
                obj.lya_T_interval, params, obj.ode_opts, obj.ode_solver, ...
                rhs, obj.t_ex, obj.u_ex, jac_wrapper, n_state, obj);

            if isfield(obj.lya_results, 'LLE')
                fprintf('Largest Lyapunov Exponent: %.4f\n', obj.lya_results.LLE);
            end
        end

        function filter_lyapunov(obj)
            % FILTER_LYAPUNOV Apply lowpass filter to local Lyapunov exponent.

            if isempty(obj.lya_results)
                return;
            end

            Wn = obj.lya_filter_cutoff / (obj.lya_results.lya_fs / 2);
            [b, a] = butter(obj.lya_filter_order, Wn, 'low');

            if isfield(obj.lya_results, 'local_lya') && ~isempty(obj.lya_results.local_lya)
                obj.lya_results.local_lya = filtfilt(b, a, obj.lya_results.local_lya);
            end

            if isfield(obj.lya_results, 'local_LE_spectrum_t') && ~isempty(obj.lya_results.local_LE_spectrum_t)
                for col = 1:size(obj.lya_results.local_LE_spectrum_t, 2)
                    obj.lya_results.local_LE_spectrum_t(:, col) = filtfilt(b, a, obj.lya_results.local_LE_spectrum_t(:, col));
                end
            end
        end

        function params = get_params(obj)
            % GET_PARAMS Return base params struct.
            % Subclasses should call params = get_params@cRNN(obj) then add their own fields.

            params = struct();

            % Network architecture
            params.n = obj.n;
            params.f = obj.f;
            params.indegree = obj.indegree;
            params.alpha = obj.alpha;
            params.level_of_chaos = obj.level_of_chaos;

            % RMT parameters
            params.mu_E_tilde = obj.mu_E_tilde;
            params.mu_I_tilde = obj.mu_I_tilde;
            params.sigma_E_tilde = obj.sigma_E_tilde;
            params.sigma_I_tilde = obj.sigma_I_tilde;
            params.E_W = obj.E_W;

            % E/I indices
            params.n_E = obj.n_E;
            params.n_I = obj.n_I;
            params.E_indices = obj.E_indices;
            params.I_indices = obj.I_indices;

            % Simulation
            params.fs = obj.fs;
            params.rng_seeds = obj.rng_seeds;

            % Activation strategy (stored for use in ODE RHS)
            params.activation = obj.activation;

            % Connection matrix
            if ~isempty(obj.W)
                params.W = obj.W;
            end
            if ~isempty(obj.W_in)
                params.W_in = obj.W_in;
            end
        end

        function clear_results(obj)
            % CLEAR_RESULTS Free memory by clearing stored state data.
            obj.t_out = [];
            obj.state_out = [];
            obj.plot_data = [];
            obj.lya_results = [];
            obj.has_run = false;
            fprintf('Results cleared.\n');
        end

        function reset(obj)
            % RESET Clear built state to allow rebuilding with new parameters.
            obj.is_built = false;
            obj.W = [];
            obj.W_in = [];
            obj.u_interpolant = [];
            obj.t_ex = [];
            obj.u_ex = [];
            obj.S0 = [];
            obj.cached_params = [];
            obj.clear_results();
            fprintf('Model reset.\n');
        end
    end

    %% ====================================================================
    %              PROTECTED BUILD SUB-METHODS
    % =====================================================================
    methods (Access = protected)
        function build_network(obj)
            % BUILD_NETWORK Create W via RMTMatrix.
            %
            % Sets RNG seed, fills in default RMT tilde parameters,
            % creates and scales W. Subclasses can override or call
            % build_network@cRNN() then do additional work.

            rng(obj.rng_seeds(1));

            % Default indegree to fully connected
            if isempty(obj.indegree)
                obj.indegree = obj.n;
            end

            % Compute RMT tilde defaults
            alph = obj.indegree / obj.n;
            F = 1 / sqrt(obj.n * alph * (2 - alph));

            if isempty(obj.mu_E_tilde),    obj.mu_E_tilde = 3*F;     end
            if isempty(obj.mu_I_tilde),    obj.mu_I_tilde = -4*F;    end
            if isempty(obj.sigma_E_tilde), obj.sigma_E_tilde = F;    end
            if isempty(obj.sigma_I_tilde), obj.sigma_I_tilde = F;    end

            % Create W using RMTMatrix
            rmt = RMTMatrix(obj.n);
            rmt.alpha = alph;
            rmt.f = obj.f;
            rmt.mu_tilde_e = obj.mu_E_tilde + obj.E_W;
            rmt.mu_tilde_i = obj.mu_I_tilde + obj.E_W;
            rmt.sigma_tilde_e = obj.sigma_E_tilde;
            rmt.sigma_tilde_i = obj.sigma_I_tilde;
            rmt.zrs_mode = obj.zrs_mode;

            obj.W = obj.level_of_chaos * rmt.W;

            % Report info
            W_eigs = eig(obj.W);
            fprintf('W created: spectral radius = %.3f, abscissa = %.3f\n', ...
                max(abs(W_eigs)), max(real(W_eigs)));
        end

        function build_stimulus(obj)
            % BUILD_STIMULUS Delegate to stimulus strategy object.
            %
            % Calls obj.stimulus.build(), then copies outputs to model.

            if isempty(obj.stimulus)
                error('cRNN:NoStimulus', 'No stimulus strategy set. Assign obj.stimulus before build().');
            end

            obj.stimulus.build(obj.T_range, obj.fs, obj.n, obj.rng_seeds, obj);

            % Copy outputs from strategy to model
            obj.t_ex = obj.stimulus.t_ex;
            obj.u_ex = obj.stimulus.u_ex;
            obj.u_interpolant = obj.stimulus.u_interpolant;

            fprintf('Stimulus ready: %d time points\n', length(obj.t_ex));
        end

        function finalize_build(obj)
            % FINALIZE_BUILD Validate and cache params.

            if obj.n < 1
                error('cRNN:InvalidParams', 'n must be >= 1.');
            end
            if obj.T_range(2) <= obj.T_range(1)
                error('cRNN:InvalidParams', 'T_range(2) must be > T_range(1).');
            end

            % Compute plot_deci if not set
            if isempty(obj.plot_deci)
                obj.plot_deci = max(1, round(obj.fs / obj.plot_freq));
            end

            % Cache params struct for fast access in run
            obj.cached_params = obj.get_params();

            obj.is_built = true;
            fprintf('Model built successfully. Ready to run.\n');
        end
    end

    %% ====================================================================
    %              STATIC LYAPUNOV METHODS
    % =====================================================================
    % These are generic algorithms that work with any continuous-time RNN.
    % They operate on state trajectories, RHS functions, and Jacobians
    % passed as arguments — no model-specific knowledge needed.
    methods (Static)

        function lya_results = compute_lyapunov_exponents_internal(Lya_method, S_out, t_out, dt, fs, T_interval, params, opts, ode_solver, rhs_func, t_ex, u_ex, jac_wrapper, n_state, model_obj)
            % COMPUTE_LYAPUNOV_EXPONENTS_INTERNAL Lyapunov dispatcher.

            lya_results = struct();
            if strcmpi(Lya_method, 'none')
                return;
            end

            if strcmpi(Lya_method, 'qr')
                lya_dt = 0.1;
            elseif strcmpi(Lya_method, 'benettin')
                lya_dt = 0.02;
            else
                lya_dt = 0.1;
            end

            lya_dt_vs_dt_factor = lya_dt / dt;
            if abs(round(lya_dt_vs_dt_factor) - lya_dt_vs_dt_factor) > 1e-11
                error('lya_dt must be a multiple of dt.');
            end
            if lya_dt_vs_dt_factor < 3
                error('lya_dt must be >= 3*dt. Increase fs or increase lya_dt.');
            end

            lya_fs = 1 / lya_dt;

            switch lower(Lya_method)
                case 'benettin'
                    fprintf('Computing LLE using Benettin''s algorithm...\n');
                    d0 = 1e-3;
                    tic
                    bounds = model_obj.get_state_bounds();
                    [LLE, local_lya, finite_lya, t_lya] = cRNN.benettin_algorithm_internal( ...
                        S_out, t_out, dt, fs, d0, T_interval, lya_dt, ...
                        params, opts, rhs_func, t_ex, u_ex, ode_solver, bounds);
                    toc
                    lya_results.LLE = LLE;
                    lya_results.local_lya = local_lya;
                    lya_results.finite_lya = finite_lya;
                    lya_results.t_lya = t_lya;
                    lya_results.lya_dt = lya_dt;
                    lya_results.lya_fs = lya_fs;

                case 'qr'
                    fprintf('Computing full Lyapunov spectrum using QR method...\n');
                    tic
                    [LE_spectrum, local_LE_spectrum_t, finite_LE_spectrum_t, t_lya] = ...
                        cRNN.lyapunov_spectrum_qr_internal( ...
                        S_out, t_out, lya_dt, params, ode_solver, opts, ...
                        jac_wrapper, T_interval, n_state, fs);
                    toc
                    fprintf('Lyapunov Dimension: %.2f\n', cRNN.compute_kaplan_yorke_dimension_internal(LE_spectrum));

                    lya_results.LE_spectrum = LE_spectrum;
                    lya_results.local_LE_spectrum_t = local_LE_spectrum_t;
                    lya_results.finite_LE_spectrum_t = finite_LE_spectrum_t;
                    lya_results.t_lya = t_lya;
                    lya_results.n_state = n_state;

                    [sorted_LE, sort_idx] = sort(real(lya_results.LE_spectrum), 'descend');
                    lya_results.LE_spectrum = sorted_LE;
                    lya_results.local_LE_spectrum_t = lya_results.local_LE_spectrum_t(:, sort_idx);
                    lya_results.finite_LE_spectrum_t = lya_results.finite_LE_spectrum_t(:, sort_idx);
                    lya_results.sort_idx = sort_idx;
                    lya_results.lya_dt = lya_dt;
                    lya_results.lya_fs = lya_fs;
                    fprintf('Largest Lyapunov Exponent (sorted): %.4f\n', lya_results.LE_spectrum(1));

                otherwise
                    error('Unknown Lyapunov method: %s', Lya_method);
            end
        end

        function [LLE, local_lya, finite_lya, t_lya] = benettin_algorithm_internal(X, t, dt, fs, d0, T, lya_dt, params, ode_options, dynamics_func, t_ex, u_ex, ode_solver, state_bounds)
            % BENETTIN_ALGORITHM_INTERNAL Benettin's algorithm for LLE.

            if ~isscalar(lya_dt) || ~isnumeric(lya_dt) || lya_dt <= 0
                error('lya_dt must be a positive scalar.');
            end

            deci_lya = round(lya_dt * fs);
            if deci_lya < 1
                error('lya_dt * fs must yield >= 1 sample.');
            end

            tau_lya = dt * deci_lya;
            t_lya = t(1:deci_lya:end);

            if t_lya(end) + tau_lya > T(2)
                t_lya(end) = [];
            end
            nt_lya = numel(t_lya);

            local_lya = zeros(nt_lya, 1);
            finite_lya = nan(nt_lya, 1);
            sum_log_stretching_factors = 0;

            n_state = size(X, 2);
            rnd_IC = randn(n_state, 1);
            pert = (rnd_IC ./ norm(rnd_IC)) .* d0;

            min_bnds = state_bounds(:, 1);
            max_bnds = state_bounds(:, 2);

            for k = 1:nt_lya
                idx_start = (k - 1) * deci_lya + 1;
                idx_end = idx_start + deci_lya;

                X_start = X(idx_start, :).';
                X_k_pert = X_start + pert;

                idx_violates_min = ~isnan(min_bnds) & (X_k_pert < min_bnds);
                X_k_pert(idx_violates_min) = min_bnds(idx_violates_min);

                idx_violates_max = ~isnan(max_bnds) & (X_k_pert > max_bnds);
                X_k_pert(idx_violates_max) = max_bnds(idx_violates_max);

                t_seg_detailed = t_lya(k) + (0:dt:tau_lya);

                [~, X_pert_output_all_steps] = ode_solver(dynamics_func, t_seg_detailed, X_k_pert, ode_options);

                X_pert_end = X_pert_output_all_steps(end, :).';
                X_end = X(idx_end, :).';

                delta = X_pert_end - X_end;
                d_k = norm(delta);
                local_lya(k) = log(d_k / d0) / tau_lya;

                if ~isfinite(local_lya(k))
                    warning('System diverged at t=%f. Truncating.', t_lya(k));
                    if k > 1
                        last_valid = finite_lya(1:k-1);
                        last_valid = last_valid(~isnan(last_valid));
                        if ~isempty(last_valid)
                            LLE = last_valid(end);
                        else
                            LLE = 0;
                        end
                    else
                        LLE = 0;
                    end
                    local_lya(k:end) = [];
                    finite_lya(k:end) = [];
                    t_lya(k:end) = [];
                    return;
                end

                pert = (delta ./ d_k) .* d0;

                if t_lya(k) >= 0
                    sum_log_stretching_factors = sum_log_stretching_factors + log(d_k / d0);
                    finite_lya(k, 1) = sum_log_stretching_factors / max(t_lya(k) + tau_lya, eps);
                end
            end

            last_valid = finite_lya(~isnan(finite_lya));
            if ~isempty(last_valid)
                LLE = last_valid(end);
            else
                LLE = 0;
            end
        end

        function [LE_spectrum, local_LE_spectrum_t, finite_LE_spectrum_t, t_lya_vec] = lyapunov_spectrum_qr_internal(X_fid_traj, t_fid_traj, lya_dt_interval, params, ode_solver, ode_options_main, jacobian_func_handle, T_full_interval, N_states_sys, fs_fid)
            % LYAPUNOV_SPECTRUM_QR_INTERNAL QR method for full Lyapunov spectrum.

            fiducial_interpolants = cell(N_states_sys, 1);
            for i = 1:N_states_sys
                fiducial_interpolants{i} = griddedInterpolant(t_fid_traj, X_fid_traj(:, i), 'pchip');
            end

            dt_fid = 1 / fs_fid;
            dt_var = lya_dt_interval;
            var_steps_factor = round(dt_var / dt_fid);

            t_analysis_start = T_full_interval(1);
            t_analysis_end = T_full_interval(2);

            t_lya_vec = (t_analysis_start:dt_var:t_analysis_end)';
            if t_lya_vec(end) + dt_var > t_analysis_end + dt_fid/2
                t_lya_vec(end) = [];
            end
            K_steps = length(t_lya_vec);

            if K_steps < 1
                error('Lyapunov interval too short.');
            end

            sum_log_R_diag = zeros(N_states_sys, 1);
            local_LE_spectrum_t = zeros(K_steps, N_states_sys);
            finite_LE_spectrum_t = zeros(K_steps, N_states_sys);
            total_positive_time_accumulated = 0;

            Q_current = eye(N_states_sys);
            MaxStep_var = dt_fid;
            opts_var = odeset('RelTol', 1e-7, 'AbsTol', 1e-7, 'MaxStep', MaxStep_var);

            for k_step = 1:K_steps
                t_seg_start = t_lya_vec(k_step);
                t_seg_end = t_seg_start + dt_var;
                t_span_seg = [t_seg_start, t_seg_end];

                Psi_IC_vec = reshape(Q_current, [], 1);

                var_rhs = @(tt, Psi_vec) cRNN.variational_eqs_ode_internal( ...
                    tt, Psi_vec, fiducial_interpolants, N_states_sys, jacobian_func_handle, params);

                [~, Psi_out] = ode_solver(var_rhs, t_span_seg, Psi_IC_vec, opts_var);

                Psi_end = reshape(Psi_out(end, :), [N_states_sys, N_states_sys]);
                [Q_current, R_matrix] = qr(Psi_end);

                for j = 1:N_states_sys
                    if R_matrix(j, j) < 0
                        Q_current(:, j) = -Q_current(:, j);
                        R_matrix(j, :) = -R_matrix(j, :);
                    end
                end

                log_R_diag = log(abs(diag(R_matrix)));
                local_LE_spectrum_t(k_step, :) = log_R_diag' / dt_var;

                if t_seg_end >= 0
                    sum_log_R_diag = sum_log_R_diag + log_R_diag;
                    total_positive_time_accumulated = total_positive_time_accumulated + dt_var;
                end

                if total_positive_time_accumulated > eps
                    finite_LE_spectrum_t(k_step, :) = (sum_log_R_diag / total_positive_time_accumulated)';
                end
            end

            if total_positive_time_accumulated > eps
                LE_spectrum = sum_log_R_diag / total_positive_time_accumulated;
            else
                warning('No accumulation over positive time for global LEs.');
                LE_spectrum = nan(N_states_sys, 1);
            end
        end

        function dPsi_vec_dt = variational_eqs_ode_internal(tt, current_Psi_vec, fiducial_interpolants, N_states_sys, jacobian_func_handle, params)
            % VARIATIONAL_EQS_ODE_INTERNAL Variational ODE for QR method.
            X_fid_at_tt = zeros(N_states_sys, 1);
            for state_idx = 1:N_states_sys
                X_fid_at_tt(state_idx) = fiducial_interpolants{state_idx}(tt);
            end
            J_matrix = jacobian_func_handle(tt, X_fid_at_tt, params);
            Psi_matrix = reshape(current_Psi_vec, [N_states_sys, N_states_sys]);
            dPsi_matrix_dt = J_matrix * Psi_matrix;
            dPsi_vec_dt = reshape(dPsi_matrix_dt, [], 1);
        end

        function D_KY = compute_kaplan_yorke_dimension_internal(lambda)
            % COMPUTE_KAPLAN_YORKE_DIMENSION_INTERNAL Kaplan-Yorke dimension.
            lambda = sort(lambda, 'descend');
            cumsum_lambda = cumsum(lambda);
            j = find(cumsum_lambda >= 0, 1, 'last');
            if isempty(j)
                D_KY = 0;
            elseif j == length(lambda)
                D_KY = length(lambda);
            else
                D_KY = j + cumsum_lambda(j) / abs(lambda(j + 1));
            end
        end
    end
end
