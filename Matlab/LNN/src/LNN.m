classdef LNN < handle
    % LNN Liquid Time-Constant Neural Network class
    %
    % Implements the LTC ODE from Hasani et al. (2021):
    %   dx/dt = -(1./tau + abs(f)) .* x + f .* A
    % where f = activation(W_r * x + W_in * I(t) + mu)
    %
    % Standalone class — only external dependency is RMTMatrix.m.
    %
    % Usage:
    %   model = LNN('n', 50, 'n_in', 2);
    %   model.build();
    %   model.run();
    %   model.plot();
    %
    % See also: RMTMatrix, SRNNModel2

    %% Network Architecture Properties
    properties
        n = 100                     % Number of hidden neurons
        n_in = 2                    % Input dimension
        activation = 'tanh'         % 'tanh', 'sigmoid', 'piecewise_sigmoid', 'relu'
    end

    %% LTC Parameter Defaults
    properties
        tau_init = 1.0              % Scalar default for tau vector initialization
        A_init = 0.0                % Scalar default for A vector initialization
        sigma_w = sqrt(2)           % Weight initialization std scale
        sigma_b = 1.0               % Bias initialization std scale
    end

    %% Piecewise Sigmoid Parameters (only used when activation = 'piecewise_sigmoid')
    properties
        S_a = 0.9                   % Piecewise sigmoid linear fraction parameter
        S_c = 0.35                  % Piecewise sigmoid center parameter
    end

    %% RMT Connectivity Properties (for W_r)
    properties
        f = 0.5                     % Fraction of excitatory neurons in RMTMatrix
        indegree                    % Expected in-degree (default: fully connected = n)
        mu_E_tilde                  % RMT excitatory mean (default: 3*F)
        mu_I_tilde                  % RMT inhibitory mean (default: -4*F)
        sigma_E_tilde               % RMT excitatory std (default: F)
        sigma_I_tilde               % RMT inhibitory std (default: F)
        level_of_chaos = 1.0        % Spectral radius scaling factor
        zrs_mode = 'none'           % ZRS mode: 'none', 'ZRS', 'SZRS', 'Partial_SZRS'
    end

    %% Simulation Settings Properties
    properties
        fs = 400                    % Sampling frequency (Hz); MaxStep = 1/fs
        T_range = [0, 10]           % Simulation time interval [start, end]
        T_plot                      % Plotting time interval (defaults to T_range)
        ode_solver = @ode45         % ODE solver function handle (public, settable)
        ode_opts                    % ODE options (auto-set from fs if empty)
        solver_mode = 'ode'         % 'ode' or 'fused'
        fused_substeps = 6          % Number of fused solver sub-steps per dt
        rng_seed = 1                % RNG seed for reproducibility
    end

    %% Storage and Plotting Properties
    properties
        store_full_state = true     % Whether to keep full x_out in memory
        plot_deci                   % Decimation factor for plotting
        plot_freq = 10              % Target plotting frequency (Hz)
    end

    %% Input Properties
    properties
        input_func                  % Function handle @(t) -> (n_in x 1) for custom input
        u_ex                        % Stored input matrix (n_in x nt), set in build
        t_ex                        % Time vector, set in build
    end

    %% Dependent Properties
    properties (Dependent)
        alpha                       % Sparsity = indegree/n
    end

    %% Protected Properties (build outputs)
    properties (SetAccess = protected)
        W_r                         % Recurrent weight matrix (n x n)
        W_in                        % Input weight matrix (n x n_in)
        mu                          % Bias vector (n x 1)
        tau                         % Time constant vector (n x 1)
        A                           % Reversal potential / bias target vector (n x 1)
        S0                          % Initial state vector (n x 1)
        u_interpolant               % griddedInterpolant for input
        is_built = false            % Flag: network is built
        cached_params               % Cached params struct for fast ODE access
    end

    %% Results Properties
    properties (SetAccess = protected)
        t_out                       % Time vector from solver
        x_out                       % State trajectory (nt x n)
        plot_data                   % Struct with decimated data for plotting
        has_run = false             % Flag: simulation has completed
    end

    %% Constructor
    methods
        function obj = LNN(varargin)
            % LNN Constructor with name-value pairs
            %
            % Usage:
            %   model = LNN()                        % All defaults
            %   model = LNN('n', 50, 'n_in', 2)
            %   model = LNN('activation', 'sigmoid', 'level_of_chaos', 1.5)

            % Parse name-value pairs
            for i = 1:2:length(varargin)
                if isprop(obj, varargin{i})
                    obj.(varargin{i}) = varargin{i+1};
                else
                    warning('LNN:UnknownProperty', 'Unknown property: %s', varargin{i});
                end
            end

            % Compute plot_deci from fs and plot_freq if not explicitly set
            if isempty(obj.plot_deci)
                obj.plot_deci = max(1, round(obj.fs / obj.plot_freq));
            end
        end
    end

    %% Dependent Property Getters
    methods
        function val = get.alpha(obj)
            if isempty(obj.indegree)
                val = 1.0;  % Fully connected
            else
                val = obj.indegree / obj.n;
            end
        end
    end

    %% Public Methods
    methods
        function build(obj)
            % BUILD Initialize the network: create W_r, W_in, mu, tau, A, stimulus
            %
            % Delegates to three protected sub-methods (overridable by subclasses):
            %   1. build_network()   - Create W_r, W_in, mu, tau, A
            %   2. build_stimulus()  - Generate input, interpolant, S0
            %   3. finalize_build()  - Validate and cache params

            obj.build_network();
            obj.build_stimulus();
            obj.finalize_build();
        end

        function run(obj)
            % RUN Execute the LTC simulation
            %
            % Integrates the LTC ODE using ode_solver (default ode45) or
            % the fused semi-implicit Euler solver.

            if ~obj.is_built
                error('LNN:NotBuilt', 'Model must be built before running. Call build() first.');
            end

            params = obj.cached_params;

            % Set up ODE options if not provided
            dt = 1 / obj.fs;
            if isempty(obj.ode_opts)
                obj.ode_opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-8, 'MaxStep', dt);
            end

            if strcmpi(obj.solver_mode, 'fused')
                % Fused semi-implicit Euler solver
                fprintf('Running LTC with fused solver (%d substeps per dt)...\n', obj.fused_substeps);
                tic
                [t_raw, x_raw] = obj.run_fused(params);
                run_time = toc;
            else
                % ODE solver mode
                % Cache interpolant into params for closure
                params.u_interpolant = obj.u_interpolant;
                rhs = @(t, x) LNN.dynamics_ltc(t, x, params);

                fprintf('Integrating LTC equations with %s...\n', func2str(obj.ode_solver));
                tic
                [t_raw, x_raw] = obj.ode_solver(rhs, obj.t_ex, obj.S0, obj.ode_opts);
                run_time = toc;

                % Verify output times match input times
                if length(t_raw) ~= length(obj.t_ex) || max(abs(t_raw - obj.t_ex)) > 1e-9
                    warning('LNN:TimeMismatch', 'ODE solver output times differ from input. Max diff: %.2e', max(abs(t_raw(:) - obj.t_ex(:))));
                end
            end

            fprintf('Integration complete in %.2f seconds.\n', run_time);

            obj.t_out = t_raw;
            obj.x_out = x_raw;

            % Decimate for plotting
            obj.decimate_and_unpack();

            obj.has_run = true;
            fprintf('Simulation complete.\n');
        end

        function [fig_handle, ax_handles] = plot(obj, varargin)
            % PLOT Generate time series plots for LTC simulation
            %
            % Usage:
            %   model.plot()
            %   model.plot('T_plot', [2, 8])
            %   [fig, axes] = model.plot()

            if ~obj.has_run
                error('LNN:NotRun', 'Model must be run before plotting. Call run() first.');
            end

            if isempty(obj.plot_data)
                error('LNN:NoPlotData', 'Plot data not available.');
            end

            % Parse optional arguments
            T_plot_arg = obj.T_plot;
            for i = 1:2:length(varargin)
                if strcmpi(varargin{i}, 'T_plot')
                    T_plot_arg = varargin{i+1};
                end
            end
            if isempty(T_plot_arg)
                T_plot_arg = obj.T_range;
            end

            pd = obj.plot_data;
            params = obj.cached_params;

            fig_handle = figure('Position', [100, 100, 1000, 800], 'Name', 'LNN Simulation');
            ax_handles = gobjects(4, 1);

            % Panel 1: Input
            ax_handles(1) = subplot(4, 1, 1);
            plot(pd.t, pd.u');
            title('External Input I(t)');
            ylabel('Amplitude');
            xlim(T_plot_arg);
            grid on;

            % Panel 2: Neuron states x(t)
            ax_handles(2) = subplot(4, 1, 2);
            cmap = LNN.default_colormap(params.n);
            hold on;
            for i = 1:params.n
                plot(pd.t, pd.x(:, i), 'Color', cmap(i, :), 'LineWidth', 0.5);
            end
            hold off;
            title('Neuron States x(t)');
            ylabel('State');
            xlim(T_plot_arg);
            grid on;

            % Panel 3: Effective tau_sys
            ax_handles(3) = subplot(4, 1, 3);
            hold on;
            for i = 1:params.n
                plot(pd.t, pd.tau_sys(:, i), 'Color', cmap(i, :), 'LineWidth', 0.5);
            end
            hold off;
            title('Effective \tau_{sys}(t) = \tau / (1 + \tau \cdot |f|)');
            ylabel('\tau_{sys}');
            xlim(T_plot_arg);
            grid on;

            % Panel 4: f(t) values
            ax_handles(4) = subplot(4, 1, 4);
            hold on;
            for i = 1:params.n
                plot(pd.t, pd.f_vals(:, i), 'Color', cmap(i, :), 'LineWidth', 0.5);
            end
            hold off;
            title('Nonlinearity f(t) = activation(W_r x + W_{in} I + \mu)');
            ylabel('f');
            xlabel('Time (s)');
            xlim(T_plot_arg);
            grid on;

            linkaxes(ax_handles, 'x');
        end

        function params = get_params(obj)
            % GET_PARAMS Return params struct for ODE and analysis

            params = struct();
            params.n = obj.n;
            params.n_in = obj.n_in;
            params.activation = obj.activation;
            params.S_a = obj.S_a;
            params.S_c = obj.S_c;
            params.f_frac = obj.f;
            params.level_of_chaos = obj.level_of_chaos;
            params.fs = obj.fs;
            params.rng_seed = obj.rng_seed;

            % Build outputs (if available)
            if ~isempty(obj.W_r),  params.W_r = obj.W_r;   end
            if ~isempty(obj.W_in), params.W_in = obj.W_in;  end
            if ~isempty(obj.mu),   params.mu = obj.mu;      end
            if ~isempty(obj.tau),  params.tau = obj.tau;     end
            if ~isempty(obj.A),    params.A = obj.A;        end
        end

        function clear_results(obj)
            % CLEAR_RESULTS Free memory by clearing stored state data
            obj.t_out = [];
            obj.x_out = [];
            obj.plot_data = [];
            obj.has_run = false;
            fprintf('Results cleared.\n');
        end

        function reset(obj)
            % RESET Clear built state to allow rebuilding with new parameters
            obj.is_built = false;
            obj.W_r = [];
            obj.W_in = [];
            obj.mu = [];
            obj.tau = [];
            obj.A = [];
            obj.S0 = [];
            obj.u_interpolant = [];
            obj.t_ex = [];
            obj.u_ex = [];
            obj.cached_params = [];
            obj.clear_results();
            fprintf('Model reset.\n');
        end
    end

    %% Protected Build Sub-Methods (overridable by subclasses)
    methods (Access = protected)
        function build_network(obj)
            % BUILD_NETWORK Create W_r via RMTMatrix, initialize W_in, mu, tau, A

            rng(obj.rng_seed);

            %% Compute RMT defaults
            % F = 1/sqrt(N*alpha*(2-alpha))
            if isempty(obj.indegree)
                obj.indegree = obj.n;  % Fully connected by default
            end
            alph = obj.indegree / obj.n;
            F = 1 / sqrt(obj.n * alph * (2 - alph));

            if isempty(obj.mu_E_tilde),    obj.mu_E_tilde = 3*F;     end
            if isempty(obj.mu_I_tilde),    obj.mu_I_tilde = -4*F;    end
            if isempty(obj.sigma_E_tilde), obj.sigma_E_tilde = F;    end
            if isempty(obj.sigma_I_tilde), obj.sigma_I_tilde = F;    end

            %% Create W_r using RMTMatrix
            rmt = RMTMatrix(obj.n);
            rmt.alpha = alph;
            rmt.f = obj.f;
            rmt.mu_tilde_e = obj.mu_E_tilde;
            rmt.mu_tilde_i = obj.mu_I_tilde;
            rmt.sigma_tilde_e = obj.sigma_E_tilde;
            rmt.sigma_tilde_i = obj.sigma_I_tilde;
            rmt.zrs_mode = obj.zrs_mode;

            obj.W_r = obj.level_of_chaos * rmt.W;

            % Report spectral radius
            W_eigs = eig(obj.W_r);
            fprintf('W_r created: spectral radius = %.3f, abscissa = %.3f\n', ...
                max(abs(W_eigs)), max(real(W_eigs)));

            %% Initialize W_in (Gaussian, n x n_in)
            weight_scale = obj.sigma_w / sqrt(obj.n);
            obj.W_in = weight_scale * randn(obj.n, obj.n_in);

            %% Initialize mu (bias vector, n x 1)
            obj.mu = obj.sigma_b * randn(obj.n, 1);

            %% Initialize tau (time constant vector, n x 1)
            obj.tau = abs(obj.tau_init * ones(obj.n, 1) + 0.1 * randn(obj.n, 1));
            obj.tau = max(obj.tau, 1e-4);  % Ensure positive

            %% Initialize A (reversal potential / bias target, n x 1)
            obj.A = obj.A_init * ones(obj.n, 1) + obj.sigma_w * randn(obj.n, 1) / sqrt(obj.n);

            fprintf('Network initialized: n=%d, n_in=%d, activation=%s\n', obj.n, obj.n_in, obj.activation);
        end

        function build_stimulus(obj)
            % BUILD_STIMULUS Generate input, interpolant, and initial state
            %
            % If input_func is provided, uses it. Otherwise generates
            % a default sinusoidal circular trajectory.

            dt = 1 / obj.fs;
            obj.t_ex = (obj.T_range(1):dt:obj.T_range(2))';

            if ~isempty(obj.input_func)
                % Use custom input function
                nt = length(obj.t_ex);
                obj.u_ex = zeros(obj.n_in, nt);
                for k = 1:nt
                    obj.u_ex(:, k) = obj.input_func(obj.t_ex(k));
                end
            elseif isempty(obj.u_ex)
                % Default: sinusoidal circular trajectory
                T_dur = obj.T_range(2) - obj.T_range(1);
                freq = 1.0;  % 1 Hz
                if obj.n_in >= 2
                    obj.u_ex = zeros(obj.n_in, length(obj.t_ex));
                    obj.u_ex(1, :) = sin(2 * pi * freq * obj.t_ex');
                    obj.u_ex(2, :) = cos(2 * pi * freq * obj.t_ex');
                    % Remaining inputs are zero
                elseif obj.n_in == 1
                    obj.u_ex = sin(2 * pi * freq * obj.t_ex');
                end
            end

            % Build interpolant for input
            % u_ex is (n_in x nt), interpolant needs (nt x n_in)
            obj.u_interpolant = griddedInterpolant(obj.t_ex, obj.u_ex', 'linear', 'nearest');

            % Initialize state vector (small random perturbation)
            obj.S0 = 0.01 * randn(obj.n, 1);

            fprintf('Stimulus generated: T=[%.1f, %.1f] s, %d time steps\n', ...
                obj.T_range(1), obj.T_range(2), length(obj.t_ex));
        end

        function finalize_build(obj)
            % FINALIZE_BUILD Validate and cache params

            % Validate
            if obj.n < 1
                error('LNN:InvalidParams', 'n must be >= 1.');
            end
            if obj.n_in < 1
                error('LNN:InvalidParams', 'n_in must be >= 1.');
            end
            if obj.T_range(2) <= obj.T_range(1)
                error('LNN:InvalidParams', 'T_range(2) must be > T_range(1).');
            end
            valid_activations = {'tanh', 'sigmoid', 'piecewise_sigmoid', 'relu'};
            if ~ismember(obj.activation, valid_activations)
                error('LNN:InvalidParams', 'Unknown activation: %s. Valid: %s', ...
                    obj.activation, strjoin(valid_activations, ', '));
            end

            % Cache params struct
            obj.cached_params = obj.get_params();

            obj.is_built = true;
            fprintf('Model built successfully. Ready to run.\n');
        end

        function decimate_and_unpack(obj)
            % DECIMATE_AND_UNPACK Decimate state and compute derived quantities

            deci = obj.plot_deci;
            t_d = obj.t_out(1:deci:end);
            x_d = obj.x_out(1:deci:end, :);
            nt_d = length(t_d);

            params = obj.cached_params;

            % Compute u, f, tau_sys at decimated times
            u_d = zeros(nt_d, params.n_in);
            f_d = zeros(nt_d, params.n);
            tau_sys_d = zeros(nt_d, params.n);

            for k = 1:nt_d
                I_k = obj.u_interpolant(t_d(k))';  % (n_in x 1)
                u_d(k, :) = I_k';
                z = params.W_r * x_d(k, :)' + params.W_in * I_k + params.mu;
                f_k = LNN.apply_activation(z, params.activation, params);
                f_d(k, :) = f_k';
                tau_sys_d(k, :) = (params.tau ./ (1 + params.tau .* abs(f_k)))';
            end

            obj.plot_data = struct();
            obj.plot_data.t = t_d;
            obj.plot_data.x = x_d;
            obj.plot_data.u = u_d;
            obj.plot_data.f_vals = f_d;
            obj.plot_data.tau_sys = tau_sys_d;
        end

        function [t_out, x_out] = run_fused(obj, params)
            % RUN_FUSED Run the fused semi-implicit Euler solver

            dt = 1 / obj.fs;
            sub_dt = dt / obj.fused_substeps;
            nt = length(obj.t_ex);

            x_out = zeros(nt, obj.n);
            x_out(1, :) = obj.S0';
            x_curr = obj.S0;

            params.u_interpolant = obj.u_interpolant;

            for k = 1:(nt - 1)
                I_k = obj.u_interpolant(obj.t_ex(k))';  % (n_in x 1)
                for s = 1:obj.fused_substeps
                    x_curr = LNN.fused_step(x_curr, I_k, sub_dt, params);
                end
                x_out(k + 1, :) = x_curr';
            end

            t_out = obj.t_ex;
        end
    end

    %% ====================================================================
    %              STATIC ODE RHS AND FUSED SOLVER
    % =====================================================================

    methods (Static)
        function dxdt = dynamics_ltc(t, x, params)
            % DYNAMICS_LTC ODE right-hand side for the LTC network.
            %
            % Implements: dx/dt = -(1./tau + abs(f)) .* x + f .* A
            % where f = activation(W_r * x + W_in * I(t) + mu)
            %
            % Following Hasani's ltc_def.m convention: abs(f) in decay,
            % signed f in drive.

            % Get input at time t via interpolant
            I_t = params.u_interpolant(t)';  % (n_in x 1)

            % Compute nonlinearity
            z = params.W_r * x + params.W_in * I_t + params.mu;
            f_val = LNN.apply_activation(z, params.activation, params);

            % LTC dynamics
            f_abs = abs(f_val);
            dxdt = -(1 ./ params.tau + f_abs) .* x + f_val .* params.A;
        end

        function x_new = fused_step(x, I, dt, params)
            % FUSED_STEP One step of the fused semi-implicit Euler solver.
            %
            % From Hasani 2021 Eq. 3:
            %   x(t+dt) = (x(t) + dt * f .* A) / (1 + dt * (1/tau + abs(f)))
            %
            % abs(f) in denominator (decay), signed f in numerator (drive).

            z = params.W_r * x + params.W_in * I + params.mu;
            f_val = LNN.apply_activation(z, params.activation, params);
            f_abs = abs(f_val);
            x_new = (x + dt * f_val .* params.A) ./ (1 + dt * (1 ./ params.tau + f_abs));
        end
    end

    %% ====================================================================
    %              INTERNALIZED ACTIVATION FUNCTIONS
    % =====================================================================

    methods (Static)
        function y = apply_activation(z, activation_name, params)
            % APPLY_ACTIVATION Dispatch to the appropriate activation function.

            switch activation_name
                case 'tanh'
                    y = LNN.tanhActivation(z);
                case 'sigmoid'
                    y = LNN.logisticSigmoid(z);
                case 'piecewise_sigmoid'
                    y = LNN.piecewiseSigmoid(z, params.S_a, params.S_c);
                case 'relu'
                    y = LNN.reluActivation(z);
                otherwise
                    error('LNN:UnknownActivation', 'Unknown activation: %s', activation_name);
            end
        end

        function y = tanhActivation(x)
            % TANHACTIVATION Hyperbolic tangent activation function.
            y = tanh(x);
        end

        function y = logisticSigmoid(x)
            % LOGISTICSIGMOID Standard logistic sigmoid: 1/(1+exp(-x)).
            y = 1 ./ (1 + exp(-x));
        end

        function y = reluActivation(x)
            % RELUACTIVATION Rectified linear unit: max(0, x).
            y = max(0, x);
        end

        function y = piecewiseSigmoid(x, a, c)
            % PIECEWISESIGMOID Hard sigmoid with rounded (quadratic) corners.
            %
            % Identical to SRNNModel2.piecewiseSigmoid. A piecewise
            % linear/quadratic sigmoid bounded in [0, 1].
            %
            % Parameters:
            %   x - input values
            %   a - linear fraction parameter (0 to 1)
            %   c - center/shift parameter

            if a < 0 || a > 1
                error('Parameter "a" must be between 0 and 1.');
            end
            a = a / 2;

            if a == 0.5
                y_linear = (x - c) + 0.5;
                y = min(max(y_linear, 0), 1);
            else
                y = zeros(size(x));
                k = 0.5 / (1 - 2*a);
                x1 = c + a - 1;
                x2 = c - a;
                x3 = c + a;
                x4 = c + 1 - a;

                mask_left_quad = (x >= x1) & (x < x2);
                mask_linear = (x >= x2) & (x <= x3);
                mask_right_quad = (x > x3) & (x <= x4);
                mask_right_sat = (x > x4);

                if any(mask_left_quad, 'all')
                    y(mask_left_quad) = k * (x(mask_left_quad) - x1).^2;
                end
                if any(mask_linear, 'all')
                    y(mask_linear) = (x(mask_linear) - c) + 0.5;
                end
                if any(mask_right_quad, 'all')
                    y(mask_right_quad) = 1 - k * (x(mask_right_quad) - x4).^2;
                end
                if any(mask_right_sat, 'all')
                    y(mask_right_sat) = 1;
                end
            end
        end
    end

    %% ====================================================================
    %              INTERNALIZED PLOTTING HELPERS
    % =====================================================================

    methods (Static, Access = protected)
        function cmap = default_colormap(n_neurons)
            % DEFAULT_COLORMAP Generate a colormap for neuron traces.
            %
            % Uses hsv-based colormap for visual distinction.

            if n_neurons <= 8
                cmap = lines(n_neurons);
            else
                hues = linspace(0, 0.85, n_neurons)';
                sats = 0.7 * ones(n_neurons, 1);
                vals = 0.8 * ones(n_neurons, 1);
                cmap = hsv2rgb([hues, sats, vals]);
            end
        end
    end
end
