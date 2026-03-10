classdef LNN < cRNN
    % LNN Liquid Time-Constant Neural Network class.
    %
    % Implements the LTC ODE from Hasani et al. (2021):
    %   dx/dt = -(1./tau + abs(f)) .* x + f .* A
    % where f = activation(W * x + W_in * I(t) + mu)
    %
    % Inherits shared properties and lifecycle from cRNN.
    % Uses strategy objects for activation, stimulus, and connectivity.
    %
    % Usage:
    %   model = LNN('n', 50, 'n_in', 2);
    %   model.build();
    %   model.run();
    %   model.plot();
    %
    % See also: cRNN, SRNNModel2, RMTConnectivity

    %% LNN-Specific Properties
    properties
        n_in = 2                    % Input dimension
        activation_name = 'tanh'    % Legacy name: 'tanh', 'sigmoid', 'piecewise_sigmoid', 'relu'
        readout_mode = 'state'      % ESN readout: 'state', 'nonlinearity', 'full'
    end

    %% LTC Parameter Initialization
    properties
        tau_init = 1.0              % Scalar default for tau vector initialization
        A_init = 0.0                % Scalar default for A vector initialization
        sigma_w = sqrt(2)           % Weight initialization std scale
        sigma_b = 1.0               % Bias initialization std scale
    end

    %% Solver Properties (LNN-specific)
    properties
        solver_mode = 'ode'         % 'ode' or 'fused'
        fused_substeps = 6          % Number of fused solver sub-steps per dt
    end

    %% LNN-Specific Protected Properties
    properties (SetAccess = protected)
        mu                          % Bias vector (n × 1)
        tau                         % Time constant vector (n × 1)
        A                           % Reversal potential vector (n × 1)
    end

    %% Constructor
    methods
        function obj = LNN(varargin)
            % LNN Constructor with name-value pairs.
            %
            % Usage:
            %   model = LNN()                         % All defaults
            %   model = LNN('n', 50, 'n_in', 2)
            %   model = LNN('activation_name', 'sigmoid')

            % Call base class constructor
            obj@cRNN();

            % LNN-specific defaults
            obj.n = 100;
            obj.T_range = [0, 10];
            obj.store_full_state = true;
            obj.lya_method = 'none';
            obj.rng_seeds = [1 1];

            % Set default activation strategy
            obj.activation = TanhActivation();

            % Set default stimulus strategy
            obj.stimulus = SinusoidalStimulus('n_in', 2);

            % Set default connectivity strategy
            obj.connectivity = RMTConnectivity();

            % Parse name-value pairs: two passes.
            % Pass 1: Handle LNN-specific special cases that need side effects
            %         (e.g., rebuilding strategies from legacy names).
            % Pass 2: Delegate remaining args to cRNN.parse_name_value_pairs
            %         which auto-forwards to connectivity/stimulus/activation.

            remaining = {};
            for i = 1:2:length(varargin)
                prop = varargin{i};
                val = varargin{i+1};

                if strcmp(prop, 'n_in')
                    obj.n_in = val;
                    obj.stimulus = SinusoidalStimulus('n_in', val);
                elseif strcmp(prop, 'activation_name')
                    obj.activation_name = val;
                    obj.activation = LNN.make_activation(val);
                elseif strcmp(prop, 'input_func')
                    obj.stimulus = SinusoidalStimulus('n_in', obj.n_in, 'input_func', val);
                else
                    remaining = [remaining, {prop, val}]; %#ok<AGROW>
                end
            end

            % Pass 2: generic parsing (model props + strategy forwarding)
            obj.parse_name_value_pairs(remaining);
        end
    end

    %% Abstract Method Implementations (from cRNN)
    methods
        function rhs = get_rhs(obj, params)
            % GET_RHS Return @(t, x) function handle for LTC dynamics.
            rhs = @(t, x) LNN.dynamics_ltc(t, x, params);
        end

        function features = get_readout_features(obj)
            % GET_READOUT_FEATURES Return readout feature matrix (n_features × T).
            %
            % Output depends on obj.readout_mode:
            %   'state' (default)    — x (n × T)
            %   'nonlinearity'       — f = activation(W*x + W_in*I + mu) (n × T)
            %   'full'               — [x; f] (2n × T)

            if isempty(obj.state_out)
                error('LNN:NoState', 'No state data available.');
            end

            switch obj.readout_mode
                case 'state'
                    features = obj.state_out';  % n × T
                case 'nonlinearity'
                    features = obj.compute_nonlinearity();  % n × T
                case 'full'
                    features = [obj.state_out'; obj.compute_nonlinearity()];  % 2n × T
                otherwise
                    error('LNN:UnknownReadoutMode', ...
                        'Unknown readout_mode: ''%s''. Valid: state, nonlinearity, full', ...
                        obj.readout_mode);
            end
        end

        function J = get_jacobian(obj, S, params)
            % GET_JACOBIAN Return Jacobian of the LTC ODE at state S.
            %
            % For dx/dt = -(1/tau + |f|)*x + f*A
            % where f = activation(W*x + W_in*I + mu)
            %
            % J = -diag(1/tau + |f|) + diag(A.*sign(f) - x.*sign(f)) * diag(f') * W
            %
            % Note: This uses the current stimulus interpolant value.

            x = S;

            % Zero input for Jacobian evaluation (static approximation)
            I_t = zeros(params.n_in, 1);

            % Compute z and f
            z = params.W * x + params.W_in * I_t + params.mu;
            f_val = params.activation.apply(z);
            f_prime = params.activation.derivative(z);
            f_sign = sign(f_val);

            % Build Jacobian
            % d/dx_j of [-(1/tau_i + |f_i|)*x_i + f_i*A_i]
            % Diagonal: -(1/tau_i + |f_i|)
            % Off-diag via chain rule through f_i = act(sum_j W_ij * x_j + ...)
            %   d(f_i)/d(x_j) = f'_i * W_ij
            %   contribution: (-x_i * sign(f_i) + A_i * sign(f_i)) * f'_i * W_ij
            %                 for the |f| term: d|f|/dx_j = sign(f) * f' * W_ij
            %   and for the f*A term: d(f*A)/dx_j = A_i * f'_i * W_ij

            diag_decay = -(1 ./ params.tau + abs(f_val));
            chain_coeff = (params.A - x) .* f_sign .* f_prime;

            J = diag(diag_decay) + diag(chain_coeff) * params.W;
        end

        function initialize_state(obj)
            % INITIALIZE_STATE Set initial conditions for LTC network.
            obj.S0 = 0.01 * randn(obj.n, 1);
        end
    end

    %% Overridden Methods
    methods
        function run(obj)
            % RUN Execute the LTC simulation.
            %
            % Handles both ODE and fused solver modes.

            if ~obj.is_built
                error('LNN:NotBuilt', 'Model must be built before running. Call build() first.');
            end

            if strcmpi(obj.solver_mode, 'fused')
                % Fused semi-implicit Euler solver (LNN-specific)
                params = obj.cached_params;
                fprintf('Running LTC with fused solver (%d substeps per dt)...\n', obj.fused_substeps);
                tic
                [t_raw, x_raw] = obj.run_fused(params);
                run_time = toc;
                fprintf('Integration complete in %.2f seconds.\n', run_time);

                obj.t_out = t_raw;
                obj.state_out = x_raw;

                % Decimate for plotting
                if obj.store_decimated_state
                    obj.decimate_and_unpack();
                end

                if ~obj.store_full_state
                    obj.state_out = [];
                end

                obj.has_run = true;
                fprintf('Simulation complete.\n');
            else
                % Standard ODE mode — delegate to base class
                run@cRNN(obj);
            end
        end

        function params = get_params(obj)
            % GET_PARAMS Return params struct for LTC ODE.
            %
            % Extends cRNN.get_params() with LNN-specific fields.

            params = get_params@cRNN(obj);

            % LNN-specific
            params.n_in = obj.n_in;
            params.activation_name = obj.activation_name;

            % Build outputs (if available)
            if ~isempty(obj.mu),  params.mu = obj.mu;    end
            if ~isempty(obj.tau), params.tau = obj.tau;   end
            if ~isempty(obj.A),   params.A = obj.A;      end
        end

        function [fig_handle, ax_handles] = plot(obj, varargin)
            % PLOT Generate time series plots for LTC simulation.
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
            title('Nonlinearity f(t) = activation(W x + W_{in} I + \mu)');
            ylabel('f');
            xlabel('Time (s)');
            xlim(T_plot_arg);
            grid on;

            linkaxes(ax_handles, 'x');
        end
    end

    %% Private Helper Methods
    methods (Access = protected)
        function f_out = compute_nonlinearity(obj)
            % COMPUTE_NONLINEARITY Compute f = activation(W*x + W_in*I + mu) for all timesteps.
            %
            % Returns n × T matrix.

            params = obj.cached_params;
            T_total = size(obj.state_out, 1);
            f_out = zeros(obj.n, T_total);

            for k = 1:T_total
                I_k = obj.u_interpolant(obj.t_out(k))';
                z = params.W * obj.state_out(k,:)' + params.W_in * I_k + params.mu;
                f_out(:, k) = params.activation.apply(z);
            end
        end
    end

    %% Protected Build Sub-Methods (LNN overrides)
    methods (Access = protected)
        function build_network(obj)
            % BUILD_NETWORK Create W via cRNN, then init W_in, mu, tau, A.

            % Call base class (delegates to connectivity strategy)
            build_network@cRNN(obj);

            % Initialize W_in (Gaussian, n × n_in)
            weight_scale = obj.sigma_w / sqrt(obj.n);
            obj.W_in = weight_scale * randn(obj.n, obj.n_in);

            % Initialize mu (bias vector, n × 1)
            obj.mu = obj.sigma_b * randn(obj.n, 1);

            % Initialize tau (time constant vector, n × 1)
            obj.tau = abs(obj.tau_init * ones(obj.n, 1) + 0.1 * randn(obj.n, 1));
            obj.tau = max(obj.tau, 1e-4);  % Ensure positive

            % Initialize A (reversal potential, n × 1)
            obj.A = obj.A_init * ones(obj.n, 1) + obj.sigma_w * randn(obj.n, 1) / sqrt(obj.n);

            fprintf('Network initialized: n=%d, n_in=%d, activation=%s\n', ...
                obj.n, obj.n_in, obj.activation_name);
        end

        function decimate_and_unpack(obj)
            % DECIMATE_AND_UNPACK Decimate state and compute derived quantities.

            deci = obj.plot_deci;
            t_d = obj.t_out(1:deci:end);
            x_d = obj.state_out(1:deci:end, :);
            nt_d = length(t_d);

            params = obj.cached_params;

            % Compute u, f, tau_sys at decimated times
            n_in = size(params.W_in, 2);  % Actual input dimension from W_in
            u_d = zeros(nt_d, n_in);
            f_d = zeros(nt_d, params.n);
            tau_sys_d = zeros(nt_d, params.n);

            for k = 1:nt_d
                I_k = obj.u_interpolant(t_d(k))';  % (n_in × 1)
                u_d(k, :) = I_k';
                z = params.W * x_d(k, :)' + params.W_in * I_k + params.mu;
                f_k = params.activation.apply(z);
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

        function [t_out_fused, x_out_fused] = run_fused(obj, params)
            % RUN_FUSED Run the fused semi-implicit Euler solver.

            dt = 1 / obj.fs;
            sub_dt = dt / obj.fused_substeps;
            nt = length(obj.t_ex);

            x_out_fused = zeros(nt, obj.n);
            x_out_fused(1, :) = obj.S0';
            x_curr = obj.S0;

            params.u_interpolant = obj.u_interpolant;

            for k = 1:(nt - 1)
                I_k = obj.u_interpolant(obj.t_ex(k))';  % (n_in × 1)
                for s = 1:obj.fused_substeps
                    x_curr = LNN.fused_step(x_curr, I_k, sub_dt, params);
                end
                x_out_fused(k + 1, :) = x_curr';
            end

            t_out_fused = obj.t_ex;
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
            % where f = activation(W * x + W_in * I(t) + mu)

            % Get input at time t via interpolant
            I_t = params.u_interpolant(t)';  % (n_in × 1)

            % Compute nonlinearity
            z = params.W * x + params.W_in * I_t + params.mu;
            f_val = params.activation.apply(z);

            % LTC dynamics
            f_abs = abs(f_val);
            dxdt = -(1 ./ params.tau + f_abs) .* x + f_val .* params.A;
        end

        function x_new = fused_step(x, I, dt, params)
            % FUSED_STEP One step of the fused semi-implicit Euler solver.
            %
            % From Hasani 2021 Eq. 3:
            %   x(t+dt) = (x(t) + dt * f .* A) / (1 + dt * (1/tau + abs(f)))

            z = params.W * x + params.W_in * I + params.mu;
            f_val = params.activation.apply(z);
            f_abs = abs(f_val);
            x_new = (x + dt * f_val .* params.A) ./ (1 + dt * (1 ./ params.tau + f_abs));
        end
    end

    %% ====================================================================
    %              STATIC HELPERS
    % =====================================================================
    methods (Static, Access = protected)
        function cmap = default_colormap(n_neurons)
            % DEFAULT_COLORMAP Generate a colormap for neuron traces.
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

    %% ====================================================================
    %              STATIC FACTORY METHODS
    % =====================================================================
    methods (Static)
        function act = make_activation(name)
            % MAKE_ACTIVATION Create an Activation strategy from a string name.
            switch lower(name)
                case 'tanh'
                    act = TanhActivation();
                case 'sigmoid'
                    act = SigmoidActivation();
                case 'piecewise_sigmoid'
                    act = PiecewiseSigmoid();
                case 'relu'
                    act = ReLUActivation();
                otherwise
                    error('LNN:UnknownActivation', 'Unknown activation: %s', name);
            end
        end
    end
end
