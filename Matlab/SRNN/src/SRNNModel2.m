classdef SRNNModel2 < cRNN
    % SRNNMODEL2 Stable Recurrent Neural Network Model.
    %
    % Implements a continuous-time rate network with spike-frequency
    % adaptation (SFA) and short-term depression (STD).
    % Inherits shared properties and lifecycle from cRNN.
    %
    % Usage:
    %   model = SRNNModel2('n_a_E', 3, 'n_b_E', 1);
    %   model.build();
    %   model.run();
    %   model.plot();
    %
    % See also: cRNN, LNN, SRNN_ESN_reservoir

    %% Spike-Frequency Adaptation (SFA) Properties
    properties
        n_a_E = 0                   % Number of adaptation timescales for E neurons
        n_a_I = 0                   % Number of adaptation timescales for I neurons
        tau_a_E                     % Adaptation time constants for E neurons (1 x n_a_E)
        tau_a_I                     % Adaptation time constants for I neurons (1 x n_a_I)
        c_E = 0.15/3                % Adaptation scaling for E neurons
        c_I = 0.15/3                % Adaptation scaling for I neurons
    end

    %% Short-Term Depression (STD) Properties
    properties
        n_b_E = 0                   % Number of STD timescales for E neurons (0 or 1)
        n_b_I = 0                   % Number of STD timescales for I neurons (0 or 1)
        tau_b_E_rec = 1             % STD recovery time constant for E neurons (s)
        tau_b_E_rel = 0.25          % STD release time constant for E neurons (s)
        tau_b_I_rec = 1             % STD recovery time constant for I neurons (s)
        tau_b_I_rel = 0.25          % STD release time constant for I neurons (s)
    end

    %% SRNN Dynamics Properties
    properties
        tau_d = 0.1                 % Dendritic time constant (s)
        activation_function         % Activation function handle (legacy, built from obj.activation)
        activation_function_derivative  % Derivative handle (legacy, built from obj.activation)
    end

    %% Input Configuration Properties
    properties
        input_config                % Struct with stimulus parameters (legacy)
        reps = 1                    % Repetition index
        readout_mode = 'firing_rate' % ESN readout: 'firing_rate', 'state', 'full_state', 'synaptic'
    end


    %% SRNN Dependent Properties
    properties (Dependent)
        R                   % Theoretical spectral radius (from connectivity strategy)
        N_sys_eqs           % Total state dimension
    end

    %% Constructor
    methods
        function obj = SRNNModel2(varargin)
            % SRNNMODEL2 Constructor with name-value pairs
            %
            % Usage:
            %   model = SRNNModel2()  % All defaults (n=300, indegree=100)
            %   model = SRNNModel2('n', 200, 'level_of_chaos', 2.0)
            %   model = SRNNModel2('n_a_E', 3, 'n_b_E', 1)

            % Call base class constructor
            obj@cRNN();

            % SRNN-specific defaults
            obj.n = 300;
            obj.indegree = 100;
            obj.T_range = [0, 50];
            obj.lya_method = 'benettin';

            % Set default activation strategy
            obj.activation = PiecewiseSigmoid('S_a', 0.9, 'S_c', 0.35);

            % Set default stimulus strategy
            obj.stimulus = StepStimulus();

            % Set default connectivity strategy (RMT with SRNN defaults)
            obj.connectivity = RMTConnectivity();

            % Build legacy function handles from activation strategy
            obj.activation_function = @(x) obj.activation.apply(x);
            obj.activation_function_derivative = @(x) obj.activation.derivative(x);

            % Parse name-value pairs (delegates to cRNN.parse_name_value_pairs)
            obj.parse_name_value_pairs(varargin);
        end
    end

    %% SRNN Dependent Property Getters
    methods

        function val = get.R(obj)
            % R Theoretical spectral radius — delegates to connectivity strategy.
            if ~isempty(obj.connectivity) && isprop(obj.connectivity, 'R')
                val = obj.connectivity.R;
            else
                val = NaN;
            end
        end

        function val = get.N_sys_eqs(obj)
            nE = obj.n_E;
            nI = obj.n_I;
            val = nE * obj.n_a_E + nI * obj.n_a_I + nE * obj.n_b_E + nI * obj.n_b_I + obj.n;
        end
    end

    %% Abstract Method Implementations (from cRNN)
    methods
        function rhs = get_rhs(obj, params)
            % GET_RHS Return @(t, S) function handle for SRNN dynamics.
            rhs = @(t, S) SRNNModel2.dynamics_fast(t, S, params);
        end

        function features = get_readout_features(obj)
            % GET_READOUT_FEATURES Return readout feature matrix (n_features × T).
            %
            % Output depends on obj.readout_mode:
            %   'firing_rate' (default) — [r_E; r_I] (n × T)
            %   'state'                 — [x_E; x_I] (n × T)
            %   'full_state'            — full state vector (N_sys_eqs × T)
            %   'synaptic'              — [br_E; br_I] (n × T)

            if isempty(obj.state_out)
                error('SRNNModel2:NoState', 'No state data available.');
            end

            switch obj.readout_mode
                case 'firing_rate'
                    params = obj.cached_params;
                    [~, ~, ~, r, ~] = obj.unpack_and_compute_states(obj.state_out, params);
                    features = [r.E; r.I];  % n × T
                case 'state'
                    params = obj.cached_params;
                    [x, ~, ~, ~, ~] = obj.unpack_and_compute_states(obj.state_out, params);
                    features = [x.E; x.I];  % n × T
                case 'full_state'
                    features = obj.state_out';  % N_sys_eqs × T
                case 'synaptic'
                    params = obj.cached_params;
                    [~, ~, ~, ~, br] = obj.unpack_and_compute_states(obj.state_out, params);
                    features = [br.E; br.I];  % n × T
                otherwise
                    error('SRNNModel2:UnknownReadoutMode', ...
                        'Unknown readout_mode: ''%s''. Valid: firing_rate, state, full_state, synaptic', ...
                        obj.readout_mode);
            end
        end

        function J = get_jacobian(obj, S, params)
            % GET_JACOBIAN Return Jacobian matrix for SRNN at state S.
            J = SRNNModel2.compute_Jacobian_fast(S, params);
        end

        function initialize_state(obj)
            % INITIALIZE_STATE Set initial state vector for SRNN.
            %
            % State layout: S0 = [a_E(:); a_I(:); b_E(:); b_I(:); x(:)]

            params = obj.get_params();

            % Adaptation states
            a0_E = [];
            if params.n_a_E > 0
                a0_E = zeros(params.n_E * params.n_a_E, 1);
            end
            a0_I = [];
            if params.n_a_I > 0
                a0_I = zeros(params.n_I * params.n_a_I, 1);
            end

            % STD states (start at 1 = no depression)
            b0_E = [];
            if params.n_b_E > 0
                b0_E = ones(params.n_E * params.n_b_E, 1);
            end
            b0_I = [];
            if params.n_b_I > 0
                b0_I = ones(params.n_I * params.n_b_I, 1);
            end

            % Dendritic states
            x0 = 0.1 .* randn(params.n, 1);

            obj.S0 = [a0_E; a0_I; b0_E; b0_I; x0];
        end

        function n_state = get_n_state(obj)
            % GET_N_STATE Return total SRNN state dimension (packed).
            n_state = obj.N_sys_eqs;
        end

        function bounds = get_state_bounds(obj)
            % GET_STATE_BOUNDS Return NaN bounds for SRNN state variables.
            bounds = nan(obj.N_sys_eqs, 2);
        end


        function [fig_handle, ax_handles] = plot(obj, varargin)
            % PLOT Generate time series plots for SRNN simulation
            %
            % Usage:
            %   model.plot()
            %   model.plot('T_plot', [10, 40])
            %   [fig, axes] = model.plot()

            if ~obj.has_run
                error('SRNNModel:NotRun', 'Model must be run before plotting. Call run() first.');
            end

            if isempty(obj.plot_data)
                error('SRNNModel:NoPlotData', 'Plot data not available. Set store_decimated_state=true.');
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

            params = obj.cached_params;

            [fig_handle, ax_handles] = SRNNModel2.plot_SRNN_tseries(obj.plot_data.t, obj.plot_data.u, obj.plot_data.x, obj.plot_data.r, obj.plot_data.a, obj.plot_data.b, obj.plot_data.br, params, obj.lya_results, obj.lya_method, T_plot_arg);
        end

        function [fig_handle, ax_handles] = plot_eigenvalues(obj, J_times_sec)
            % PLOT_EIGENVALUES Plot Jacobian eigenvalues at specified times
            %
            % Usage:
            %   model.plot_eigenvalues([5, 10, 15])  % Times in seconds

            if isempty(obj.state_out)
                error('SRNNModel:NoStateData', 'State data required. Set store_full_state=true.');
            end

            params = obj.cached_params;

            % Convert times to indices
            J_times = round((J_times_sec - obj.t_out(1)) * obj.fs) + 1;
            J_times = unique(max(1, min(J_times, size(obj.state_out, 1))));

            fprintf('Computing Jacobian at %d time points\n', length(J_times));
            J_array = SRNNModel2.compute_Jacobian_at_indices(obj.state_out, J_times, params);

            % Compute eigenvalues
            n_plots = length(J_times);
            eigenvalues_all = cell(n_plots, 1);
            for i = 1:n_plots
                eigenvalues_all{i} = eig(J_array(:,:,i));
            end

            % Determine subplot layout
            if n_plots <= 4
                n_rows = 1;
                n_cols = n_plots;
            else
                n_cols = ceil(sqrt(n_plots));
                n_rows = ceil(n_plots / n_cols);
            end

            % Compute global axis limits
            all_real = [];
            all_imag = [];
            for i = 1:n_plots
                evals = eigenvalues_all{i};
                all_real = [all_real; real(evals)];
                all_imag = [all_imag; imag(evals)];
            end
            global_xlim = [min(all_real), max(all_real)];
            global_ylim = [min(all_imag), max(all_imag)];
            x_range = diff(global_xlim);
            y_range = diff(global_ylim);
            global_xlim = global_xlim + [-0.1, 0.1] * x_range;
            global_ylim = global_ylim + [-0.1, 0.1] * y_range;

            % Create figure
            fig_handle = figure('Position', [1312, 526, 600, 360]);
            ax_handles = zeros(n_plots, 1);

            for i = 1:n_plots
                ax = subplot(n_rows, n_cols, i);
                evals = eigenvalues_all{i};
                time_val = obj.t_out(J_times(i));
                ax_handles(i) = SRNNModel2.plot_eigenvalues_helper(evals, ax, time_val, global_xlim, global_ylim);
                set(ax_handles(i), 'Color', 'none');
            end

            linkaxes(ax_handles, 'xy');
        end

        function [fig_handle, ax_handles] = plot_W_spectrum(obj)
            % PLOT_W_SPECTRUM Plot eigenvalue spectra of -I+W and the LTI Jacobian
            %
            % This method creates a 2-panel figure showing:
            %   Left:  Eigenvalues of (-I + W) (unscaled Jacobian)
            %   Right: Eigenvalues of (-I + W)/tau_d (the LTI Jacobian)
            %
            % The theoretical spectral radius R (from Harris 2023 Eq 18) is shown
            % as a circle centered at -1. For the Jacobian, the circle is
            % centered at -1/tau_d and scaled by 1/tau_d.
            %
            % Usage:
            %   model.build();
            %   model.plot_W_spectrum();

            if ~obj.is_built
                error('SRNNModel:NotBuilt', 'Model must be built before plotting spectrum. Call build() first.');
            end

            % Compute eigenvalues
            J_unscaled = -eye(obj.n) + obj.W;
            eigs_unscaled = eig(J_unscaled);
            J_lti = J_unscaled / obj.tau_d;
            eigs_J = eig(J_lti);

            % Theoretical predictions
            R_W = obj.R;  % Already scaled by level_of_chaos
            outlier_threshold = 1.04;

            % Create figure
            fig_handle = figure('Position', [100, 300, 900, 400]);
            ax_handles = gobjects(2, 1);

            %% Left panel: (-I + W) eigenvalues (unscaled Jacobian)
            ax_handles(1) = subplot(1, 2, 1);
            center_unscaled = -1;  % Shifted by -I
            obj.plot_spectrum_helper(ax_handles(1), eigs_unscaled, R_W, center_unscaled, outlier_threshold);
            title(ax_handles(1), sprintf('-I + W eigenvalues (R = %.2f)', R_W), 'FontWeight', 'bold');
            xlabel(ax_handles(1), 'Re(\lambda)');
            ylabel(ax_handles(1), 'Im(\lambda)');

            % Add stability line at Re = 0
            hold(ax_handles(1), 'on');
            yl = ylim(ax_handles(1));
            plot(ax_handles(1), [0, 0], yl, 'r--', 'LineWidth', 1.5);
            hold(ax_handles(1), 'off');

            %% Right panel: LTI Jacobian eigenvalues (-I + W)/tau_d
            ax_handles(2) = subplot(1, 2, 2);
            % For the Jacobian: center shifts to -1/tau_d, radius scales by 1/tau_d
            R_J = R_W / obj.tau_d;
            center_J = -1 / obj.tau_d;
            obj.plot_spectrum_helper(ax_handles(2), eigs_J, R_J, center_J, outlier_threshold);
            title(ax_handles(2), sprintf('(-I + W)/\\tau_d eigenvalues (R_J = %.2f)', R_J), 'FontWeight', 'bold');
            xlabel(ax_handles(2), 'Re(\lambda)');
            ylabel(ax_handles(2), 'Im(\lambda)');

            % Add stability line at Re = 0
            hold(ax_handles(2), 'on');
            yl = ylim(ax_handles(2));
            plot(ax_handles(2), [0, 0], yl, 'r--', 'LineWidth', 1.5);
            hold(ax_handles(2), 'off');
        end

        function params = get_params(obj)
            % GET_PARAMS Return params struct for SRNN dynamics.
            %
            % Extends cRNN.get_params() with SRNN-specific fields.

            params = get_params@cRNN(obj);

            % RMT computed values (from connectivity strategy)
            params.R = obj.R;

            % Computed E/I params
            params.N_sys_eqs = obj.N_sys_eqs;

            % Adaptation params
            params.n_a_E = obj.n_a_E;
            params.n_a_I = obj.n_a_I;
            params.tau_a_E = obj.tau_a_E;
            params.tau_a_I = obj.tau_a_I;
            params.c_E = obj.c_E;
            params.c_I = obj.c_I;

            % STD params
            params.n_b_E = obj.n_b_E;
            params.n_b_I = obj.n_b_I;
            params.tau_b_E_rec = obj.tau_b_E_rec;
            params.tau_b_E_rel = obj.tau_b_E_rel;
            params.tau_b_I_rec = obj.tau_b_I_rec;
            params.tau_b_I_rel = obj.tau_b_I_rel;

            % Dynamics
            params.tau_d = obj.tau_d;
            params.activation_function = obj.activation_function;
            params.activation_function_derivative = obj.activation_function_derivative;
        end

        function dS_dt = dynamics(obj, t, S)
            % DYNAMICS Compute the right-hand side of the SRNN ODE system
            %
            % This is a convenience method that wraps dynamics_fast.
            % For performance-critical code (e.g., ODE integration), use
            % dynamics_fast directly with a params struct.

            params = obj.cached_params;
            params.u_interpolant = obj.u_interpolant;
            dS_dt = SRNNModel2.dynamics_fast(t, S, params);
        end
    end

    %% Protected Build Sub-Methods (SRNN overrides)
    methods (Access = protected)
        function build_network(obj)
            % BUILD_NETWORK Set SRNN-specific defaults, then delegate to cRNN.
            %
            % Sets tau_a defaults, then calls cRNN.build_network() which
            % delegates to the connectivity strategy.

            % Compute tau_a arrays if n_a > 0 but tau_a not set
            if obj.n_a_E > 0 && isempty(obj.tau_a_E)
                obj.tau_a_E = logspace(log10(0.25), log10(10), obj.n_a_E);
            end
            if obj.n_a_I > 0 && isempty(obj.tau_a_I)
                obj.tau_a_I = logspace(log10(0.25), log10(10), obj.n_a_I);
            end

            % Call base class build_network (delegates to connectivity strategy)
            build_network@cRNN(obj);

            % Default W_in = eye(n): external input maps 1-to-1 to neurons
            if isempty(obj.W_in)
                obj.W_in = eye(obj.n);
            end
        end

        function validate(obj)
            % VALIDATE SRNN-specific parameter validation.

            if obj.n_E < 1
                error('SRNNModel:InvalidParams', 'n_E must be >= 1. Current: %d (n=%d, f=%.2f)', obj.n_E, obj.n, obj.f);
            end

            if obj.n_I < 1
                warning('SRNNModel:NoInhibitory', 'No inhibitory neurons (n_I=%d).', obj.n_I);
            end

            if obj.n_a_E > 0 && isempty(obj.tau_a_E)
                error('SRNNModel:InvalidParams', 'tau_a_E must be set when n_a_E > 0');
            end
            if obj.n_a_I > 0 && isempty(obj.tau_a_I)
                error('SRNNModel:InvalidParams', 'tau_a_I must be set when n_a_I > 0');
            end

            if ~isempty(obj.connectivity) && isprop(obj.connectivity, 'level_of_chaos')
                if obj.connectivity.level_of_chaos <= 0
                    warning('SRNNModel:InvalidParams', 'level_of_chaos should be > 0. Current: %.2f', obj.connectivity.level_of_chaos);
                end
            end
        end
    end

    %% SRNN-specific Protected Methods
    methods (Access = protected)
        function decimate_and_unpack(obj)
            % DECIMATE_AND_UNPACK Decimate state data and unpack for plotting

            params = obj.cached_params;

            % Decimate
            [t_plot, S_plot, plot_indices] = obj.decimate_states(obj.t_out, obj.state_out, obj.plot_deci);

            % Unpack state vector and compute firing rates
            [x_plot, a_plot, b_plot, r_plot, br_plot] = obj.unpack_and_compute_states(S_plot, params);

            % Split external input into E and I
            u_ex_plot = obj.u_ex(:, plot_indices);
            u_plot.E = u_ex_plot(obj.E_indices, :);
            u_plot.I = u_ex_plot(obj.I_indices, :);

            % Trim to T_plot if specified
            T_plot_range = obj.T_plot;
            if isempty(T_plot_range)
                T_plot_range = obj.T_range;
            end

            keep_mask = t_plot >= T_plot_range(1) & t_plot <= T_plot_range(2);

            t_plot = t_plot(keep_mask);
            u_plot.E = u_plot.E(:, keep_mask);
            u_plot.I = u_plot.I(:, keep_mask);
            x_plot = obj.trim_struct_data(x_plot, 2, keep_mask);
            r_plot = obj.trim_struct_data(r_plot, 2, keep_mask);
            b_plot = obj.trim_struct_data(b_plot, 2, keep_mask);
            br_plot = obj.trim_struct_data(br_plot, 2, keep_mask);
            a_plot = obj.trim_struct_data(a_plot, 3, keep_mask);

            % Trim Lyapunov results if present
            if ~isempty(obj.lya_results) && isfield(obj.lya_results, 't_lya')
                keep_mask_lya = obj.lya_results.t_lya >= T_plot_range(1) & obj.lya_results.t_lya <= T_plot_range(2);
                obj.lya_results.t_lya = obj.lya_results.t_lya(keep_mask_lya);

                if isfield(obj.lya_results, 'local_lya')
                    obj.lya_results.local_lya = obj.lya_results.local_lya(keep_mask_lya);
                end
                if isfield(obj.lya_results, 'finite_lya')
                    obj.lya_results.finite_lya = obj.lya_results.finite_lya(keep_mask_lya);
                end
                if isfield(obj.lya_results, 'local_LE_spectrum_t')
                    obj.lya_results.local_LE_spectrum_t = obj.lya_results.local_LE_spectrum_t(keep_mask_lya, :);
                end
                if isfield(obj.lya_results, 'finite_LE_spectrum_t')
                    obj.lya_results.finite_LE_spectrum_t = obj.lya_results.finite_LE_spectrum_t(keep_mask_lya, :);
                end
            end

            % Store plot data
            obj.plot_data = struct();
            obj.plot_data.t = t_plot;
            obj.plot_data.u = u_plot;
            obj.plot_data.x = x_plot;
            obj.plot_data.r = r_plot;
            obj.plot_data.a = a_plot;
            obj.plot_data.b = b_plot;
            obj.plot_data.br = br_plot;
        end

        function plot_spectrum_helper(~, ax, eigs, R, center, outlier_threshold)
            % PLOT_SPECTRUM_HELPER Helper function for plotting eigenvalue spectra
            %
            % Inputs:
            %   ax               - Axes handle to plot on
            %   eigs             - Vector of eigenvalues
            %   R                - Theoretical spectral radius
            %   center           - Center of the spectral disc (real part)
            %   outlier_threshold - Multiplier for R to classify far outliers

            % Compute distances from center for all eigenvalues
            distances = abs(eigs - center);

            % Plot interior eigenvalues (within R) as black circles
            mSize = 4;
            interior_mask = distances <= R;
            interior_eigs = eigs(interior_mask);
            plot(ax, real(interior_eigs), imag(interior_eigs), 'ko', 'MarkerSize', mSize, 'MarkerFaceColor', 'none', 'LineWidth', 0.5);
            hold(ax, 'on');

            % Plot theoretical radius circle (Eq 18)
            theta = linspace(0, 2*pi, 100);
            plot(ax, center + R*cos(theta), R*sin(theta), 'k-', 'LineWidth', 2);

            % Plot near outlier eigenvalues (between R and outlier_threshold*R) as black Xs
            near_outlier_mask = (distances > R) & (distances <= outlier_threshold * R);
            near_outlier_eigs = eigs(near_outlier_mask);
            if ~isempty(near_outlier_eigs)
                plot(ax, real(near_outlier_eigs), imag(near_outlier_eigs), 'kx', 'MarkerSize', mSize, 'LineWidth', 0.5);
            end

            % Plot far outlier eigenvalues (beyond outlier_threshold*R) as green filled circles
            far_outlier_mask = distances > outlier_threshold * R;
            far_outlier_eigs = eigs(far_outlier_mask);
            if ~isempty(far_outlier_eigs)
                plot(ax, real(far_outlier_eigs), imag(far_outlier_eigs), 'o', 'MarkerSize', mSize, 'MarkerFaceColor', [0 .7 0], 'MarkerEdgeColor', [0 .7 0]);
            end

            % Add axis lines through origin
            xl = xlim(ax);
            yl = ylim(ax);
            plot(ax, xl, [0 0], 'k-', 'LineWidth', 0.5);
            plot(ax, [0 0], yl, 'k-', 'LineWidth', 0.5);

            grid(ax, 'on');
            axis(ax, 'equal');
            hold(ax, 'off');
        end

        function [t_plot, S_plot, indices] = decimate_states(~, t_out, S_out, deci)
            % DECIMATE_STATES Decimates state trajectory for plotting
            %
            % Inputs:
            %   t_out - Time vector
            %   S_out - State matrix (nt x N)
            %   deci  - Decimation factor (integer)
            %
            % Outputs:
            %   t_plot  - Decimated time vector
            %   S_plot  - Decimated state matrix
            %   indices - Indices used for decimation

            indices = 1:deci:length(t_out);
            t_plot = t_out(indices);
            S_plot = S_out(indices, :);
        end


        function [x, a, b, r, br] = unpack_and_compute_states(~, S_out, params, a_zeros_b_ones)
            % UNPACK_AND_COMPUTE_STATES Unpack state vector and compute dependent variables
            %
            % Unpacks the state trajectory S_out into individual state variables,
            % splits them into excitatory and inhibitory components, and computes
            % the firing rate r and synaptic output br.
            %
            % Inputs:
            %   S_out          - State trajectory (nt x N_sys_eqs) or column vector
            %   params         - Struct containing network parameters
            %   a_zeros_b_ones - (Optional) If true, returns a as zeros and b as ones

            % Handle optional parameter
            if nargin < 4
                a_zeros_b_ones = false;
            end

            nt = size(S_out, 1);
            current_idx = 0;

            %% Unpack adaptation states

            % Unpack adaptation states for E neurons (a_E)
            len_a_E = params.n_E * params.n_a_E;
            if len_a_E > 0
                a_E_ts = reshape(S_out(:, current_idx + (1:len_a_E))', params.n_E, params.n_a_E, nt);
            else
                a_E_ts = [];
            end
            current_idx = current_idx + len_a_E;

            % Unpack adaptation states for I neurons (a_I)
            len_a_I = params.n_I * params.n_a_I;
            if len_a_I > 0
                a_I_ts = reshape(S_out(:, current_idx + (1:len_a_I))', params.n_I, params.n_a_I, nt);
            else
                a_I_ts = [];
            end
            current_idx = current_idx + len_a_I;

            %% Unpack STD variables (b)

            % Unpack b states for E neurons (b_E)
            len_b_E = params.n_E * params.n_b_E;
            if len_b_E > 0
                b_E_ts = S_out(:, current_idx + (1:len_b_E))';  % n_E x nt
            else
                b_E_ts = [];
            end
            current_idx = current_idx + len_b_E;

            % Unpack b states for I neurons (b_I)
            len_b_I = params.n_I * params.n_b_I;
            if len_b_I > 0
                b_I_ts = S_out(:, current_idx + (1:len_b_I))';  % n_I x nt
            else
                b_I_ts = [];
            end
            current_idx = current_idx + len_b_I;

            %% Unpack dendritic states (x)
            x_ts = S_out(:, current_idx + (1:params.n))';  % n x nt

            %% Compute firing rates with adaptation and STD

            % Compute effective dendritic state (x_eff = x - c * sum(a))
            x_eff_ts = x_ts;  % n x nt

            % Apply adaptation effect to E neurons (scaled by c_E)
            if params.n_E > 0 && params.n_a_E > 0 && ~isempty(a_E_ts)
                % sum(a_E_ts, 2) is n_E x 1 x nt, need to squeeze to n_E x nt
                sum_a_E = squeeze(sum(a_E_ts, 2));  % n_E x nt
                if size(sum_a_E, 1) ~= params.n_E  % Handle case where nt=1
                    sum_a_E = sum_a_E';
                end
                x_eff_ts(params.E_indices, :) = x_eff_ts(params.E_indices, :) - params.c_E * sum_a_E;
            end

            % Apply adaptation effect to I neurons (scaled by c_I)
            if params.n_I > 0 && params.n_a_I > 0 && ~isempty(a_I_ts)
                sum_a_I = squeeze(sum(a_I_ts, 2));  % n_I x nt
                if size(sum_a_I, 1) ~= params.n_I
                    sum_a_I = sum_a_I';
                end
                x_eff_ts(params.I_indices, :) = x_eff_ts(params.I_indices, :) - params.c_I * sum_a_I;
            end

            % Apply STD effect (b multiplicative factor)
            b_ts = ones(params.n, nt);  % Initialize b = 1 for all neurons (no depression)
            if params.n_b_E > 0 && ~isempty(b_E_ts)
                b_ts(params.E_indices, :) = b_E_ts;
            end
            if params.n_b_I > 0 && ~isempty(b_I_ts)
                b_ts(params.I_indices, :) = b_I_ts;
            end

            % Compute firing rates: r = phi(x_eff) (raw rate)
            r_ts = params.activation_function(x_eff_ts);  % n x nt

            % Compute synaptic output: br = b .* r (presynaptically depressed)
            br_ts = b_ts .* r_ts; % n x nt

            %% Split into E and I components and package into structs

            % x: dendritic states
            x.E = x_ts(params.E_indices, :);  % n_E x nt
            x.I = x_ts(params.I_indices, :);  % n_I x nt

            % a: adaptation variables
            a.E = a_E_ts;  % n_E x n_a_E x nt (or empty)
            a.I = a_I_ts;  % n_I x n_a_I x nt (or empty)

            % b: STD variables
            if isempty(b_E_ts)
                b.E = ones(params.n_E, nt);  % Default to no depression
            else
                b.E = b_E_ts;  % n_E x nt
            end

            if isempty(b_I_ts)
                b.I = ones(params.n_I, nt);  % Default to no depression
            else
                b.I = b_I_ts;  % n_I x nt
            end

            % r: firing rates
            r.E = r_ts(params.E_indices, :);  % n_E x nt
            r.I = r_ts(params.I_indices, :);  % n_I x nt

            % br: synaptic output
            br.E = br_ts(params.E_indices, :);
            br.I = br_ts(params.I_indices, :);

            %% Override with zeros/ones if requested (for Jacobian computation)
            if a_zeros_b_ones
                % Return x as simple array instead of struct (n x nt)
                x = x_ts;

                % Replace a with zeros (n x nt)
                a = zeros(params.n, nt);

                % Replace b with ones (n x nt)
                b = ones(params.n, nt);

                % Return r as simple array instead of struct (n x nt)
                r = r_ts;

                % Return br as simple array (equal to r since b=1)
                br = r_ts;
            end
        end
    end

    methods (Static, Access = protected)
        function s_out = trim_struct_data(s_in, dim, mask)
            % TRIM_STRUCT_DATA Helper to trim fields of a struct along a dimension
            s_out = s_in;
            fields = fieldnames(s_in);
            for i = 1:length(fields)
                val = s_in.(fields{i});
                if ~isempty(val)
                    if dim == 2
                        s_out.(fields{i}) = val(:, mask);
                    elseif dim == 3
                        s_out.(fields{i}) = val(:, :, mask);
                    end
                end
            end
        end

        function ax = plot_eigenvalues_helper(eigenvalues, ax, time_value, x_lim, y_lim, circle_params)
            % PLOT_EIGENVALUES_HELPER Plot eigenvalue distribution on complex plane
            % Internalized from src/plotting/plot_eigenvalues.m (Option B: 3-tier outlier coloring)

            if nargin < 6, circle_params = []; end
            if nargin < 5, y_lim = []; end
            if nargin < 4, x_lim = []; end

            axes(ax);
            mSize = 4;
            hold on;

            has_circle = ~isempty(circle_params) && isfield(circle_params, 'center') && isfield(circle_params, 'radius');

            if has_circle
                R = circle_params.radius;
                xc = real(circle_params.center);
                yc = imag(circle_params.center);

                if isfield(circle_params, 'outlier_threshold')
                    outlier_threshold = circle_params.outlier_threshold;
                else
                    outlier_threshold = 1.04;
                end

                distances = abs(eigenvalues - circle_params.center);

                % Interior eigenvalues (within R): black unfilled circles
                interior_mask = distances <= R;
                interior_eigs = eigenvalues(interior_mask);
                plot(real(interior_eigs), imag(interior_eigs), 'ko', 'MarkerSize', mSize, 'MarkerFaceColor', 'none', 'LineWidth', 0.5);

                % Near outlier eigenvalues (between R and outlier_threshold*R): black Xs
                near_outlier_mask = (distances > R) & (distances <= outlier_threshold * R);
                near_outlier_eigs = eigenvalues(near_outlier_mask);
                if ~isempty(near_outlier_eigs)
                    plot(real(near_outlier_eigs), imag(near_outlier_eigs), 'kx', 'MarkerSize', mSize, 'LineWidth', 0.5);
                end

                % Far outlier eigenvalues (beyond outlier_threshold*R): green filled circles
                far_outlier_mask = distances > outlier_threshold * R;
                far_outlier_eigs = eigenvalues(far_outlier_mask);
                if ~isempty(far_outlier_eigs)
                    plot(real(far_outlier_eigs), imag(far_outlier_eigs), 'o', 'MarkerSize', mSize, 'MarkerFaceColor', [0 .7 0], 'MarkerEdgeColor', [0 .7 0]);
                end

                % Draw theoretical radius as solid black circle
                theta = linspace(0, 2*pi, 100);
                plot(xc + R*cos(theta), yc + R*sin(theta), 'k-', 'LineWidth', 2);
            else
                % No circle params: plot all eigenvalues as black unfilled circles
                plot(real(eigenvalues), imag(eigenvalues), 'ko', 'MarkerSize', mSize, 'MarkerFaceColor', 'none', 'LineWidth', 0.5);
            end

            if isempty(x_lim), x_lim = xlim; end
            if x_lim(2) < 0, x_lim(2) = 0.05; end
            if isempty(y_lim), y_lim = ylim; end

            axis off;
            hold on;
            h_x = plot(x_lim, [0, 0], 'k', 'LineWidth', 1.25);
            h_y = plot([0, 0], y_lim, 'k', 'LineWidth', 1.25);
            uistack([h_x, h_y], 'bottom');

            text(1.02*x_lim(2), 0, ' Re($\lambda$)', 'Interpreter', 'latex', 'VerticalAlignment', 'middle');
            text(0, y_lim(2), 'Im($\lambda$)', 'Interpreter', 'latex', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');

            xlim(x_lim);
            ylim(y_lim);
            hold off;
            axis equal;
        end

        function dS_dt = dynamics_fast(t, S, params)
            % DYNAMICS_FAST Static method for fast ODE evaluation
            %
            % This static method avoids OOP overhead by taking all parameters
            % as a struct. The interpolant must be in params.u_interpolant.
            %
            % Implements:
            %   dx_i/dt = (-x_i + sum_j(w_ij * r_j) + u_i) / tau_d
            %   r_i = b_i * phi(x_i - c * sum_k(a_i,k))
            %   da_i,k/dt = (-a_i,k + r_i) / tau_k
            %   db_i/dt = (1 - b_i) / tau_rec - (b_i * r_i) / tau_rel
            %
            % State organization: S = [a_E(:); a_I(:); b_E(:); b_I(:); x(:)]

            %% Interpolate external input
            u_raw = params.u_interpolant(t)';  % (n_in x 1) raw input
            u = params.W_in * u_raw;           % (n x 1) projected input

            %% Load parameters
            n = params.n;
            n_E = params.n_E;
            n_I = params.n_I;
            E_indices = params.E_indices;
            I_indices = params.I_indices;

            n_a_E = params.n_a_E;
            n_a_I = params.n_a_I;
            n_b_E = params.n_b_E;
            n_b_I = params.n_b_I;

            W = params.W;
            tau_d = params.tau_d;
            tau_a_E = params.tau_a_E;
            tau_a_I = params.tau_a_I;
            tau_b_E_rec = params.tau_b_E_rec;
            tau_b_E_rel = params.tau_b_E_rel;
            tau_b_I_rec = params.tau_b_I_rec;
            tau_b_I_rel = params.tau_b_I_rel;

            c_E = params.c_E;
            c_I = params.c_I;
            activation_fn = params.activation_function;

            %% Unpack state variables
            current_idx = 0;

            % Adaptation states for E neurons (a_E)
            len_a_E = n_E * n_a_E;
            if len_a_E > 0
                a_E = reshape(S(current_idx + (1:len_a_E)), n_E, n_a_E);
            else
                a_E = [];
            end
            current_idx = current_idx + len_a_E;

            % Adaptation states for I neurons (a_I)
            len_a_I = n_I * n_a_I;
            if len_a_I > 0
                a_I = reshape(S(current_idx + (1:len_a_I)), n_I, n_a_I);
            else
                a_I = [];
            end
            current_idx = current_idx + len_a_I;

            % STD states for E neurons (b_E)
            len_b_E = n_E * n_b_E;
            if len_b_E > 0
                b_E = S(current_idx + (1:len_b_E));
            else
                b_E = [];
            end
            current_idx = current_idx + len_b_E;

            % STD states for I neurons (b_I)
            len_b_I = n_I * n_b_I;
            if len_b_I > 0
                b_I = S(current_idx + (1:len_b_I));
            else
                b_I = [];
            end
            current_idx = current_idx + len_b_I;

            % Dendritic states (x)
            x = S(current_idx + (1:n));

            %% Compute firing rates
            x_eff = x;

            % Apply adaptation effect to E neurons
            if n_E > 0 && n_a_E > 0 && ~isempty(a_E)
                x_eff(E_indices) = x_eff(E_indices) - c_E * sum(a_E, 2);
            end

            % Apply adaptation effect to I neurons
            if n_I > 0 && n_a_I > 0 && ~isempty(a_I)
                x_eff(I_indices) = x_eff(I_indices) - c_I * sum(a_I, 2);
            end

            % Apply STD effect
            b = ones(n, 1);
            if n_b_E > 0 && ~isempty(b_E)
                b(E_indices) = b_E;
            end
            if n_b_I > 0 && ~isempty(b_I)
                b(I_indices) = b_I;
            end

            r = activation_fn(x_eff);

            %% Compute derivatives
            dx_dt = (-x + W * (b .* r) + u) / tau_d;

            da_E_dt = [];
            if n_E > 0 && n_a_E > 0 && ~isempty(a_E)
                da_E_dt = (r(E_indices) - a_E) ./ tau_a_E;
            end

            da_I_dt = [];
            if n_I > 0 && n_a_I > 0 && ~isempty(a_I)
                da_I_dt = (r(I_indices) - a_I) ./ tau_a_I;
            end

            db_E_dt = [];
            if n_E > 0 && n_b_E > 0 && ~isempty(b_E)
                db_E_dt = (1 - b_E) / tau_b_E_rec - (b_E .* r(E_indices)) / tau_b_E_rel;
            end

            db_I_dt = [];
            if n_I > 0 && n_b_I > 0 && ~isempty(b_I)
                db_I_dt = (1 - b_I) / tau_b_I_rec - (b_I .* r(I_indices)) / tau_b_I_rel;
            end

            %% Pack derivatives
            dS_dt = [da_E_dt(:); da_I_dt(:); db_E_dt(:); db_I_dt(:); dx_dt];
        end
    end

    %% ====================================================================
    %              INTERNALIZED JACOBIAN COMPUTATION
    % =====================================================================
    % Internalized from src/algorithms/Jacobian/ to make SRNNModel2 standalone.

    methods (Static)
        function J = compute_Jacobian_fast(S, params)
            % COMPUTE_JACOBIAN_FAST Sparse/vectorized Jacobian assembly for the SRNN system.
            % Internalized from src/algorithms/Jacobian/compute_Jacobian_fast.m

            n = params.n;
            n_E = params.n_E;
            n_I = params.n_I;
            E_indices = params.E_indices;
            I_indices = params.I_indices;

            n_a_E = params.n_a_E;
            n_a_I = params.n_a_I;
            n_b_E = params.n_b_E;
            n_b_I = params.n_b_I;

            if n_b_E > 1 || n_b_I > 1
                error('compute_Jacobian_fast:UnsupportedSTDStates', ...
                      'Fast Jacobian currently supports at most one STD state per neuron.');
            end

            W = params.W;
            tau_d = params.tau_d;
            tau_a_E = params.tau_a_E;
            tau_a_I = params.tau_a_I;
            tau_b_E_rec = params.tau_b_E_rec;
            tau_b_E_rel = params.tau_b_E_rel;
            tau_b_I_rec = params.tau_b_I_rec;
            tau_b_I_rel = params.tau_b_I_rel;

            c_E = SRNNModel2.safe_get_param(params, 'c_E', 1.0);
            c_I = SRNNModel2.safe_get_param(params, 'c_I', 1.0);

            if ~isfield(params, 'activation_function_derivative') || ...
               ~isa(params.activation_function_derivative, 'function_handle')
                error('compute_Jacobian_fast:MissingActivationFunctionDerivative', ...
                      'params.activation_function_derivative must be provided as a function handle');
            end
            phi_prime = params.activation_function_derivative;

            if ~isfield(params, 'activation_function') || ...
               ~isa(params.activation_function, 'function_handle')
                error('compute_Jacobian_fast:MissingActivationFunction', ...
                      'params.activation_function must be provided as a function handle');
            end
            phi_fun = params.activation_function;

            %% Unpack state variables
            current_idx = 0;
            len_a_E = n_E * n_a_E;
            len_a_I = n_I * n_a_I;
            len_b_E = n_E * n_b_E;
            len_b_I = n_I * n_b_I;

            if len_a_E > 0
                a_E = reshape(S(current_idx + (1:len_a_E)), n_E, n_a_E);
            else
                a_E = [];
            end
            current_idx = current_idx + len_a_E;

            if len_a_I > 0
                a_I = reshape(S(current_idx + (1:len_a_I)), n_I, n_a_I);
            else
                a_I = [];
            end
            current_idx = current_idx + len_a_I;

            if len_b_E > 0
                b_E = S(current_idx + (1:len_b_E));
            else
                b_E = [];
            end
            current_idx = current_idx + len_b_E;

            if len_b_I > 0
                b_I = S(current_idx + (1:len_b_I));
            else
                b_I = [];
            end
            current_idx = current_idx + len_b_I;

            x = S(current_idx + (1:n));

            %% Effective potentials and rates
            x_eff = x;
            if len_a_E > 0
                x_eff(E_indices) = x_eff(E_indices) - c_E * sum(a_E, 2);
            end
            if len_a_I > 0
                x_eff(I_indices) = x_eff(I_indices) - c_I * sum(a_I, 2);
            end

            b = ones(n, 1);
            if len_b_E > 0
                b(E_indices) = b_E;
            end
            if len_b_I > 0
                b(I_indices) = b_I;
            end

            phi_x_eff = phi_fun(x_eff);
            phi_prime_x_eff = phi_prime(x_eff);
            r_vec = b .* phi_x_eff;

            %% Dimensions and indexing
            N_sys_eqs = len_a_E + len_a_I + len_b_E + len_b_I + n;

            row_a_E = 1:len_a_E;
            row_a_I = len_a_E + (1:len_a_I);
            row_b_E = len_a_E + len_a_I + (1:len_b_E);
            row_b_I = len_a_E + len_a_I + len_b_E + (1:len_b_I);
            row_x   = len_a_E + len_a_I + len_b_E + len_b_I + (1:n);

            col_a_E = row_a_E;
            col_a_I = row_a_I;
            col_b_E = row_b_E;
            col_b_I = row_b_I;
            col_x   = row_x;

            W_sparse = sparse(W);
            J = sparse(N_sys_eqs, N_sys_eqs);

            %% SFA blocks (E)
            if len_a_E > 0
                tau_inv_E = 1 ./ tau_a_E(:);
                diag_block_E = kron(speye(n_E), spdiags(-tau_inv_E, 0, n_a_E, n_a_E));
                gamma_E = c_E * (b(E_indices) .* phi_prime_x_eff(E_indices));
                row_template_E = sparse(tau_inv_E * ones(1, n_a_E));
                coupling_block_E = kron(spdiags(-gamma_E, 0, n_E, n_E), row_template_E);
                J(row_a_E, col_a_E) = diag_block_E + coupling_block_E;

                beta_E = b(E_indices) .* phi_prime_x_eff(E_indices);
                vals = kron(beta_E, tau_inv_E);
                rows = (1:len_a_E)';
                cols = repelem(E_indices(:), n_a_E);
                J(row_a_E, col_x) = sparse(rows, cols, vals, len_a_E, n);

                if len_b_E > 0
                    phi_E = phi_x_eff(E_indices);
                    J(row_a_E, col_b_E) = kron(spdiags(phi_E, 0, n_E, n_E), sparse(tau_inv_E));
                end
            end

            %% SFA blocks (I)
            if len_a_I > 0
                tau_inv_I = 1 ./ tau_a_I(:);
                diag_block_I = kron(speye(n_I), spdiags(-tau_inv_I, 0, n_a_I, n_a_I));
                gamma_I = c_I * (b(I_indices) .* phi_prime_x_eff(I_indices));
                row_template_I = sparse(tau_inv_I * ones(1, n_a_I));
                coupling_block_I = kron(spdiags(-gamma_I, 0, n_I, n_I), row_template_I);
                J(row_a_I, col_a_I) = diag_block_I + coupling_block_I;

                beta_I = b(I_indices) .* phi_prime_x_eff(I_indices);
                vals = kron(beta_I, tau_inv_I);
                rows = (1:len_a_I)';
                cols = repelem(I_indices(:), n_a_I);
                J(row_a_I, col_x) = sparse(rows, cols, vals, len_a_I, n);

                if len_b_I > 0
                    phi_I = phi_x_eff(I_indices);
                    J(row_a_I, col_b_I) = kron(spdiags(phi_I, 0, n_I, n_I), sparse(tau_inv_I));
                end
            end

            %% STD blocks (E)
            if len_b_E > 0
                phi_prime_E = phi_prime_x_eff(E_indices);
                coeff_a_E = (b(E_indices).^2) * c_E .* phi_prime_E / tau_b_E_rel;
                if len_a_E > 0
                    J(row_b_E, col_a_E) = kron(spdiags(coeff_a_E, 0, n_E, n_E), sparse(ones(1, n_a_E)));
                end
                diag_vals_b_E = -1/tau_b_E_rec - 2 * r_vec(E_indices) / tau_b_E_rel;
                J(row_b_E, col_b_E) = spdiags(diag_vals_b_E, 0, len_b_E, len_b_E);
                J(row_b_E, col_x) = sparse(1:n_E, E_indices, - (b(E_indices).^2) .* phi_prime_E / tau_b_E_rel, n_E, n);
            end

            %% STD blocks (I)
            if len_b_I > 0
                phi_prime_I = phi_prime_x_eff(I_indices);
                coeff_a_I = (b(I_indices).^2) * c_I .* phi_prime_I / tau_b_I_rel;
                if len_a_I > 0
                    J(row_b_I, col_a_I) = kron(spdiags(coeff_a_I, 0, n_I, n_I), sparse(ones(1, n_a_I)));
                end
                diag_vals_b_I = -1/tau_b_I_rec - 2 * r_vec(I_indices) / tau_b_I_rel;
                J(row_b_I, col_b_I) = spdiags(diag_vals_b_I, 0, len_b_I, len_b_I);
                J(row_b_I, col_x) = sparse(1:n_I, I_indices, - (b(I_indices).^2) .* phi_prime_I / tau_b_I_rel, n_I, n);
            end

            %% dx/dt blocks
            if len_a_E > 0
                replicate_a_E = kron(speye(n_E), ones(1, n_a_E));
                block = -c_E * W_sparse(:, E_indices) * spdiags(b(E_indices) .* phi_prime_x_eff(E_indices), 0, n_E, n_E);
                J(row_x, col_a_E) = (block * replicate_a_E) / tau_d;
            end

            if len_a_I > 0
                replicate_a_I = kron(speye(n_I), ones(1, n_a_I));
                block = -c_I * W_sparse(:, I_indices) * spdiags(b(I_indices) .* phi_prime_x_eff(I_indices), 0, n_I, n_I);
                J(row_x, col_a_I) = (block * replicate_a_I) / tau_d;
            end

            if len_b_E > 0
                replicate_b_E = kron(speye(n_E), ones(1, max(1, n_b_E)));
                block = W_sparse(:, E_indices) * spdiags(phi_x_eff(E_indices), 0, n_E, n_E);
                J(row_x, col_b_E) = (block * replicate_b_E) / tau_d;
            end

            if len_b_I > 0
                replicate_b_I = kron(speye(n_I), ones(1, max(1, n_b_I)));
                block = W_sparse(:, I_indices) * spdiags(phi_x_eff(I_indices), 0, n_I, n_I);
                J(row_x, col_b_I) = (block * replicate_b_I) / tau_d;
            end

            diag_term = spdiags(-ones(n,1)/tau_d, 0, n, n);
            gain_diag = spdiags(b .* phi_prime_x_eff, 0, n, n);
            J(row_x, col_x) = diag_term + (W_sparse * gain_diag) / tau_d;
        end

        function J_array = compute_Jacobian_at_indices(S_out, J_times, params)
            % COMPUTE_JACOBIAN_AT_INDICES Computes Jacobian matrices at multiple time indices.
            % Internalized from src/algorithms/Jacobian/compute_Jacobian_at_indices.m

            N_sys_eqs = size(S_out, 2);
            n_times = length(J_times);

            nt = size(S_out, 1);
            if any(J_times < 1) || any(J_times > nt)
                error('J_times contains invalid indices. Must be between 1 and %d', nt);
            end

            J_array = zeros(N_sys_eqs, N_sys_eqs, n_times);

            for i = 1:n_times
                S = S_out(J_times(i), :)';
                J_array(:,:,i) = full(SRNNModel2.compute_Jacobian_fast(S, params));
            end
        end
    end

    methods (Static, Access = private)
        function value = safe_get_param(params, field, default_value)
            % SAFE_GET_PARAM Helper to get a field from params with a default.
            if isfield(params, field)
                value = params.(field);
            else
                value = default_value;
            end
        end
    end

    %% ====================================================================
    %              INTERNALIZED PLOTTING FUNCTIONS
    % =====================================================================
    % Internalized from src/plotting/ to make SRNNModel2 standalone.

    methods (Static, Access = protected)
        function [fig_handle, ax_handles] = plot_SRNN_tseries(t_out, u, x, r, a, b, br, params, lya_results, Lya_method, T_plot)
            % PLOT_SRNN_TSERIES Create comprehensive time series plots for SRNN simulation.
            % Internalized from src/plotting/plot_SRNN_tseries.m

            if nargin < 11
                T_plot = [];
            end

            % Determine which subplots are needed
            has_adaptation = params.n_a_E > 0 || params.n_a_I > 0;
            has_std_vars = (params.n_b_E > 0 || params.n_b_I > 0) && ~isempty(b);
            has_synaptic_output = ~isempty(br);
            has_lyapunov = ~strcmpi(Lya_method, 'none');
            has_firing_rate = ~isempty(r);

            n_plots = 2;  % Always: External input, Dendritic states
            if has_firing_rate, n_plots = n_plots + 1; end
            if has_synaptic_output, n_plots = n_plots + 1; end
            if has_adaptation, n_plots = n_plots + 1; end
            if has_std_vars, n_plots = n_plots + 1; end
            if has_lyapunov, n_plots = n_plots + 1; end

            fig_handle = figure();
            tiledlayout(n_plots, 1);
            ax_handles = [];

            % Always: External input
            ax_handles(end+1) = nexttile;
            SRNNModel2.plot_external_input(t_out, u);
            set(gca, 'XTick', [], 'XTickLabel', [], 'XColor', 'white');

            % Always: Dendritic states
            ax_handles(end+1) = nexttile;
            plot_mean = false;
            if isfield(params, 'plot_mean_dendrite')
                plot_mean = params.plot_mean_dendrite;
            end
            SRNNModel2.plot_dendritic_state(t_out, x, plot_mean);
            set(gca, 'XTick', [], 'XTickLabel', [], 'XColor', 'white');

            % Conditionally: Firing rates
            if has_firing_rate
                ax_handles(end+1) = nexttile;
                SRNNModel2.plot_firing_rate(t_out, r);
                set(gca, 'XTick', [], 'XTickLabel', [], 'XColor', 'white');
            end

            % Conditionally: Synaptic output (br)
            if has_synaptic_output
                ax_handles(end+1) = nexttile;
                SRNNModel2.plot_synaptic_output(t_out, br);
                set(gca, 'XTick', [], 'XTickLabel', [], 'XColor', 'white');
            end

            % Conditionally: Adaptation variables
            if has_adaptation
                ax_handles(end+1) = nexttile;
                SRNNModel2.plot_adaptation(t_out, a, params);
                set(gca, 'XTick', [], 'XTickLabel', [], 'XColor', 'white');
            end

            % Conditionally: STD variables (b)
            if has_std_vars
                ax_handles(end+1) = nexttile;
                SRNNModel2.plot_std_variable(t_out, b, params);
                set(gca, 'XTick', [], 'XTickLabel', [], 'XColor', 'white');
            end

            % Conditionally: Lyapunov exponent(s)
            if has_lyapunov
                ax_handles(end+1) = nexttile;
                if strcmpi(Lya_method, 'benettin')
                    SRNNModel2.plot_lyapunov(lya_results, Lya_method, {'local', 'EOC', 'value'});
                else
                    SRNNModel2.plot_lyapunov(lya_results, Lya_method);
                end
                set(gca, 'XTick', [], 'XTickLabel', [], 'XColor', 'white');
            end

            linkaxes(ax_handles,'x');

            if ~isempty(T_plot)
                xlim(ax_handles(end), T_plot);
            end

            % Add time scale bar overlay in lower right of last subplot
            axes(ax_handles(end));
            hold on;
            xlims = xlim;
            ylims = ylim;

            scale_bar_length = round(0.1 * (xlims(2) - xlims(1)));
            if scale_bar_length < 1
                scale_bar_length = 0.1 * (xlims(2) - xlims(1));
            end

            x_end = xlims(1) + 0.95 * (xlims(2) - xlims(1));
            x_start = x_end - scale_bar_length;
            y_pos = ylims(1) + 0.10 * (ylims(2) - ylims(1));
            plot([x_start, x_end], [y_pos, y_pos], 'k-', 'LineWidth', 4);

            text_x = (x_start + x_end) / 2;
            text_y = ylims(1) + 0.05 * (ylims(2) - ylims(1));
            text(text_x, text_y, sprintf('%g seconds', scale_bar_length), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
            hold off;
        end

        function plot_external_input(t, u)
            % PLOT_EXTERNAL_INPUT Plot external input for E and I neurons.
            % Internalized from src/plotting/plot_external_input.m
            cmap_I = SRNNModel2.inhibitory_colormap(8);
            cmap_E = SRNNModel2.excitatory_colormap(8);
            SRNNModel2.plot_lines_with_colormap(t, u.I, cmap_I);
            hold on;
            SRNNModel2.plot_lines_with_colormap(t, u.E, cmap_E);
            hold off;
            ylabel('stim');
            yl = ylim;
            ylim(yl+[-0.05 0])
            yticks([-1 0 1]);
        end

        function plot_dendritic_state(t, x, plot_mean)
            % PLOT_DENDRITIC_STATE Plot dendritic states for E and I neurons.
            % Internalized from src/plotting/plot_dendritic_state.m
            if nargin < 3, plot_mean = false; end
            cmap_I = SRNNModel2.inhibitory_colormap(8);
            cmap_E = SRNNModel2.excitatory_colormap(8);
            SRNNModel2.plot_lines_with_colormap(t, x.I, cmap_I);
            hold on;
            SRNNModel2.plot_lines_with_colormap(t, x.E, cmap_E);
            if plot_mean
                mean_x = mean(x.E, 1);
                plot(t, mean_x, 'k', 'LineWidth', 3);
            end
            hold off;
            ylabel('dendrite');
        end

        function plot_firing_rate(t, r)
            % PLOT_FIRING_RATE Plot firing rates for E and I neurons.
            % Internalized from src/plotting/plot_firing_rate.m
            cmap_I = SRNNModel2.inhibitory_colormap(8);
            cmap_E = SRNNModel2.excitatory_colormap(8);
            SRNNModel2.plot_lines_with_colormap(t, r.I, cmap_I);
            hold on;
            SRNNModel2.plot_lines_with_colormap(t, r.E, cmap_E);
            hold off;
            ylabel('firing rate');
            yticks([0, 1]);
            ylim([0, 1]);
        end

        function plot_synaptic_output(t, br)
            % PLOT_SYNAPTIC_OUTPUT Plot synaptic output (br = b .* r) for E and I neurons.
            % Internalized from src/plotting/plot_synaptic_output.m
            cmap_I = SRNNModel2.inhibitory_colormap(8);
            cmap_E = SRNNModel2.excitatory_colormap(8);
            SRNNModel2.plot_lines_with_colormap(t, br.I, cmap_I);
            hold on;
            SRNNModel2.plot_lines_with_colormap(t, br.E, cmap_E);
            hold off;
            ylabel('synaptic output');
            yticks([0, 1]);
            ylim([0, 1]);
        end

        function plot_adaptation(t, a, params)
            % PLOT_ADAPTATION Plot adaptation variables for E and I neurons.
            % Internalized from src/plotting/plot_adaptation.m
            cmap_I = SRNNModel2.inhibitory_colormap(8);
            cmap_E = SRNNModel2.excitatory_colormap(8);
            has_adaptation = false;

            if ~isempty(a.I) && params.n_a_I > 0
                a_I_sum = sum(a.I, 2);
                a_I_summed = reshape(a_I_sum, params.n_I, []);
                SRNNModel2.plot_lines_with_colormap(t, a_I_summed, cmap_I);
                has_adaptation = true;
            end

            if ~isempty(a.E) && params.n_a_E > 0
                if has_adaptation, hold on; end
                a_E_sum = sum(a.E, 2);
                a_E_summed = reshape(a_E_sum, params.n_E, []);
                SRNNModel2.plot_lines_with_colormap(t, a_E_summed, cmap_E);
                has_adaptation = true;
            end

            if has_adaptation
                hold off;
                ylabel('adaptation');
            else
                text(0.5, 0.5, 'No adaptation variables', 'HorizontalAlignment', 'center');
                axis off;
            end
        end

        function plot_std_variable(t, b, params)
            % PLOT_STD_VARIABLE Plot short-term depression variables for E and I neurons.
            % Internalized from src/plotting/plot_std_variable.m
            cmap_I = SRNNModel2.inhibitory_colormap(8);
            cmap_E = SRNNModel2.excitatory_colormap(8);
            has_std = false;

            if params.n_b_I > 0 && ~isempty(b.I) && size(b.I, 1) > 0
                if ~all(b.I(:) == 1)
                    SRNNModel2.plot_lines_with_colormap(t, b.I, cmap_I);
                    has_std = true;
                end
            end

            if params.n_b_E > 0 && ~isempty(b.E) && size(b.E, 1) > 0
                if ~all(b.E(:) == 1)
                    if has_std, hold on; end
                    SRNNModel2.plot_lines_with_colormap(t, b.E, cmap_E);
                    has_std = true;
                end
            end

            if has_std
                hold off;
                ylabel('depression');
                ylim([0, 1]);
                yticks([0, 1]);
            else
                text(0.5, 0.5, 'No STD variables', 'HorizontalAlignment', 'center');
                axis off;
            end
        end

        function plot_lyapunov(lya_results, Lya_method, plot_options)
            % PLOT_LYAPUNOV Plot Lyapunov exponent(s) on current axes.
            % Internalized from src/plotting/plot_lyapunov.m

            if nargin < 3
                plot_options = {'local', 'asym', 'EOC', 'value'};
            end

            if strcmpi(Lya_method, 'benettin')
                valid_options = {'local', 'asym', 'EOC', 'value'};
                if ~iscell(plot_options)
                    error('plot_options must be a cell array of strings');
                end
                for i = 1:length(plot_options)
                    if strcmpi(plot_options{i}, 'filtered')
                        error(['The ''filtered'' option has been removed. ', ...
                            'Filtering now occurs in SRNNModel before decimation. ', ...
                            'Set model.filter_local_lya = true and use ''local'' instead.']);
                    end
                    if ~any(strcmpi(plot_options{i}, valid_options))
                        error('Invalid plot_option: %s. Valid options are: %s', ...
                            plot_options{i}, strjoin(valid_options, ', '));
                    end
                end
            end

            if strcmpi(Lya_method, 'benettin')
                plot_local = any(strcmpi('local', plot_options));
                plot_asym = any(strcmpi('asym', plot_options));
                plot_EOC = any(strcmpi('EOC', plot_options));
                plot_value = any(strcmpi('value', plot_options));

                legend_entries = {};
                plot_started = false;

                if plot_local
                    colors = lines(1);
                    plot(lya_results.t_lya, lya_results.local_lya, 'Color', colors(1,:))
                    hold on
                    plot_started = true;
                    legend_entries{end+1} = 'Local LLE';
                end

                if plot_asym
                    if ~plot_started, hold on; plot_started = true; end
                    plot([lya_results.t_lya(1), lya_results.t_lya(end)], ...
                        [lya_results.LLE, lya_results.LLE], '--r', 'LineWidth', 1.5);
                end

                if plot_EOC
                    if ~plot_started, hold on; plot_started = true; end
                    plot([lya_results.t_lya(1), lya_results.t_lya(end)], [0, 0], '--k');
                end

                ylabel('\lambda_1')

                if plot_value
                    ylims = ylim;
                    xlims = xlim;
                    text_y = 0.05 * (ylims(2) - ylims(1));
                    text_x = xlims(2);
                    text(text_x, text_y, ['$\lambda_1 = ' sprintf('%.2f', lya_results.LLE) '$'], ...
                        'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', ...
                        'Interpreter', 'latex');
                end

                hold off
                box off

            elseif strcmpi(Lya_method, 'qr')
                plot_data = lya_results.local_LE_spectrum_t(:, end:-1:1);
                line_handles = plot(lya_results.t_lya, plot_data);
                line_handles = line_handles(end:-1:1);

                hold on
                yline(0, '--k')
                ylabel('\lambda_1')

                legend_count = min(5, lya_results.params.N_sys_eqs);
                legend_entries = cell(1, legend_count);
                for i = 1:legend_count
                    legend_entries{i} = sprintf('\\lambda_{%d} = %.3f', i, lya_results.LE_spectrum(i));
                end
                legend(line_handles(1:legend_count), legend_entries, 'Location', 'best')
                hold off
                box off
            end
        end

        function plot_lines_with_colormap(t, data, cmap, varargin)
            % PLOT_LINES_WITH_COLORMAP Plot multiple lines with explicit colormap assignment.
            % Internalized from src/plotting/plot_lines_with_colormap.m
            if isempty(data), return; end
            n_lines = size(data, 1);
            n_colors = size(cmap, 1);
            hold on;
            for i = 1:n_lines
                color_idx = mod(i - 1, n_colors) + 1;
                plot(t, data(i, :), 'Color', cmap(color_idx, :), varargin{:});
            end
        end

        function cmap = inhibitory_colormap(n_colors)
            % INHIBITORY_COLORMAP Custom colormap for inhibitory neurons.
            % Internalized from src/plotting/inhibitory_colormap.m
            if nargin < 1, n_colors = 8; end
            base_palette = [
                0.00, 0.45, 0.74;
                0.00, 0.75, 1.00;
                0.20, 0.47, 0.62;
                0.00, 0.50, 0.50;
                0.30, 0.75, 0.93;
                0.25, 0.62, 0.75;
                0.00, 0.80, 0.80;
                0.15, 0.55, 0.65;
                ];
            n_base = size(base_palette, 1);
            if n_colors == n_base
                cmap = base_palette;
            elseif n_colors < n_base
                indices = round(linspace(1, n_base, n_colors));
                cmap = base_palette(indices, :);
            else
                x_base = linspace(1, n_colors, n_base);
                x_new = 1:n_colors;
                cmap = zeros(n_colors, 3);
                for i = 1:3
                    cmap(:, i) = interp1(x_base, base_palette(:, i), x_new, 'pchip');
                end
                cmap = max(0, min(1, cmap));
            end
        end

        function cmap = excitatory_colormap(n_colors)
            % EXCITATORY_COLORMAP Custom colormap for excitatory neurons.
            % Internalized from src/plotting/excitatory_colormap.m
            if nargin < 1, n_colors = 8; end
            base_palette = [
                1.00, 0.00, 0.00;
                1.00, 0.75, 0.00;
                0.85, 0.20, 0.45;
                0.90, 0.10, 0.60;
                0.90, 0.55, 0.00;
                0.55, 0.27, 0.27;
                0.86, 0.08, 0.24;
                0.60, 0.15, 0.45;
                ];
            n_base = size(base_palette, 1);
            if n_colors == n_base
                cmap = base_palette;
            elseif n_colors < n_base
                indices = round(linspace(1, n_base, n_colors));
                cmap = base_palette(indices, :);
            else
                x_base = linspace(1, n_colors, n_base);
                x_new = 1:n_colors;
                cmap = zeros(n_colors, 3);
                for i = 1:3
                    cmap(:, i) = interp1(x_base, base_palette(:, i), x_new, 'pchip');
                end
                cmap = max(0, min(1, cmap));
            end
        end
    end
end

