classdef LNN2 < cRNN
    % LNN2 LTC network faithful to Hasani's MATLAB trajectory analysis code.
    %
    % Faithful port of:
    %   liquid_time_constant_networks/trajectory_length_analysis/ltc_def.m
    %
    % Two-dendrite model: separate feedforward and recurrent pathways.
    % Multi-layer support: n_layers layers of k neurons each.
    %
    % Dynamics for neuron i in layer j:
    %   f_ff_i  = act(W_ff{j}' * source + b_ff{j})
    %   f_rec_i = act(W_rec{j}' * x_layer_j + b_rec{j})
    %   dx_i/dt = -x_i*(1/tau_i + |f_ff_i| + |f_rec_i|)
    %             + f_ff_i * sum(E_ff{j}(:,i_local))
    %             + f_rec_i * sum(E_rec{j}(:,i_local))
    %
    % See also: LNN, cRNN

    %% LNN2-Specific Properties
    properties
        n_in = 2                    % External input dimension
        k                           % Neurons per layer (computed from n/n_layers)
        n_layers = 1                % Number of layers
        activation_name = 'tanh'    % 'tanh', 'sigmoid', 'relu', 'Htanh', 'piecewise_sigmoid'
    end

    %% Initialization Parameters
    properties
        sigma_w = sqrt(2)           % Weight distribution scale
        sigma_b = 1.0               % Bias distribution scale
    end

    %% LNN2-Specific Protected Properties
    properties (SetAccess = protected)
        W_ff                        % Cell array of feedforward weight matrices
        b_ff                        % Cell array of feedforward bias vectors
        E_ff                        % Cell array of feedforward reversal matrices
        W_rec                       % Cell array of recurrent weight matrices
        b_rec                       % Cell array of recurrent bias vectors
        E_rec                       % Cell array of recurrent reversal matrices
        tau                         % Time constant vector (n × 1)
    end

    %% Constructor
    methods
        function obj = LNN2(varargin)
            obj@cRNN();

            % Defaults
            obj.n = 100;
            obj.T_range = [0, 10];
            obj.store_full_state = true;
            obj.lya_method = 'none';
            obj.rng_seeds = [1 1];

            % Default strategies
            obj.activation = TanhActivation();
            obj.stimulus = SinusoidalStimulus('n_in', 2);
            obj.connectivity = RMTConnectivity();

            % Parse name-value pairs
            remaining = {};
            for i = 1:2:length(varargin)
                prop = varargin{i};
                val = varargin{i+1};

                if strcmp(prop, 'n_in')
                    obj.n_in = val;
                    obj.stimulus = SinusoidalStimulus('n_in', val);
                elseif strcmp(prop, 'activation_name')
                    obj.activation_name = val;
                    obj.activation = LNN2.make_activation(val);
                else
                    remaining = [remaining, {prop, val}]; %#ok<AGROW>
                end
            end

            obj.parse_name_value_pairs(remaining);
        end
    end

    %% Abstract Method Implementations (from cRNN)
    methods
        function rhs = get_rhs(obj, params)
            rhs = @(t, x) LNN2.dynamics_ltc2(t, x, params);
        end

        function features = get_readout_features(obj)
            if isempty(obj.state_out)
                error('LNN2:NoState', 'No state data available.');
            end
            features = obj.state_out';  % n × T
        end

        function J = get_jacobian(~, ~, ~)
            error('LNN2:NotImplemented', ...
                'Jacobian not yet implemented for LNN2. Use numerical Jacobian or implement analytically.');
        end

        function initialize_state(obj)
            obj.S0 = 0.01 * randn(obj.n, 1);
        end
    end

    %% Overridden Methods
    methods
        function params = get_params(obj)
            params = get_params@cRNN(obj);

            params.n_in = obj.n_in;
            params.k = obj.k;
            params.n_layers = obj.n_layers;
            params.activation_name = obj.activation_name;

            if ~isempty(obj.W_ff),  params.W_ff  = obj.W_ff;  end
            if ~isempty(obj.b_ff),  params.b_ff  = obj.b_ff;  end
            if ~isempty(obj.E_ff),  params.E_ff  = obj.E_ff;  end
            if ~isempty(obj.W_rec), params.W_rec = obj.W_rec; end
            if ~isempty(obj.b_rec), params.b_rec = obj.b_rec; end
            if ~isempty(obj.E_rec), params.E_rec = obj.E_rec; end
            if ~isempty(obj.tau),   params.tau   = obj.tau;   end
        end
    end

    %% Protected Build Sub-Methods
    methods (Access = protected)
        function build_network(obj)
            % Do NOT call base class build_network (we don't use RMT for W).
            % Initialize all per-layer weight matrices from Gaussian.

            if isempty(obj.k)
                obj.k = obj.n;  % single layer default
            end
            if mod(obj.n, obj.k) ~= 0
                error('LNN2:InvalidDims', 'n (%d) must be divisible by k (%d)', obj.n, obj.k);
            end
            obj.n_layers = obj.n / obj.k;

            k = obj.k;
            n_in = obj.n_in;
            n_layers = obj.n_layers;
            w_scale = obj.sigma_w * sqrt(2) / k;

            rng_state = rng(obj.rng_seeds(1));

            % Allocate cell arrays
            obj.W_ff  = cell(n_layers, 1);
            obj.b_ff  = cell(n_layers, 1);
            obj.E_ff  = cell(n_layers, 1);
            obj.W_rec = cell(n_layers, 1);
            obj.b_rec = cell(n_layers, 1);
            obj.E_rec = cell(n_layers, 1);

            for j = 1:n_layers
                if j == 1
                    ff_in = n_in;
                else
                    ff_in = k;
                end

                obj.W_ff{j}  = w_scale * randn(ff_in, k);
                obj.b_ff{j}  = obj.sigma_b * randn(k, 1);
                obj.E_ff{j}  = w_scale * randn(k, k);
                obj.W_rec{j} = w_scale * randn(k, k);
                obj.b_rec{j} = obj.sigma_b * randn(k, 1);
                obj.E_rec{j} = w_scale * randn(k, k);
            end

            % Time constants: abs(Gaussian), positive
            obj.tau = abs(obj.sigma_b * randn(obj.n, 1));
            obj.tau = max(obj.tau, 1e-4);

            % W and W_in not used by LNN2, but set for cRNN compatibility
            obj.W = [];
            obj.W_in = [];

            rng(rng_state);

            fprintf('LNN2 network initialized: n=%d, k=%d, n_layers=%d, n_in=%d, activation=%s\n', ...
                obj.n, obj.k, obj.n_layers, obj.n_in, obj.activation_name);
        end

        function decimate_and_unpack(obj)
            deci = obj.plot_deci;
            t_d = obj.t_out(1:deci:end);
            x_d = obj.state_out(1:deci:end, :);
            nt_d = length(t_d);

            params = obj.cached_params;
            k = params.k;
            n_layers = params.n_layers;

            % Compute u, f_ff, f_rec at decimated times
            u_d = zeros(nt_d, params.n_in);
            f_ff_d = zeros(nt_d, params.n);
            f_rec_d = zeros(nt_d, params.n);
            tau_sys_d = zeros(nt_d, params.n);

            for t_idx = 1:nt_d
                I_t = obj.u_interpolant(t_d(t_idx))';
                u_d(t_idx, :) = I_t';
                x_t = x_d(t_idx, :)';

                for j = 1:n_layers
                    idx = ((j-1)*k+1):(j*k);
                    x_layer = x_t(idx);

                    if j == 1
                        source = I_t;
                    else
                        idx_prev = ((j-2)*k+1):((j-1)*k);
                        source = x_t(idx_prev);
                    end

                    z_ff = params.W_ff{j}' * source + params.b_ff{j};
                    f_ff_j = params.activation.apply(z_ff);
                    f_ff_d(t_idx, idx) = f_ff_j';

                    z_rec = params.W_rec{j}' * x_layer + params.b_rec{j};
                    f_rec_j = params.activation.apply(z_rec);
                    f_rec_d(t_idx, idx) = f_rec_j';

                    tau_i = params.tau(idx);
                    tau_sys_d(t_idx, idx) = (tau_i ./ (1 + tau_i .* (abs(f_ff_j) + abs(f_rec_j))))';
                end
            end

            obj.plot_data = struct();
            obj.plot_data.t = t_d;
            obj.plot_data.x = x_d;
            obj.plot_data.u = u_d;
            obj.plot_data.f_ff = f_ff_d;
            obj.plot_data.f_rec = f_rec_d;
            obj.plot_data.tau_sys = tau_sys_d;
        end
    end

    %% Static ODE RHS
    methods (Static)
        function dxdt = dynamics_ltc2(t, x, params)
            k = params.k;
            n_layers = params.n_layers;
            N = params.n;

            I_t = params.u_interpolant(t)';

            dxdt = zeros(N, 1);

            for j = 1:n_layers
                idx = ((j-1)*k+1):(j*k);
                x_layer = x(idx);

                % Feedforward dendrite
                if j == 1
                    source = I_t;
                else
                    idx_prev = ((j-2)*k+1):((j-1)*k);
                    source = x(idx_prev);
                end

                z_ff = params.W_ff{j}' * source + params.b_ff{j};
                f_ff = params.activation.apply(z_ff);

                % Recurrent dendrite
                z_rec = params.W_rec{j}' * x_layer + params.b_rec{j};
                f_rec = params.activation.apply(z_rec);

                % Drive: f_i * sum(E(:,i)) for each neuron i
                drive_ff  = f_ff .* sum(params.E_ff{j}, 1)';
                drive_rec = f_rec .* sum(params.E_rec{j}, 1)';

                % Dynamics
                tau_layer = params.tau(idx);
                dxdt(idx) = -x_layer .* (1 ./ tau_layer + abs(f_ff) + abs(f_rec)) ...
                            + drive_ff + drive_rec;
            end
        end
    end

    %% Static Helpers
    methods (Static)
        function act = make_activation(name)
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
                    error('LNN2:UnknownActivation', 'Unknown activation: %s', name);
            end
        end

        function cmap = default_colormap(n_neurons)
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
