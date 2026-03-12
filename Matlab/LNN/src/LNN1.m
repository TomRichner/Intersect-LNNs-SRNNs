classdef LNN1 < cRNN
    % LNN1 LTC network faithful to Hasani's Python training code.
    %
    % Faithful port of:
    %   liquid_time_constant_networks/experiments_with_ltcs/ltc_model.py
    %
    % Per-synapse conductance-based model with separate sensory and
    % recurrent pathways. Per-synapse sigmoid gates: σ(σ_ji·(v_j − μ_ji))
    %
    % Continuous ODE:
    %   cm_i · dv_i/dt = g_leak_i · (v_leak_i − v_i)
    %                   + Σ_j g_rec_ji · (E_rec_ji − v_i)
    %                   + Σ_k g_sens_ki · (E_sens_ki − v_i)
    %
    % Fused semi-implicit solver:
    %   v_i ← (cm·v + gleak·vleak + Σ g·E) / (cm + gleak + Σ g)
    %
    % See also: LNN, LNN2, cRNN

    %% LNN1-Specific Properties
    properties
        n_in = 2                    % Input dimension
        solver_mode = 'semi_implicit'  % 'semi_implicit', 'explicit', 'runge_kutta'
        ode_solver_unfolds = 6      % Number of ODE sub-steps per dt
    end

    %% LNN1 Protected Properties (per-synapse parameters)
    properties (SetAccess = protected)
        % Recurrent (N × N)
        W_syn                       % Synaptic weights (non-negative)
        mu_syn                      % Gate centers
        sigma_syn                   % Gate widths
        erev                        % Reversal potentials (±1)
        % Sensory (M × N)
        sensory_W                   % Sensory weights (non-negative)
        sensory_mu                  % Sensory gate centers
        sensory_sigma               % Sensory gate widths
        sensory_erev                % Sensory reversal potentials
        % Leak & membrane
        vleak                       % Leak reversal (N × 1)
        gleak                       % Leak conductance (N × 1)
        cm                          % Membrane capacitance (N × 1)
        % Input mapping
        input_w                     % Input affine weight (M × 1)
        input_b                     % Input affine bias (M × 1)
    end

    %% Constructor
    methods
        function obj = LNN1(varargin)
            obj@cRNN();

            % Defaults
            obj.n = 32;
            obj.T_range = [0, 10];
            obj.store_full_state = true;
            obj.lya_method = 'none';
            obj.rng_seeds = [1 1];

            % Default strategies
            obj.activation = [];  % Not used (per-synapse sigmoid)
            obj.stimulus = SinusoidalStimulus('n_in', 2);
            obj.connectivity = [];  % Not used

            % Parse
            remaining = {};
            for i = 1:2:length(varargin)
                prop = varargin{i};
                val = varargin{i+1};

                if strcmp(prop, 'n_in')
                    obj.n_in = val;
                    obj.stimulus = SinusoidalStimulus('n_in', val);
                else
                    remaining = [remaining, {prop, val}]; %#ok<AGROW>
                end
            end

            obj.parse_name_value_pairs(remaining);
        end
    end

    %% Abstract Method Implementations
    methods
        function rhs = get_rhs(obj, params)
            rhs = @(t, v) LNN1.dynamics_ltc1(t, v, params);
        end

        function features = get_readout_features(obj)
            if isempty(obj.state_out)
                error('LNN1:NoState', 'No state data available.');
            end
            features = obj.state_out';
        end

        function J = get_jacobian(~, ~, ~)
            error('LNN1:NotImplemented', ...
                'Jacobian not yet implemented for LNN1.');
        end

        function initialize_state(obj)
            obj.S0 = zeros(obj.n, 1);  % Python default: zero init
        end
    end

    %% Overridden Methods
    methods
        function run(obj)
            if ~obj.is_built
                error('LNN1:NotBuilt', 'Call build() first.');
            end

            if strcmpi(obj.solver_mode, 'semi_implicit') || ...
               strcmpi(obj.solver_mode, 'explicit') || ...
               strcmpi(obj.solver_mode, 'runge_kutta')
                % Fused solver (all three modes)
                params = obj.cached_params;
                fprintf('Running LNN1 with %s solver (%d unfolds)...\n', ...
                    obj.solver_mode, obj.ode_solver_unfolds);

                tic
                [t_raw, x_raw] = obj.run_fused(params);
                run_time = toc;
                fprintf('Integration complete in %.2f seconds.\n', run_time);

                obj.t_out = t_raw;
                obj.state_out = x_raw;

                if obj.store_decimated_state
                    obj.decimate_and_unpack();
                end
                if ~obj.store_full_state
                    obj.state_out = [];
                end
                obj.has_run = true;
            else
                % Standard ODE mode
                run@cRNN(obj);
            end
        end

        function params = get_params(obj)
            params = struct();
            params.n = obj.n;
            params.n_in = obj.n_in;
            params.fs = obj.fs;

            if ~isempty(obj.W_syn),       params.W_syn = obj.W_syn; end
            if ~isempty(obj.mu_syn),       params.mu_syn = obj.mu_syn; end
            if ~isempty(obj.sigma_syn),    params.sigma_syn = obj.sigma_syn; end
            if ~isempty(obj.erev),         params.erev = obj.erev; end
            if ~isempty(obj.sensory_W),    params.sensory_W = obj.sensory_W; end
            if ~isempty(obj.sensory_mu),   params.sensory_mu = obj.sensory_mu; end
            if ~isempty(obj.sensory_sigma),params.sensory_sigma = obj.sensory_sigma; end
            if ~isempty(obj.sensory_erev), params.sensory_erev = obj.sensory_erev; end
            if ~isempty(obj.vleak),        params.vleak = obj.vleak; end
            if ~isempty(obj.gleak),        params.gleak = obj.gleak; end
            if ~isempty(obj.cm),           params.cm = obj.cm; end
            if ~isempty(obj.input_w),      params.input_w = obj.input_w; end
            if ~isempty(obj.input_b),      params.input_b = obj.input_b; end
        end
    end

    %% Protected Build Methods
    methods (Access = protected)
        function build_network(obj)
            % Initialize all per-synapse parameters (no RMT connectivity).
            n = obj.n;
            m = obj.n_in;

            rng_state = rng(obj.rng_seeds(1));

            % Recurrent (N × N)
            obj.W_syn     = rand(n, n) * 0.99 + 0.01;       % [0.01, 1.0]
            obj.mu_syn    = rand(n, n) * 0.5 + 0.3;         % [0.3, 0.8]
            obj.sigma_syn = rand(n, n) * 5.0 + 3.0;         % [3.0, 8.0]
            obj.erev      = 2 * randi([0, 1], n, n) - 1;    % ±1

            % Sensory (M × N)
            obj.sensory_W     = rand(m, n) * 0.99 + 0.01;
            obj.sensory_mu    = rand(m, n) * 0.5 + 0.3;
            obj.sensory_sigma = rand(m, n) * 5.0 + 3.0;
            obj.sensory_erev  = 2 * randi([0, 1], m, n) - 1;

            % Leak & membrane
            obj.vleak = rand(n, 1) * 0.4 - 0.2;             % [-0.2, 0.2]
            obj.gleak = ones(n, 1);                           % 1.0
            obj.cm    = 0.5 * ones(n, 1);                     % 0.5

            % Input mapping
            obj.input_w = ones(m, 1);
            obj.input_b = zeros(m, 1);

            % W and W_in not used by LNN1
            obj.W = [];
            obj.W_in = [];

            rng(rng_state);

            fprintf('LNN1 network initialized: n=%d, n_in=%d (per-synapse model)\n', n, m);
        end

        function [t_out_fused, x_out_fused] = run_fused(obj, params)
            dt = 1 / obj.fs;
            nt = length(obj.t_ex);

            x_out_fused = zeros(nt, obj.n);
            x_out_fused(1, :) = obj.S0';
            v = obj.S0;

            params.u_interpolant = obj.u_interpolant;

            for k_step = 1:(nt - 1)
                I_raw = obj.u_interpolant(obj.t_ex(k_step))';
                I_mapped = params.input_w .* I_raw + params.input_b;

                if strcmpi(obj.solver_mode, 'semi_implicit')
                    for s = 1:obj.ode_solver_unfolds
                        v = LNN1.fused_step_semi_implicit(v, I_mapped, params);
                    end
                elseif strcmpi(obj.solver_mode, 'explicit')
                    for s = 1:obj.ode_solver_unfolds
                        v = v + 0.1 * LNN1.f_prime(v, I_mapped, params);
                    end
                elseif strcmpi(obj.solver_mode, 'runge_kutta')
                    h = 0.1;
                    for s = 1:obj.ode_solver_unfolds
                        k1 = h * LNN1.f_prime(v, I_mapped, params);
                        k2 = h * LNN1.f_prime(v + 0.5*k1, I_mapped, params);
                        k3 = h * LNN1.f_prime(v + 0.5*k2, I_mapped, params);
                        k4 = h * LNN1.f_prime(v + k3, I_mapped, params);
                        v = v + (1/6) * (k1 + 2*k2 + 2*k3 + k4);
                    end
                end

                x_out_fused(k_step + 1, :) = v';
            end

            t_out_fused = obj.t_ex;
        end

        function decimate_and_unpack(obj)
            deci = obj.plot_deci;
            t_d = obj.t_out(1:deci:end);
            x_d = obj.state_out(1:deci:end, :);

            % Compute inputs at decimated times
            nt_d = length(t_d);
            u_d = zeros(nt_d, obj.n_in);
            for t_idx = 1:nt_d
                I_raw = obj.u_interpolant(t_d(t_idx))';
                u_d(t_idx, :) = I_raw';
            end

            obj.plot_data = struct();
            obj.plot_data.t = t_d;
            obj.plot_data.x = x_d;
            obj.plot_data.u = u_d;
        end
    end

    %% Static Methods
    methods (Static)
        function gate = per_synapse_sigmoid(v_pre, mu, sigma)
            % PER_SYNAPSE_SIGMOID σ(σ_ji · (v_j − μ_ji))
            % v_pre: (n×1), mu: (n×n) or (m×n), sigma: (n×n) or (m×n)
            v = v_pre(:);  % column
            gate = 1 ./ (1 + exp(-sigma .* (v - mu)));
        end

        function v_new = fused_step_semi_implicit(v, I_mapped, params)
            % Sensory pathway
            s_gate = LNN1.per_synapse_sigmoid(I_mapped, params.sensory_mu, params.sensory_sigma);
            s_w_act = params.sensory_W .* s_gate;
            s_rev_act = s_w_act .* params.sensory_erev;
            w_num_s = sum(s_rev_act, 1)';
            w_den_s = sum(s_w_act, 1)';

            % Recurrent pathway
            r_gate = LNN1.per_synapse_sigmoid(v, params.mu_syn, params.sigma_syn);
            r_w_act = params.W_syn .* r_gate;
            r_rev_act = r_w_act .* params.erev;
            w_num = sum(r_rev_act, 1)' + w_num_s;
            w_den = sum(r_w_act, 1)' + w_den_s;

            % Fused update
            numerator = params.cm .* v + params.gleak .* params.vleak + w_num;
            denominator = params.cm + params.gleak + w_den;
            v_new = numerator ./ denominator;
        end

        function dvdt = f_prime(v, I_mapped, params)
            % Continuous ODE RHS for explicit/RK solvers
            s_gate = LNN1.per_synapse_sigmoid(I_mapped, params.sensory_mu, params.sensory_sigma);
            s_w_act = params.sensory_W .* s_gate;
            w_reduced_s = sum(s_w_act, 1)';
            s_in = params.sensory_erev .* s_w_act;

            r_gate = LNN1.per_synapse_sigmoid(v, params.mu_syn, params.sigma_syn);
            r_w_act = params.W_syn .* r_gate;
            w_reduced_r = sum(r_w_act, 1)';
            r_in = params.erev .* r_w_act;

            sum_in = sum(s_in, 1)' - v .* w_reduced_r ...
                   + sum(r_in, 1)' - v .* w_reduced_s;

            dvdt = (1 ./ params.cm) .* (params.gleak .* (params.vleak - v) + sum_in);
        end

        function dxdt = dynamics_ltc1(t, v, params)
            I_raw = params.u_interpolant(t)';
            I_mapped = params.input_w .* I_raw + params.input_b;
            dxdt = LNN1.f_prime(v, I_mapped, params);
        end
    end
end
