classdef RMTConnectivity < Connectivity
    % RMTCONNECTIVITY Random Matrix Theory connectivity strategy.
    %
    % Generates a recurrent weight matrix W using the RMTMatrix class
    % (Harris 2023). Parameterized by tilde-notation means and standard
    % deviations for excitatory and inhibitory populations.
    %
    % Usage:
    %   conn = RMTConnectivity();                        % All defaults
    %   conn = RMTConnectivity('E_W', 0.5);              % Mean offset
    %   conn = RMTConnectivity('rescale_by_abscissa', true);
    %   conn.build(300, 0.5, 100, 42);
    %
    % See also: Connectivity, RMTMatrix, cRNN

    %% RMT Parameters (tilde notation — Harris 2023)
    properties
        mu_E_tilde              % Normalized excitatory mean
        mu_I_tilde              % Normalized inhibitory mean
        sigma_E_tilde           % Normalized excitatory std dev
        sigma_I_tilde           % Normalized inhibitory std dev
        E_W = 0                 % Mean offset: added to both mu_E_tilde and mu_I_tilde
        zrs_mode = 'none'       % ZRS mode: 'none', 'ZRS', 'SZRS', 'Partial_SZRS'
        rescale_by_abscissa = false  % Whether to rescale W so abscissa = level_of_chaos
    end

    %% Computed Properties (available after build)
    properties (Dependent)
        default_val             % Normalization factor F = 1/sqrt(N*alpha*(2-alpha))
        mu_se                   % Sparse excitatory mean
        mu_si                   % Sparse inhibitory mean
        sigma_se                % Sparse excitatory std dev
        sigma_si                % Sparse inhibitory std dev
        R                       % Theoretical spectral radius (Harris 2023 Eq 18)
    end

    %% Stored dimensions (set during build)
    properties (SetAccess = protected)
        n_stored                % Number of neurons (cached from build)
        f_stored                % Fraction excitatory (cached from build)
        indegree_stored         % In-degree (cached from build)
    end

    %% Constructor
    methods
        function obj = RMTConnectivity(varargin)
            % RMTCONNECTIVITY Constructor with name-value pairs.
            for i = 1:2:length(varargin)
                if isprop(obj, varargin{i})
                    obj.(varargin{i}) = varargin{i+1};
                else
                    warning('RMTConnectivity:UnknownProperty', ...
                        'Unknown property: %s', varargin{i});
                end
            end
        end
    end

    %% Build
    methods
        function build(obj, n, f, indegree, rng_seed)
            % BUILD Generate W using RMTMatrix.
            %
            % Inputs:
            %   n        - Number of neurons
            %   f        - Fraction excitatory
            %   indegree - Expected in-degree ([] = fully connected)
            %   rng_seed - RNG seed for reproducibility

            rng(rng_seed);

            % Store dimensions for dependent property computation
            obj.n_stored = n;
            obj.f_stored = f;
            if isempty(indegree)
                indegree = n;
            end
            obj.indegree_stored = indegree;

            % Compute defaults for tilde parameters
            alph = indegree / n;
            F = 1 / sqrt(n * alph * (2 - alph));

            if isempty(obj.mu_E_tilde),    obj.mu_E_tilde = 3 * F;   end
            if isempty(obj.mu_I_tilde),    obj.mu_I_tilde = -4 * F;  end
            if isempty(obj.sigma_E_tilde), obj.sigma_E_tilde = F;    end
            if isempty(obj.sigma_I_tilde), obj.sigma_I_tilde = F;    end

            % Create W using RMTMatrix
            rmt = RMTMatrix(n);
            rmt.alpha = alph;
            rmt.f = f;
            rmt.mu_tilde_e = obj.mu_E_tilde + obj.E_W;
            rmt.mu_tilde_i = obj.mu_I_tilde + obj.E_W;
            rmt.sigma_tilde_e = obj.sigma_E_tilde;
            rmt.sigma_tilde_i = obj.sigma_I_tilde;
            rmt.zrs_mode = obj.zrs_mode;

            obj.W = obj.level_of_chaos * rmt.W;

            % Rescale by spectral abscissa if requested
            if obj.rescale_by_abscissa
                W_eigs = eig(obj.W);
                abscissa_0 = max(real(W_eigs));
                if abs(abscissa_0) > eps
                    obj.W = obj.W * (obj.level_of_chaos / abscissa_0);
                end
                W_eigs_rescaled = eig(obj.W);
                fprintf('Rescaled by abscissa: spectral radius = %.3f, abscissa = %.3f\n', ...
                    max(abs(W_eigs_rescaled)), max(real(W_eigs_rescaled)));
            else
                W_eigs = eig(obj.W);
                fprintf('W created: spectral radius = %.3f, abscissa = %.3f\n', ...
                    max(abs(W_eigs)), max(real(W_eigs)));
            end

            fprintf('Theoretical R = %.3f\n', obj.R);
            obj.is_built = true;
        end

        function params = get_params(obj)
            % GET_PARAMS Return RMT connectivity parameters.
            params = get_params@Connectivity(obj);
            params.mu_E_tilde = obj.mu_E_tilde;
            params.mu_I_tilde = obj.mu_I_tilde;
            params.sigma_E_tilde = obj.sigma_E_tilde;
            params.sigma_I_tilde = obj.sigma_I_tilde;
            params.E_W = obj.E_W;
            params.zrs_mode = obj.zrs_mode;
            params.rescale_by_abscissa = obj.rescale_by_abscissa;
            params.R = obj.R;
            params.default_val = obj.default_val;
        end
    end

    %% Dependent Property Getters
    methods
        function val = get.default_val(obj)
            % DEFAULT_VAL Normalization factor F = 1/sqrt(N*alpha*(2-alpha))
            if isempty(obj.n_stored) || isempty(obj.indegree_stored)
                val = NaN; return;
            end
            alph = obj.indegree_stored / obj.n_stored;
            val = 1 / sqrt(obj.n_stored * alph * (2 - alph));
        end

        function val = get.mu_se(obj)
            if isempty(obj.mu_E_tilde) || isempty(obj.n_stored)
                val = NaN; return;
            end
            alph = obj.indegree_stored / obj.n_stored;
            val = alph * (obj.mu_E_tilde + obj.E_W);
        end

        function val = get.mu_si(obj)
            if isempty(obj.mu_I_tilde) || isempty(obj.n_stored)
                val = NaN; return;
            end
            alph = obj.indegree_stored / obj.n_stored;
            val = alph * (obj.mu_I_tilde + obj.E_W);
        end

        function val = get.sigma_se(obj)
            if isempty(obj.sigma_E_tilde) || isempty(obj.mu_E_tilde) || isempty(obj.n_stored)
                val = NaN; return;
            end
            alph = obj.indegree_stored / obj.n_stored;
            mu_eff = obj.mu_E_tilde + obj.E_W;
            val = sqrt(alph * (1 - alph) * mu_eff^2 + alph * obj.sigma_E_tilde^2);
        end

        function val = get.sigma_si(obj)
            if isempty(obj.sigma_I_tilde) || isempty(obj.mu_I_tilde) || isempty(obj.n_stored)
                val = NaN; return;
            end
            alph = obj.indegree_stored / obj.n_stored;
            mu_eff = obj.mu_I_tilde + obj.E_W;
            val = sqrt(alph * (1 - alph) * mu_eff^2 + alph * obj.sigma_I_tilde^2);
        end

        function val = get.R(obj)
            se = obj.sigma_se;
            si = obj.sigma_si;
            if isnan(se) || isnan(si) || isempty(obj.n_stored) || isempty(obj.f_stored)
                val = NaN; return;
            end
            val = sqrt(obj.n_stored * (obj.f_stored * se^2 + (1 - obj.f_stored) * si^2)) * obj.level_of_chaos;
        end
    end
end
