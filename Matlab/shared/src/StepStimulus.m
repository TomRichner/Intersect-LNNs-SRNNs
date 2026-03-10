classdef StepStimulus < Stimulus
    % STEPSTIMULUS Sparse step-function stimulus for SRNN simulations.
    %
    % Extracted from SRNNModel2.generate_stimulus() and generate_external_input().
    % Generates piecewise-constant step inputs where a random sparse subset
    % of neurons receive random-amplitude input during each step period.
    %
    % Usage:
    %   stim = StepStimulus();
    %   stim = StepStimulus('n_steps', 5, 'amp', 0.8);
    %
    % See also: Stimulus, SinusoidalStimulus, ESNStimulus

    properties
        n_steps = 3             % Number of step periods
        step_density = 0.2      % Fraction of neurons receiving input (legacy)
        step_density_E = 0.15   % Fraction of E neurons receiving input
        step_density_I = 0      % Fraction of I neurons receiving input
        amp = 0.5               % Amplitude of step inputs
        no_stim_pattern         % Logical vector: true = no stimulus in that step
        intrinsic_drive = []    % Constant drive added to all neurons (n × 1)
        positive_only = false   % If true, use abs(randn) for amplitudes
        u_ex_scale = 1.0        % Post-generation scaling factor
    end

    methods
        function obj = StepStimulus(varargin)
            for i = 1:2:length(varargin)
                if isprop(obj, varargin{i})
                    obj.(varargin{i}) = varargin{i+1};
                end
            end
            % Set default no_stim_pattern if not provided
            if isempty(obj.no_stim_pattern)
                obj.no_stim_pattern = false(1, obj.n_steps);
                obj.no_stim_pattern(1:2:end) = true;
            end
        end

        function build(obj, T_range, fs, n, rng_seeds, model)
            % BUILD Generate sparse step stimulus.
            %
            % Uses rng_seeds(2) for stimulus reproducibility.
            % Uses model.f, model.E_indices, model.I_indices for E/I split.

            rng(rng_seeds(2));

            dt = 1 / fs;
            T_stim = T_range(2);
            t_stim = (0:dt:T_stim)';
            nt = length(t_stim);

            % Get E/I indices from model
            n_E = round(model.f * n);
            E_idx = 1:n_E;
            I_idx = (n_E + 1):n;

            % Set intrinsic drive
            if isempty(obj.intrinsic_drive)
                id = zeros(n, 1);
            else
                id = obj.intrinsic_drive;
            end

            step_period = fix(T_stim / obj.n_steps);
            step_length = round(step_period * fs);

            % Generate random amplitudes
            if obj.positive_only
                random_sparse_step = obj.amp * abs(randn(n, obj.n_steps));
            else
                random_sparse_step = obj.amp * randn(n, obj.n_steps);
            end

            % Create sparse mask with separate E/I densities
            sparse_mask = false(n, obj.n_steps);
            sparse_mask(E_idx, :) = rand(length(E_idx), obj.n_steps) < obj.step_density_E;
            sparse_mask(I_idx, :) = rand(length(I_idx), obj.n_steps) < obj.step_density_I;

            random_sparse_step = random_sparse_step .* sparse_mask;
            random_sparse_step(:, obj.no_stim_pattern) = 0;

            % Build u_ex matrix
            u_stim = zeros(n, nt);
            for step_idx = 1:obj.n_steps
                start_idx = (step_idx - 1) * step_length + 1;
                end_idx = min(step_idx * step_length, nt);
                if start_idx > nt
                    break;
                end
                u_stim(:, start_idx:end_idx) = repmat(random_sparse_step(:, step_idx), 1, end_idx - start_idx + 1);
            end

            u_stim = u_stim + id;

            % Handle negative start time (prepend zeros for settling)
            if T_range(1) < 0
                t_pre = (T_range(1):dt:-dt)';
                u_pre = zeros(n, length(t_pre));
                obj.t_ex = [t_pre; t_stim];
                obj.u_ex = [u_pre, u_stim];
            else
                % Slice if start time is positive
                indices = t_stim >= T_range(1);
                obj.t_ex = t_stim(indices);
                obj.u_ex = u_stim(:, indices);
            end

            % Apply scaling
            obj.u_ex = obj.u_ex .* obj.u_ex_scale;

            % Build interpolant
            obj.u_interpolant = griddedInterpolant(obj.t_ex, obj.u_ex', 'linear', 'none');

            fprintf('StepStimulus built: %d time points, %d neurons\n', length(obj.t_ex), n);
        end
    end
end
