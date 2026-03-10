classdef ESNStimulus < Stimulus
    % ESNSTIMULUS Echo State Network scalar input stimulus.
    %
    % Extracted from the duplicated build_stimulus() code in
    % SRNN_ESN_reservoir and LNN_ESN_reservoir. Generates a scalar input
    % sequence (white noise, bandlimited, or 1/f^alpha) and maps it to
    % neural input via a sparse input weight vector W_in_esn.
    %
    % Usage:
    %   stim = ESNStimulus('T_wash', 2000, 'T_train', 3000, 'T_test', 3000);
    %   stim = ESNStimulus('input_type', 'bandlimited', 'u_f_cutoff', 5.0);
    %
    % See also: Stimulus, StepStimulus, ESN_reservoir

    %% ESN Protocol Properties
    properties
        T_wash = 500            % Washout samples
        T_train = 5000          % Training samples
        T_test = 5000           % Testing samples
        input_type = 'white'    % 'white', 'bandlimited', 'one_over_f'
    end

    %% Input Scaling Properties
    properties
        f_in = 0.1              % Fraction of neurons receiving input
        sigma_in = 0.1          % Std dev of nonzero input weights
        u_scale = 1.0           % Input amplitude scaling
        u_offset = 0.0          % Input DC offset
        u_alpha = 1.0           % Spectral exponent for 1/f noise
        u_f_cutoff = []         % Cutoff frequency for bandlimited (Hz), auto if empty
    end

    %% Outputs (public for ESN_reservoir access)
    properties (SetAccess = protected)
        W_in_esn                % Sparse input weight vector (n × 1)
        u_scalar                % Scalar input sequence (T_total × 1)
    end

    methods
        function obj = ESNStimulus(varargin)
            for i = 1:2:length(varargin)
                if isprop(obj, varargin{i})
                    obj.(varargin{i}) = varargin{i+1};
                end
            end
        end

        function build(obj, ~, fs, n, rng_seeds, ~)
            % BUILD Generate ESN stimulus: W_in_esn, u_scalar, u_ex, interpolant.
            %
            % Uses rng_seeds(2) for stimulus generation.

            %% 1. Generate sparse input weights
            rng(rng_seeds(1));  % Use network seed for input weight placement
            n_input = max(1, round(obj.f_in * n));
            input_indices = randperm(n, n_input);

            obj.W_in_esn = zeros(n, 1);
            obj.W_in_esn(input_indices) = obj.sigma_in * randn(n_input, 1);

            %% 2. Generate scalar input sequence
            T_total = obj.T_wash + obj.T_train + obj.T_test;
            rng(rng_seeds(2));  % Stimulus seed

            if strcmpi(obj.input_type, 'bandlimited')
                u_raw = rand(T_total, 1) - 0.5;

                if isempty(obj.u_f_cutoff)
                    f_cut = fs / 20;  % Default: reasonable cutoff
                else
                    f_cut = obj.u_f_cutoff;
                end

                f_nyq = fs / 2;
                [b_filt, a_filt] = butter(3, f_cut / f_nyq, 'low');
                u_filtered = filtfilt(b_filt, a_filt, u_raw);
                u_normalized = (u_filtered - min(u_filtered)) / (max(u_filtered) - min(u_filtered)) - 0.5;
                obj.u_scalar = obj.u_offset + obj.u_scale * u_normalized;
                fprintf('ESNStimulus: bandlimited input (f_cutoff = %.2f Hz)\n', f_cut);

            elseif strcmpi(obj.input_type, 'one_over_f')
                alpha_val = obj.u_alpha;
                N = T_total;
                X = randn(N, 1) + 1i * randn(N, 1);
                df = fs / N;
                freq = (0:(N-1))' * df;
                freq(freq > fs/2) = freq(freq > fs/2) - fs;
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
                fprintf('ESNStimulus: 1/f^%.2f noise input\n', alpha_val);

            elseif strcmpi(obj.input_type, 'white')
                obj.u_scalar = obj.u_offset + obj.u_scale * (rand(T_total, 1) - 0.5);
                fprintf('ESNStimulus: white noise input\n');
            else
                error('ESNStimulus:InvalidInputType', ...
                    'Unknown input_type ''%s''. Valid: ''white'', ''bandlimited'', ''one_over_f''', ...
                    obj.input_type);
            end

            %% 3. Map scalar to neural input (for u_ex storage / plotting)
            dt = 1 / fs;
            obj.t_ex = (0:(T_total-1))' * dt;
            obj.u_ex = obj.W_in_esn * obj.u_scalar';  % n × T (for cRNN.build_stimulus)

            %% 4. Build interpolant — returns scalar u(t)
            % Models apply W_in in their dynamics, so the interpolant
            % stores only the raw scalar input signal.
            obj.u_interpolant = griddedInterpolant(obj.t_ex, obj.u_scalar, 'linear', 'none');

            fprintf('ESNStimulus built: %d samples, %d neurons receive input\n', ...
                T_total, sum(obj.W_in_esn ~= 0));
        end
    end
end
