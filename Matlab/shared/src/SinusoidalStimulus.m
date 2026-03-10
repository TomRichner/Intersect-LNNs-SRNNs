classdef SinusoidalStimulus < Stimulus
    % SINUSOIDALSTIMULUS Sinusoidal or custom function stimulus.
    %
    % Extracted from LNN.build_stimulus(). Generates a default sinusoidal
    % circular trajectory for multi-input networks, or accepts a custom
    % input function handle.
    %
    % Usage:
    %   stim = SinusoidalStimulus();
    %   stim = SinusoidalStimulus('freq', 2.0, 'n_in', 3);
    %   stim = SinusoidalStimulus('input_func', @(t) sin(t));
    %
    % See also: Stimulus, StepStimulus, ESNStimulus

    properties
        n_in = 2                % Input dimension
        freq = 1.0              % Frequency of sinusoidal input (Hz)
        input_func              % Custom input function: @(t) -> (n_in × 1)
    end

    methods
        function obj = SinusoidalStimulus(varargin)
            for i = 1:2:length(varargin)
                if isprop(obj, varargin{i})
                    obj.(varargin{i}) = varargin{i+1};
                end
            end
        end

        function build(obj, T_range, fs, ~, ~, ~)
            % BUILD Generate sinusoidal or custom input stimulus.

            dt = 1 / fs;
            obj.t_ex = (T_range(1):dt:T_range(2))';
            nt = length(obj.t_ex);

            if ~isempty(obj.input_func)
                % Custom input function
                obj.u_ex = zeros(obj.n_in, nt);
                for k = 1:nt
                    obj.u_ex(:, k) = obj.input_func(obj.t_ex(k));
                end
            else
                % Default: sinusoidal circular trajectory
                if obj.n_in >= 2
                    obj.u_ex = zeros(obj.n_in, nt);
                    obj.u_ex(1, :) = sin(2 * pi * obj.freq * obj.t_ex');
                    obj.u_ex(2, :) = cos(2 * pi * obj.freq * obj.t_ex');
                    % Remaining inputs are zero
                elseif obj.n_in == 1
                    obj.u_ex = sin(2 * pi * obj.freq * obj.t_ex');
                end
            end

            % Build interpolant: u_ex is (n_in × nt), interpolant needs (nt × n_in)
            obj.u_interpolant = griddedInterpolant(obj.t_ex, obj.u_ex', 'linear', 'nearest');

            fprintf('SinusoidalStimulus built: T=[%.1f, %.1f] s, %d time steps, n_in=%d\n', ...
                T_range(1), T_range(2), nt, obj.n_in);
        end
    end
end
