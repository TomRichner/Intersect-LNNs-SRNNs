classdef (Abstract) Stimulus < handle
    % STIMULUS Abstract base class for stimulus generation.
    %
    % Strategy object for cRNN: provides pluggable input generation.
    % Subclasses must implement build(), which populates t_ex, u_ex,
    % and u_interpolant.
    %
    % The build() method is called once during cRNN.build(). The outputs
    % (t_ex, u_ex, u_interpolant) are then copied to the model and captured
    % in a closure for ODE integration. The Stimulus object is NOT accessed
    % during run() — it is build-time only.
    %
    % See also: StepStimulus, SinusoidalStimulus, ESNStimulus

    properties (SetAccess = protected)
        t_ex            % Time vector (nt × 1)
        u_ex            % Input matrix (n × nt) — neural-space input
        u_interpolant   % griddedInterpolant for ODE solver
    end

    methods (Abstract)
        build(obj, T_range, fs, n, rng_seeds, model)
        % BUILD Generate stimulus.
        %
        % Inputs:
        %   T_range   - [t_start, t_end] simulation interval
        %   fs        - Sampling frequency (Hz)
        %   n         - Number of neurons
        %   rng_seeds - RNG seed array (uses element 2 for stimulus)
        %   model     - Reference to the cRNN model (for model-specific params)
        %
        % Must set: obj.t_ex, obj.u_ex, obj.u_interpolant
    end
end
