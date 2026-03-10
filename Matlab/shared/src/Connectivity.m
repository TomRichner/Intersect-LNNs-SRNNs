classdef (Abstract) Connectivity < handle
    % CONNECTIVITY Abstract base class for network connectivity strategies.
    %
    % Subclasses implement build() to generate the recurrent weight matrix W.
    % Each connectivity scheme can define its own parameters and analytics.
    %
    % Usage (via concrete subclass):
    %   conn = RMTConnectivity('level_of_chaos', 2.0, 'mu_E_tilde', 0.3);
    %   conn.build(300, 0.5, 100, 42);  % n, f, indegree, rng_seed
    %   W = conn.W;
    %
    % See also: RMTConnectivity, cRNN

    %% Outputs
    properties (SetAccess = protected)
        W                       % Generated weight matrix (n × n)
        is_built = false        % Whether build() has been called
    end

    %% Common connectivity parameters
    properties
        level_of_chaos = 1.0    % Scaling factor for W (spectral radius control)
    end

    %% Abstract methods
    methods (Abstract)
        build(obj, n, f, indegree, rng_seed)
        % BUILD Generate the weight matrix W.
        %
        % Inputs:
        %   n        - Number of neurons
        %   f        - Fraction excitatory
        %   indegree - Expected in-degree ([] = fully connected)
        %   rng_seed - RNG seed for reproducibility
    end

    %% Common methods
    methods
        function params = get_params(obj)
            % GET_PARAMS Return connectivity parameters as struct.
            % Subclasses should call get_params@Connectivity(obj) and add fields.
            params = struct();
            params.level_of_chaos = obj.level_of_chaos;
        end
    end
end
