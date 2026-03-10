classdef TanhActivation < Activation
    % TANHACTIVATION Hyperbolic tangent activation function.
    %
    % Default activation for LNN models.
    %
    % Usage:
    %   act = TanhActivation();
    %   y = act.apply(x);
    %   dy = act.derivative(x);
    %
    % See also: Activation, PiecewiseSigmoid

    methods
        function y = apply(~, x)
            y = tanh(x);
        end

        function dy = derivative(~, x)
            dy = 1 - tanh(x).^2;
        end
    end
end
