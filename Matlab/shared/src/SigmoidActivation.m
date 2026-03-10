classdef SigmoidActivation < Activation
    % SIGMOIDACTIVATION Standard logistic sigmoid: 1/(1+exp(-x)).
    %
    % Usage:
    %   act = SigmoidActivation();
    %   y = act.apply(x);
    %   dy = act.derivative(x);
    %
    % See also: Activation, TanhActivation

    methods
        function y = apply(~, x)
            y = 1 ./ (1 + exp(-x));
        end

        function dy = derivative(obj, x)
            y = obj.apply(x);
            dy = y .* (1 - y);
        end
    end
end
