classdef ReLUActivation < Activation
    % RELUACTIVATION Rectified linear unit: max(0, x).
    %
    % Usage:
    %   act = ReLUActivation();
    %   y = act.apply(x);
    %   dy = act.derivative(x);
    %
    % See also: Activation, TanhActivation

    methods
        function y = apply(~, x)
            y = max(0, x);
        end

        function dy = derivative(~, x)
            dy = double(x > 0);
        end
    end
end
