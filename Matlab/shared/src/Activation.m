classdef (Abstract) Activation < handle
    % ACTIVATION Abstract base class for activation functions.
    %
    % Strategy object for cRNN: provides a pluggable nonlinearity.
    % Subclasses must implement apply() and derivative().
    %
    % See also: PiecewiseSigmoid, TanhActivation, SigmoidActivation, ReLUActivation

    methods (Abstract)
        y = apply(obj, x)           % Forward pass: y = sigma(x)
        dy = derivative(obj, x)     % Derivative:   dy = sigma'(x)
    end
end
