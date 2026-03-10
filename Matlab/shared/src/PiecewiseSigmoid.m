classdef PiecewiseSigmoid < Activation
    % PIECEWISESIGMOID Hard sigmoid with rounded (quadratic) corners.
    %
    % Piecewise linear/quadratic sigmoid bounded in [0, 1].
    % Extracted from SRNNModel2.piecewiseSigmoid and LNN.piecewiseSigmoid
    % (previously duplicated in both classes).
    %
    % Parameters:
    %   S_a - Linear fraction parameter (0 to 1), default 0.9
    %   S_c - Center/shift parameter, default 0.35
    %
    % Usage:
    %   act = PiecewiseSigmoid();
    %   act = PiecewiseSigmoid('S_a', 0.9, 'S_c', 0.35);
    %   y = act.apply(x);
    %   dy = act.derivative(x);
    %
    % See also: Activation, TanhActivation

    properties
        S_a = 0.9       % Linear fraction parameter (0 to 1)
        S_c = 0.35      % Center/shift parameter
    end

    methods
        function obj = PiecewiseSigmoid(varargin)
            for i = 1:2:length(varargin)
                if isprop(obj, varargin{i})
                    obj.(varargin{i}) = varargin{i+1};
                end
            end
        end

        function y = apply(obj, x)
            % APPLY Piecewise sigmoid forward pass.

            a = obj.S_a;
            c = obj.S_c;

            if a < 0 || a > 1
                error('PiecewiseSigmoid:InvalidParam', 'S_a must be between 0 and 1.');
            end
            a = a / 2;

            if a == 0.5
                y_linear = (x - c) + 0.5;
                y = min(max(y_linear, 0), 1);
            else
                y = zeros(size(x));
                k = 0.5 / (1 - 2*a);
                x1 = c + a - 1;
                x2 = c - a;
                x3 = c + a;
                x4 = c + 1 - a;

                mask_left_quad  = (x >= x1) & (x < x2);
                mask_linear     = (x >= x2) & (x <= x3);
                mask_right_quad = (x > x3) & (x <= x4);
                mask_right_sat  = (x > x4);

                if any(mask_left_quad, 'all')
                    y(mask_left_quad) = k * (x(mask_left_quad) - x1).^2;
                end
                if any(mask_linear, 'all')
                    y(mask_linear) = (x(mask_linear) - c) + 0.5;
                end
                if any(mask_right_quad, 'all')
                    y(mask_right_quad) = 1 - k * (x(mask_right_quad) - x4).^2;
                end
                if any(mask_right_sat, 'all')
                    y(mask_right_sat) = 1;
                end
            end
        end

        function dy = derivative(obj, x)
            % DERIVATIVE Piecewise sigmoid derivative.

            a = obj.S_a;
            c = obj.S_c;
            a = a / 2;

            if a == 0.5
                dy = double((x - c + 0.5) >= 0 & (x - c + 0.5) <= 1);
            else
                dy = zeros(size(x));
                k = 0.5 / (1 - 2*a);
                x1 = c + a - 1;
                x2 = c - a;
                x3 = c + a;
                x4 = c + 1 - a;

                mask_left_quad  = (x >= x1) & (x < x2);
                mask_linear     = (x >= x2) & (x <= x3);
                mask_right_quad = (x > x3) & (x <= x4);

                if any(mask_left_quad, 'all')
                    dy(mask_left_quad) = 2 * k * (x(mask_left_quad) - x1);
                end
                if any(mask_linear, 'all')
                    dy(mask_linear) = 1;
                end
                if any(mask_right_quad, 'all')
                    dy(mask_right_quad) = -2 * k * (x(mask_right_quad) - x4);
                end
            end
        end
    end
end
