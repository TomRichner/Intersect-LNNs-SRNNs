classdef SRNNComparison < handle
    % SRNNCOMPARISON Compare multiple SRNNModel2 simulations.
    %
    % Collects built/run SRNNModel2 objects and provides tools for
    % visual and quantitative comparison: parameter diffs, trajectory
    % overlays, Lyapunov exponent comparison, and summary tables.
    %
    % Usage:
    %   comp = SRNNComparison();
    %   comp.add(model1, 'ode45');
    %   comp.add(model2, 'fused(6)');
    %   comp.summary();
    %   comp.param_diff();
    %   comp.compare_LLE();
    %   comp.compare_lyapunov();
    %   comp.compare_traces([1 2 3]);
    %
    % See also: SRNNModel2

    properties
        models = {}             % Cell array of SRNNModel2 objects
        labels = {}             % Cell array of string labels
    end

    properties (Dependent)
        n_models                % Number of models in the comparison
    end

    methods

        function obj = SRNNComparison(varargin)
            % SRNNCOMPARISON Constructor.
            %
            % Usage:
            %   comp = SRNNComparison()
            %   comp = SRNNComparison(m1, 'label1', m2, 'label2', ...)

            for i = 1:2:length(varargin)
                obj.add(varargin{i}, varargin{i+1});
            end
        end

        function val = get.n_models(obj)
            val = numel(obj.models);
        end

        function add(obj, model, label)
            % ADD  Add an SRNNModel2 to the comparison.
            %
            %   comp.add(model, 'label')

            if ~isa(model, 'SRNNModel2')
                error('SRNNComparison:InvalidModel', 'Model must be an SRNNModel2 instance.');
            end
            if nargin < 3 || isempty(label)
                label = sprintf('model_%d', obj.n_models + 1);
            end
            obj.models{end+1} = model;
            obj.labels{end+1} = char(label);
        end

        function build_all(obj)
            % BUILD_ALL  Build all models that have not been built yet.
            for i = 1:obj.n_models
                if ~obj.models{i}.is_built
                    obj.models{i}.build();
                end
            end
        end

        function run_all(obj)
            % RUN_ALL  Run all models that have not been run yet.
            %   Builds first if needed. Records runtime for each.
            obj.build_all();
            for i = 1:obj.n_models
                if ~obj.models{i}.has_run
                    obj.models{i}.run();
                end
            end
        end

        %% ================================================================
        %  Parameter comparison
        %  ================================================================

        function T = param_diff(obj)
            % PARAM_DIFF  Display a table of parameters that differ across models.
            %
            %   T = comp.param_diff()   % returns MATLAB table
            %
            % Iterates over all public, non-dependent properties and shows
            % only those that differ between at least two models.

            if obj.n_models < 2
                fprintf('Need at least 2 models to compare parameters.\n');
                T = table();
                return;
            end

            ref = obj.models{1};
            mc = metaclass(ref);

            % Properties to skip (runtime artifacts, not configuration)
            skip = {'S0', 'cached_params', 'u_interpolant', 'ode_opts', ...
                    't_out', 'state_out', 'plot_data', 'lya_results', ...
                    'activation_function', 'activation_function_derivative', ...
                    'input_config', 'W', 'W_in', 't_ex', 'u_ex', ...
                    'is_built', 'has_run', 'plot_deci', 'T_plot'};

            diff_names = {};
            diff_values = {};

            for p = 1:numel(mc.PropertyList)
                prop = mc.PropertyList(p);

                % Skip dependent, non-public, or internal properties
                if prop.Dependent, continue; end
                if ~strcmp(prop.GetAccess, 'public'), continue; end
                if ismember(prop.Name, skip), continue; end

                % Collect values across models
                vals = cell(1, obj.n_models);
                for i = 1:obj.n_models
                    vals{i} = obj.models{i}.(prop.Name);
                end

                % Check if any differ from the first
                any_diff = false;
                for i = 2:obj.n_models
                    if isa(vals{1}, 'function_handle') && isa(vals{i}, 'function_handle')
                        if ~strcmp(func2str(vals{1}), func2str(vals{i}))
                            any_diff = true; break;
                        end
                    elseif ~isequaln(vals{1}, vals{i})
                        any_diff = true; break;
                    end
                end

                if any_diff
                    diff_names{end+1} = prop.Name; %#ok<AGROW>
                    diff_values{end+1} = vals; %#ok<AGROW>
                end
            end

            % Also check connectivity strategy properties (e.g. level_of_chaos)
            conn_props = {};
            for i = 1:obj.n_models
                if ~isempty(obj.models{i}.connectivity)
                    mc_conn = metaclass(obj.models{i}.connectivity);
                    for p = 1:numel(mc_conn.PropertyList)
                        cp = mc_conn.PropertyList(p);
                        if ~cp.Dependent && strcmp(cp.GetAccess, 'public')
                            conn_props = union(conn_props, {cp.Name});
                        end
                    end
                end
            end
            for cp = 1:numel(conn_props)
                pname = conn_props{cp};
                vals = cell(1, obj.n_models);
                for i = 1:obj.n_models
                    if ~isempty(obj.models{i}.connectivity) && isprop(obj.models{i}.connectivity, pname)
                        vals{i} = obj.models{i}.connectivity.(pname);
                    else
                        vals{i} = [];
                    end
                end
                any_diff = false;
                for i = 2:obj.n_models
                    if ~isequaln(vals{1}, vals{i})
                        any_diff = true; break;
                    end
                end
                if any_diff
                    diff_names{end+1} = ['connectivity.' pname]; %#ok<AGROW>
                    diff_values{end+1} = vals; %#ok<AGROW>
                end
            end

            if isempty(diff_names)
                fprintf('All public parameters are identical across %d models.\n', obj.n_models);
                T = table();
                return;
            end

            % Build table
            col_data = cell(numel(diff_names), obj.n_models);
            for d = 1:numel(diff_names)
                for i = 1:obj.n_models
                    col_data{d, i} = SRNNComparison.val_to_str(diff_values{d}{i});
                end
            end

            T = cell2table(col_data, 'RowNames', diff_names, 'VariableNames', obj.labels);

            % Print
            fprintf('\n--- Parameter differences across %d models ---\n', obj.n_models);
            disp(T);
        end

        %% ================================================================
        %  Summary table
        %  ================================================================

        function summary(obj)
            % SUMMARY  Print a summary table of all models.

            fprintf('\n%-4s  %-20s  %8s  %10s  %6s  %6s  %6s  %6s  %12s\n', ...
                '#', 'Label', 'LLE', 'Solver', 'n', 'n_a_E', 'n_b_E', 'fs', 'T_range');
            fprintf('%s\n', repmat('-', 1, 100));

            for i = 1:obj.n_models
                m = obj.models{i};
                lle_str = '—';
                if m.has_run && ~isempty(m.lya_results) && isfield(m.lya_results, 'LLE')
                    lle_str = sprintf('%.4f', m.lya_results.LLE);
                end

                solver_str = m.solver_mode;
                if strcmp(m.solver_mode, 'fused')
                    solver_str = sprintf('fused(%d)', m.fused_substeps);
                end

                fprintf('%-4d  %-20s  %8s  %10s  %6d  %6d  %6d  %6d  [%g, %g]\n', ...
                    i, obj.labels{i}, lle_str, solver_str, ...
                    m.n, m.n_a_E, m.n_b_E, m.fs, m.T_range(1), m.T_range(2));
            end
            fprintf('\n');
        end

        %% ================================================================
        %  LLE comparison
        %  ================================================================

        function [fig, ax] = compare_LLE(obj)
            % COMPARE_LLE  Bar chart of largest Lyapunov exponents.

            obj.require_run();

            LLEs = nan(obj.n_models, 1);
            for i = 1:obj.n_models
                if ~isempty(obj.models{i}.lya_results) && isfield(obj.models{i}.lya_results, 'LLE')
                    LLEs(i) = obj.models{i}.lya_results.LLE;
                end
            end

            fig = figure('Name', 'LLE Comparison');
            ax = axes(fig);
            bar(ax, LLEs);
            set(ax, 'XTickLabel', obj.labels, 'XTickLabelRotation', 30);
            ylabel(ax, 'LLE (\lambda_1)');
            title(ax, 'Largest Lyapunov Exponent');
            yline(ax, 0, ':k');

            % Print values
            fprintf('\n--- LLE Comparison ---\n');
            for i = 1:obj.n_models
                fprintf('  %-20s  LLE = %8.4f\n', obj.labels{i}, LLEs(i));
            end
        end

        %% ================================================================
        %  Local Lyapunov overlay
        %  ================================================================

        function [fig, ax] = compare_lyapunov(obj)
            % COMPARE_LYAPUNOV  Overlay local Lyapunov exponent time series.

            obj.require_run();

            fig = figure('Name', 'Lyapunov Comparison', 'Position', [100, 100, 900, 400]);
            ax = axes(fig);
            hold(ax, 'on');

            colors = lines(obj.n_models);
            styles = {'-', '--', ':', '-.', '-', '--', ':', '-.'};

            for i = 1:obj.n_models
                m = obj.models{i};
                if isempty(m.lya_results) || ~isfield(m.lya_results, 'local_lya')
                    continue;
                end
                style_idx = mod(i - 1, numel(styles)) + 1;
                plot(ax, m.lya_results.t_lya, m.lya_results.local_lya, ...
                    styles{style_idx}, 'Color', colors(i,:), 'LineWidth', 1.2, ...
                    'DisplayName', sprintf('%s (LLE=%.3f)', obj.labels{i}, m.lya_results.LLE));
            end

            yline(ax, 0, ':k', 'HandleVisibility', 'off');
            hold(ax, 'off');

            ylabel(ax, '\lambda_1 (local)');
            xlabel(ax, 'Time (s)');
            title(ax, 'Local Lyapunov Exponent Comparison');
            legend(ax, 'Location', 'best');
        end

        %% ================================================================
        %  Trajectory overlay
        %  ================================================================

        function [fig, ax] = compare_traces(obj, neuron_indices, state_type)
            % COMPARE_TRACES  Overlay neuron traces across models.
            %
            %   comp.compare_traces([1 2 5])                % dendritic state x
            %   comp.compare_traces([1 2], 'r')             % firing rate
            %   comp.compare_traces([1 2], 'br')            % synaptic output
            %
            % neuron_indices are global (1..n), mapped to E or I automatically.

            if nargin < 3 || isempty(state_type)
                state_type = 'x';
            end

            obj.require_run();
            obj.require_plot_data();

            if nargin < 2 || isempty(neuron_indices)
                neuron_indices = 1:min(5, obj.models{1}.n);
            end

            n_traces = numel(neuron_indices);
            fig = figure('Name', sprintf('Trace Comparison (%s)', state_type), ...
                'Position', [100, 50, 1000, 150 * n_traces]);
            tl = tiledlayout(n_traces, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

            colors = lines(obj.n_models);
            styles = {'-', '--', ':', '-.', '-', '--', ':', '-.'};

            for j = 1:n_traces
                ax(j) = nexttile(tl); %#ok<AGROW>
                hold(ax(j), 'on');

                idx = neuron_indices(j);
                n_E = obj.models{1}.n_E;

                for i = 1:obj.n_models
                    pd = obj.models{i}.plot_data;
                    t = pd.t;

                    % Extract the requested state for this neuron
                    trace = SRNNComparison.extract_trace(pd, idx, n_E, state_type);

                    style_idx = mod(i - 1, numel(styles)) + 1;
                    plot(ax(j), t, trace, styles{style_idx}, ...
                        'Color', colors(i,:), 'LineWidth', 1);
                end

                hold(ax(j), 'off');

                if idx <= n_E
                    pop_label = sprintf('E%d', idx);
                else
                    pop_label = sprintf('I%d', idx - n_E);
                end
                ylabel(ax(j), sprintf('%s_{%s}', state_type, pop_label));

                if j == 1
                    title(ax(j), sprintf('Trace comparison — %s', state_type));
                    legend(ax(j), obj.labels, 'Location', 'northeast');
                end
            end
            xlabel(ax(end), 'Time (s)');
        end

        %% ================================================================
        %  Individual time series plots
        %  ================================================================

        function figs = plot_all(obj)
            % PLOT_ALL  Call plot() on each model, labeling figures.
            %   Returns array of figure handles.

            obj.require_run();
            figs = gobjects(obj.n_models, 1);
            for i = 1:obj.n_models
                obj.models{i}.plot();
                figs(i) = gcf;
                set(figs(i), 'Name', obj.labels{i});
                sgtitle(obj.labels{i});
            end
        end

        function save_figures(~, fig_handles, names, out_dir)
            % SAVE_FIGURES  Save figures as .fig and .png.
            %
            %   comp.save_figures(figs, names, out_dir)
            %
            %   figs     — array of figure handles
            %   names    — cell array of filenames (without extension)
            %   out_dir  — output directory (created if needed)

            if ~exist(out_dir, 'dir')
                mkdir(out_dir);
                fprintf('Created directory: %s\n', out_dir);
            end

            for i = 1:numel(fig_handles)
                fig = fig_handles(i);
                fname = names{i};

                % Sanitize filename
                fname = regexprep(fname, '[^a-zA-Z0-9_\-]', '_');

                savefig(fig, fullfile(out_dir, [fname '.fig']));
                exportgraphics(fig, fullfile(out_dir, [fname '.png']), 'Resolution', 300);
                fprintf('  Saved: %s.fig / .png\n', fname);
            end
        end

    end

    %% ====================================================================
    %  Private helpers
    %  ====================================================================

    methods (Access = private)

        function require_run(obj)
            for i = 1:obj.n_models
                if ~obj.models{i}.has_run
                    error('SRNNComparison:NotRun', ...
                        'Model ''%s'' has not been run. Call run_all() first.', obj.labels{i});
                end
            end
        end

        function require_plot_data(obj)
            for i = 1:obj.n_models
                if isempty(obj.models{i}.plot_data)
                    error('SRNNComparison:NoPlotData', ...
                        'Model ''%s'' has no plot_data. Set store_decimated_state=true.', obj.labels{i});
                end
            end
        end

    end

    %% ====================================================================
    %  Static helpers
    %  ====================================================================

    methods (Static, Access = private)

        function s = val_to_str(v)
            % Convert a value to a compact string for the param_diff table.
            if isempty(v)
                s = '[]';
            elseif ischar(v) || isstring(v)
                s = char(v);
            elseif islogical(v)
                if v, s = 'true'; else, s = 'false'; end
            elseif isnumeric(v)
                if isscalar(v)
                    if v == round(v), s = sprintf('%d', v);
                    else, s = sprintf('%.4g', v);
                    end
                elseif numel(v) <= 6
                    s = mat2str(v, 4);
                else
                    s = sprintf('[%dx%d %s]', size(v,1), size(v,2), class(v));
                end
            elseif isa(v, 'function_handle')
                s = ['@' func2str(v)];
            elseif isobject(v)
                s = class(v);
            else
                s = class(v);
            end
        end

        function trace = extract_trace(pd, idx, n_E, state_type)
            % Extract a single neuron trace from plot_data.
            if idx <= n_E
                pop = 'E';
                local_idx = idx;
            else
                pop = 'I';
                local_idx = idx - n_E;
            end

            data = pd.(state_type);
            trace = data.(pop)(local_idx, :);
        end

    end

end
