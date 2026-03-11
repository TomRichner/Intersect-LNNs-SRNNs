classdef ParamSpace < handle
    % PARAMSPACE Model-agnostic parameter space analysis for cRNN subclasses.
    %
    % Performs multi-dimensional grid search over model parameters,
    % simulating each configuration and analyzing the results.
    %
    % Works with any cRNN subclass (SRNNModel2, LNN, etc.) via a
    % user-supplied model factory function.
    %
    % Key features:
    %   - Multi-dimensional grid (ndgrid over all parameters)
    %   - Randomized execution order for representative early-stopping
    %   - Batched parfor with checkpoint files for resume capability
    %   - Pluggable metric extraction (default: LLE; SRNN: +rate/synaptic)
    %   - No hardcoded conditions — all config flows through model_args
    %
    % Usage:
    %   ps = ParamSpace('n_levels', 5, 'note', 'demo');
    %   ps.model_factory = @(args) SRNNModel2(args{:});
    %   ps.model_args = {'n', 300, 'level_of_chaos', 1.0, 'n_a_E', 3};
    %   ps.add_grid_parameter('f', [0.4, 0.6]);
    %   ps.add_grid_parameter('reps', 1:3);
    %   ps.run();
    %   ps.plot('metric', 'LLE');
    %
    % See also: cRNN, SRNNModel2, LNN, ESN_reservoir

    %% ====================================================================
    %              GRID CONFIGURATION
    % =====================================================================
    properties
        grid_params = {}            % Cell array of parameter names for grid
        param_ranges = struct()     % Struct: param_name -> [min, max]
        n_levels = 8                % Number of levels per grid parameter
        integer_params = {'n', 'indegree', 'n_a_E', 'n_a_I', 'n_b_E', 'n_b_I'}
        explicit_vectors = struct() % Struct: param_name -> explicit vector (when length > 2)
        vector_param_config = struct() % Struct: param_name -> config for vector params
        randomize_order = true      % Whether to randomize execution order
    end

    %% ====================================================================
    %              MODEL CONFIGURATION
    % =====================================================================
    properties
        model_factory               % Function handle: @(cell_of_args) -> cRNN subclass
        model_args = {}             % Cell array of base name-value pairs for the model
        metric_extractor            % Function handle: @(model, store_lya, lya_dt) -> struct
        verbose = true              % Print progress during execution
    end

    %% ====================================================================
    %              EXECUTION SETTINGS
    % =====================================================================
    properties
        batch_size = 25             % Number of configs per batch
        output_dir                  % Base directory for saving results
        note = ''                   % Optional note for folder naming
        folder_prefix = 'param_space'  % Prefix for output folder name
        store_local_lya = false     % Whether to store decimated local Lyapunov time series
        store_local_lya_dt = 0.1    % Time resolution for stored local_lya (seconds)
        use_parallel = true         % Whether to use parfor
    end

    %% ====================================================================
    %              RESULTS (SetAccess = private)
    % =====================================================================
    properties (SetAccess = private)
        results = struct()          % Stored results after run (.all cell array)
        has_run = false             % Flag indicating if analysis has run
        analysis_start_time         % Timestamp when analysis started
        param_vectors               % Cell array of parameter level vectors
        all_configs                 % Cell array of all config structs
        shuffled_indices            % Randomized/sequential order for execution
        num_combinations            % Total number of grid points
        vector_param_lookup = struct() % Struct: param_name -> cell of pre-generated vectors
    end

    %% ====================================================================
    %              CONSTRUCTOR
    % =====================================================================
    methods
        function obj = ParamSpace(varargin)
            % PARAMSPACE Constructor with name-value pairs.
            %
            % Usage:
            %   ps = ParamSpace()
            %   ps = ParamSpace('n_levels', 8, 'batch_size', 50)

            % Set default metric extractor
            obj.metric_extractor = @ParamSpace.default_metric_extractor;

            % Parse name-value pairs
            for i = 1:2:length(varargin)
                if isprop(obj, varargin{i})
                    obj.(varargin{i}) = varargin{i+1};
                else
                    warning('ParamSpace:UnknownProperty', ...
                        'Unknown property: %s', varargin{i});
                end
            end

            % Set default output directory
            if isempty(obj.output_dir)
                src_path = fileparts(mfilename('fullpath'));
                project_root = fileparts(fileparts(fileparts(src_path)));  % shared/src -> shared -> Matlab -> project
                obj.output_dir = fullfile(project_root, 'data', 'param_space');
            end
        end
    end

    %% ====================================================================
    %              PUBLIC METHODS
    % =====================================================================
    methods
        function add_grid_parameter(obj, param_name, param_range)
            % ADD_GRID_PARAMETER Add a parameter to the multi-dimensional grid.
            %
            % Usage:
            %   ps.add_grid_parameter('level_of_chaos', [0.5, 3.0])
            %   ps.add_grid_parameter('f', [0.4, 0.5, 0.6])  % explicit vector

            if ~ischar(param_name) && ~isstring(param_name)
                error('ParamSpace:InvalidInput', ...
                    'param_name must be a string or char array');
            end

            if ~isnumeric(param_range) || length(param_range) < 2
                error('ParamSpace:InvalidInput', ...
                    'param_range must be a numeric array with at least 2 elements');
            end

            % Add to grid_params if not already present
            if ~ismember(param_name, obj.grid_params)
                obj.grid_params{end+1} = param_name;
            end

            if length(param_range) == 2
                % Range mode: [min, max] -> will use n_levels
                if param_range(2) < param_range(1)
                    error('ParamSpace:InvalidInput', ...
                        'param_range(2) must be >= param_range(1)');
                end
                obj.param_ranges.(param_name) = param_range;
                if isfield(obj.explicit_vectors, param_name)
                    obj.explicit_vectors = rmfield(obj.explicit_vectors, param_name);
                end
                if obj.verbose
                    fprintf('Added grid parameter: %s, range: [%.3g, %.3g] (n_levels=%d)\n', ...
                        param_name, param_range(1), param_range(2), obj.n_levels);
                end
            else
                % Explicit vector mode: use values directly
                obj.explicit_vectors.(param_name) = param_range;
                if isfield(obj.param_ranges, param_name)
                    obj.param_ranges = rmfield(obj.param_ranges, param_name);
                end
                if obj.verbose
                    fprintf('Added grid parameter: %s, explicit vector with %d values\n', ...
                        param_name, length(param_range));
                end
            end
        end

        function remove_grid_parameter(obj, param_name)
            % REMOVE_GRID_PARAMETER Remove a parameter from the grid.

            idx = find(strcmp(obj.grid_params, param_name));
            if ~isempty(idx)
                obj.grid_params(idx) = [];
                if isfield(obj.param_ranges, param_name)
                    obj.param_ranges = rmfield(obj.param_ranges, param_name);
                end
                if isfield(obj.explicit_vectors, param_name)
                    obj.explicit_vectors = rmfield(obj.explicit_vectors, param_name);
                end
                if obj.verbose
                    fprintf('Removed grid parameter: %s\n', param_name);
                end
            else
                warning('ParamSpace:ParamNotFound', ...
                    'Parameter %s not found in grid', param_name);
            end
        end

        function add_vector_parameter(obj, param_name, varargin)
            % ADD_VECTOR_PARAMETER Add a vector-valued parameter to the grid.
            %
            % For parameters like tau_a_E where the parameter is a vector and
            % one end is varied across levels.
            %
            % Usage:
            %   ps.add_vector_parameter('tau_a_E', ...
            %       'vary_element', 'last', ...
            %       'fixed_value', 0.25, ...
            %       'vary_range', [5, 60], ...
            %       'n_elements', 3, ...
            %       'spacing', 'log', ...
            %       'level_spacing', 'linear')

            p = inputParser;
            addRequired(p, 'param_name', @(x) ischar(x) || isstring(x));
            addParameter(p, 'vary_element', 'last', @(x) ismember(x, {'first', 'last'}));
            addParameter(p, 'fixed_value', [], @isnumeric);
            addParameter(p, 'vary_range', [], @(x) isnumeric(x) && length(x) == 2);
            addParameter(p, 'n_elements', [], @(x) isnumeric(x) && x >= 2);
            addParameter(p, 'spacing', 'linear', @(x) ismember(x, {'linear', 'log'}));
            addParameter(p, 'level_spacing', 'linear', @(x) ismember(x, {'linear', 'log'}));
            parse(p, param_name, varargin{:});

            if isempty(p.Results.fixed_value)
                error('ParamSpace:InvalidInput', 'fixed_value is required');
            end
            if isempty(p.Results.vary_range)
                error('ParamSpace:InvalidInput', 'vary_range is required');
            end
            if isempty(p.Results.n_elements)
                error('ParamSpace:InvalidInput', 'n_elements is required');
            end
            if p.Results.vary_range(2) < p.Results.vary_range(1)
                error('ParamSpace:InvalidInput', 'vary_range(2) must be >= vary_range(1)');
            end

            vpc = struct();
            vpc.vary_element = p.Results.vary_element;
            vpc.fixed_value = p.Results.fixed_value;
            vpc.vary_range = p.Results.vary_range;
            vpc.n_elements = p.Results.n_elements;
            vpc.spacing = p.Results.spacing;
            vpc.level_spacing = p.Results.level_spacing;
            obj.vector_param_config.(param_name) = vpc;

            if ~ismember(param_name, obj.grid_params)
                obj.grid_params{end+1} = param_name;
            end

            if isfield(obj.param_ranges, param_name)
                obj.param_ranges = rmfield(obj.param_ranges, param_name);
            end
            if isfield(obj.explicit_vectors, param_name)
                obj.explicit_vectors = rmfield(obj.explicit_vectors, param_name);
            end

            if obj.verbose
                fprintf('Added vector parameter: %s, vary_%s [%.3g, %.3g], %d elements, %s spacing, %s level_spacing\n', ...
                    param_name, vpc.vary_element, vpc.vary_range(1), vpc.vary_range(2), ...
                    vpc.n_elements, vpc.spacing, vpc.level_spacing);
            end
        end

        function run(obj)
            % RUN Execute the full parameter space analysis.
            %
            % This method:
            %   1. Generates the multi-dimensional parameter grid
            %   2. Randomizes execution order
            %   3. Runs batched parfor with checkpoint files
            %   4. Consolidates results into a single MAT file

            % Validate
            if isempty(obj.grid_params)
                error('ParamSpace:NoParameters', ...
                    'No grid parameters defined. Use add_grid_parameter() first.');
            end

            if isempty(obj.model_factory)
                error('ParamSpace:NoFactory', ...
                    'No model_factory set. Assign a function handle, e.g.\n  ps.model_factory = @(args) SRNNModel2(args{:});');
            end

            % Create timestamped output directory
            obj.analysis_start_time = datetime('now');
            dt_str = lower(char(obj.analysis_start_time, 'MMM_dd_yy_HH_mm'));

            if ~isempty(obj.note)
                folder_name = sprintf('%s_%s_nLevs_%d_%s', ...
                    obj.folder_prefix, obj.note, obj.n_levels, dt_str);
            else
                folder_name = sprintf('%s_nLevs_%d_%s', ...
                    obj.folder_prefix, obj.n_levels, dt_str);
            end

            obj.output_dir = fullfile(obj.output_dir, folder_name);
            if ~exist(obj.output_dir, 'dir')
                mkdir(obj.output_dir);
            end

            % Generate parameter grid
            obj.generate_grid();

            % Print summary
            fprintf('\n========================================\n');
            fprintf('=== Parameter Space Analysis ===\n');
            fprintf('========================================\n');
            fprintf('Model factory: %s\n', func2str(obj.model_factory));
            fprintf('Grid parameters: %s\n', strjoin(obj.grid_params, ', '));
            fprintf('Levels per parameter: %d\n', obj.n_levels);
            fprintf('Total grid combinations: %d\n', obj.num_combinations);
            fprintf('Batch size: %d\n', obj.batch_size);
            fprintf('Output directory: %s\n', obj.output_dir);
            fprintf('========================================\n\n');

            % Create temp directory for batch results
            temp_dir = fullfile(obj.output_dir, 'temp_batches');
            if ~exist(temp_dir, 'dir')
                mkdir(temp_dir);
            end

            overall_start = tic;

            % Run batched simulation
            obj.run_batched_simulation(temp_dir);

            % Consolidate batch results
            obj.consolidate(temp_dir);

            overall_elapsed = toc(overall_start);
            fprintf('\n========================================\n');
            fprintf('=== Analysis Complete ===\n');
            fprintf('Total time: %.2f hours\n', overall_elapsed/3600);
            fprintf('========================================\n');

            obj.has_run = true;

            % Save summary
            obj.save_summary();
        end

        function plot(obj, varargin)
            % PLOT Generate histogram plot of a metric across parameter space.
            %
            % Usage:
            %   ps.plot()
            %   ps.plot('metric', 'LLE')

            if ~obj.has_run && isempty(fieldnames(obj.results))
                error('ParamSpace:NotRun', ...
                    'Analysis has not been run yet. Call run() first.');
            end

            % Parse arguments
            metric = 'LLE';
            for i = 1:2:length(varargin)
                if strcmpi(varargin{i}, 'metric')
                    metric = varargin{i+1};
                end
            end

            % Define histogram bins
            if strcmpi(metric, 'LLE')
                hist_range = [-1.5, 1.5];
                n_bins = 25;
                y_label = 'LLE (\lambda_1)';
            elseif strcmpi(metric, 'mean_rate')
                hist_range = [0, 1];
                n_bins = 25;
                y_label = 'Mean Firing Rate';
            else
                hist_range = [-10, 10];
                n_bins = 25;
                y_label = metric;
            end

            hist_bins = [linspace(hist_range(1), hist_range(2), n_bins + 1), inf];
            if strcmpi(metric, 'LLE')
                hist_bins = [-inf, hist_bins];
            end

            % Extract metric values
            if ~isfield(obj.results, 'all')
                warning('ParamSpace:NoResults', 'No results found.');
                return;
            end

            results_cell = obj.results.all;
            values = [];
            for k = 1:length(results_cell)
                res = results_cell{k};
                if isstruct(res) && isfield(res, 'success') && res.success
                    if isfield(res, metric) && ~isnan(res.(metric))
                        values(end+1) = res.(metric); %#ok<AGROW>
                    end
                end
            end

            % Create figure
            fig = figure('Name', sprintf('%s Distribution', metric), ...
                'Position', [100, 100, 500, 300]);

            if ~isempty(values)
                [counts, edges] = histcounts(values, hist_bins);
                prob = counts / sum(counts);

                finite_edges = edges;
                step = (hist_range(2) - hist_range(1)) / n_bins;
                if isinf(finite_edges(1))
                    finite_edges(1) = hist_range(1) - step;
                end
                if isinf(finite_edges(end))
                    finite_edges(end) = hist_range(2) + step;
                end

                histogram('BinEdges', finite_edges, 'BinCounts', prob, ...
                    'EdgeColor', 'none', 'FaceColor', [0.5 0.5 0.5]);

                hold on;
                if strcmpi(metric, 'LLE')
                    xline(0, '--', 'Color', [0 0.7 0], 'LineWidth', 2);
                end
                hold off;
            end

            ylabel('Probability', 'FontSize', 12);
            xlabel(y_label, 'FontSize', 12);
            title(sprintf('%s Distribution (n=%d)', metric, length(values)), 'FontSize', 14);
            box off;

            % Save figure
            fig_dir = fullfile(obj.output_dir, 'figures');
            if ~exist(fig_dir, 'dir')
                mkdir(fig_dir);
            end
            saveas(fig, fullfile(fig_dir, sprintf('%s_distribution.png', metric)));
            saveas(fig, fullfile(fig_dir, sprintf('%s_distribution.fig', metric)));
            fprintf('Figure saved to: %s\n', fig_dir);
        end

        function plot_sensitivity(obj, varargin)
            % PLOT_SENSITIVITY Generate imagesc heatmap for 1D sensitivity.
            %
            % Usage:
            %   ps.plot_sensitivity()
            %   ps.plot_sensitivity('metric', 'LLE')
            %   ps.plot_sensitivity('metric', 'LLE', 'hist_range', [-0.3, 0.1])

            if ~obj.has_run && isempty(fieldnames(obj.results))
                error('ParamSpace:NotRun', ...
                    'Analysis has not been run yet. Call run() first.');
            end

            if ~isfield(obj.results, 'all')
                warning('ParamSpace:NoResults', 'No results found.');
                return;
            end

            % Parse arguments
            metric = 'LLE';
            hist_range = [];
            n_bins = 35;
            for i = 1:2:length(varargin)
                switch lower(varargin{i})
                    case 'metric', metric = varargin{i+1};
                    case 'hist_range', hist_range = varargin{i+1};
                    case 'n_bins', n_bins = varargin{i+1};
                end
            end

            if isempty(hist_range)
                if strcmpi(metric, 'LLE')
                    hist_range = [-0.3, 0.1];
                elseif strcmpi(metric, 'mean_rate')
                    hist_range = [0, 1];
                else
                    hist_range = [-1, 1];
                end
            end

            % Identify swept parameters (non-reps grid params)
            swept_params = setdiff(obj.grid_params, {'reps'}, 'stable');
            if isempty(swept_params)
                error('ParamSpace:NoSweptParam', ...
                    'No non-reps grid parameter found for sensitivity plot.');
            end

            hist_bins = [-inf, linspace(hist_range(1), hist_range(2), n_bins), inf];
            grid_sizes = cellfun(@length, obj.param_vectors);
            results_cell = obj.results.all;

            for sp_idx = 1:length(swept_params)
                swept_param = swept_params{sp_idx};
                param_dim = find(strcmp(obj.grid_params, swept_param));
                reps_dim = find(strcmp(obj.grid_params, 'reps'));

                % Get x-axis values
                if isfield(obj.vector_param_config, swept_param)
                    vpc = obj.vector_param_config.(swept_param);
                    vecs = obj.vector_param_lookup.(swept_param);
                    x_values = zeros(1, length(vecs));
                    for v = 1:length(vecs)
                        if strcmp(vpc.vary_element, 'last')
                            x_values(v) = vecs{v}(end);
                        else
                            x_values(v) = vecs{v}(1);
                        end
                    end
                    x_label = sprintf('%s(%s)', strrep(swept_param, '_', '\_'), vpc.vary_element);
                else
                    x_values = obj.param_vectors{param_dim};
                    x_label = strrep(swept_param, '_', '\_');
                end

                n_levels_param = length(x_values);
                n_reps = length(obj.param_vectors{reps_dim});

                fig = figure('Name', sprintf('%s Sensitivity - %s', metric, swept_param), ...
                    'Position', [100, 100, 500, 400]);

                num_hist_bins = length(hist_bins) - 1;
                histogram_matrix = zeros(num_hist_bins, n_levels_param);
                median_values = NaN(n_levels_param, 1);

                for level_idx = 1:n_levels_param
                    values_level = [];

                    for rep_idx = 1:n_reps
                        subs = cell(1, length(obj.grid_params));
                        for d = 1:length(obj.grid_params)
                            if d == param_dim
                                subs{d} = level_idx;
                            elseif d == reps_dim
                                subs{d} = rep_idx;
                            else
                                subs{d} = 1;
                            end
                        end
                        linear_idx = sub2ind(grid_sizes, subs{:});

                        if linear_idx <= length(results_cell)
                            res = results_cell{linear_idx};
                            if isstruct(res) && isfield(res, 'success') && res.success
                                if isfield(res, metric) && ~isnan(res.(metric))
                                    values_level(end+1) = res.(metric); %#ok<AGROW>
                                end
                            end
                        end
                    end

                    if ~isempty(values_level)
                        [counts, ~] = histcounts(values_level, hist_bins);
                        histogram_matrix(:, level_idx) = counts;
                        median_values(level_idx) = median(values_level);
                    end
                end

                % Compute y-coordinates
                finite_edges = hist_bins(~isinf(hist_bins));
                step_size = finite_edges(2) - finite_edges(1);
                y_coords = zeros(num_hist_bins, 1);
                y_coords(1) = finite_edges(1) - step_size/2;
                for k = 2:length(finite_edges)
                    y_coords(k) = (finite_edges(k-1) + finite_edges(k)) / 2;
                end
                y_coords(end) = finite_edges(end) + step_size/2;

                ax = gca;
                imagesc(ax, x_values, y_coords, histogram_matrix);
                hold(ax, 'on');
                yline(ax, 0, '--', 'Color', [0 0.7 0], 'LineWidth', 4, 'Alpha', 0.5);
                plot(ax, x_values, median_values, 'b-', 'LineWidth', 4, 'Color', [0 0 1 0.55]);
                hold(ax, 'off');

                colormap(ax, flipud(gray));
                clim(ax, [0, n_reps]);
                axis(ax, 'xy');
                box(ax, 'on');

                xlabel(ax, x_label, 'FontSize', 14);
                if strcmpi(metric, 'LLE')
                    ylabel(ax, '$\lambda_1$', 'Interpreter', 'latex', 'FontSize', 18);
                else
                    ylabel(ax, strrep(metric, '_', '\_'), 'FontSize', 14);
                end
                title(ax, sprintf('%s vs %s', metric, strrep(swept_param, '_', ' ')), 'FontSize', 14);

                % Save figure
                fig_dir = fullfile(obj.output_dir, 'figures');
                if ~exist(fig_dir, 'dir')
                    mkdir(fig_dir);
                end
                saveas(fig, fullfile(fig_dir, sprintf('sensitivity_%s_%s.png', metric, swept_param)));
                saveas(fig, fullfile(fig_dir, sprintf('sensitivity_%s_%s.fig', metric, swept_param)));
                fprintf('Figure saved to: %s\n', fig_dir);
            end
        end

        function load_results(obj, results_dir)
            % LOAD_RESULTS Load results from a previous run.
            %
            % Usage:
            %   ps.load_results('/path/to/param_space_...')

            obj.output_dir = results_dir;

            % Load summary
            summary_file = fullfile(results_dir, 'param_space_summary.mat');
            if exist(summary_file, 'file')
                loaded = load(summary_file);
                if isfield(loaded, 'summary_data')
                    obj.grid_params = loaded.summary_data.grid_params;
                    obj.param_ranges = loaded.summary_data.param_ranges;
                    obj.n_levels = loaded.summary_data.n_levels;
                    if isfield(loaded.summary_data, 'model_args')
                        obj.model_args = loaded.summary_data.model_args;
                    end
                    if isfield(loaded.summary_data, 'param_vectors')
                        obj.param_vectors = loaded.summary_data.param_vectors;
                    end
                    if isfield(loaded.summary_data, 'num_combinations')
                        obj.num_combinations = loaded.summary_data.num_combinations;
                    end
                end
            end

            % Load results
            results_file = fullfile(results_dir, 'results', 'param_space_results.mat');
            if exist(results_file, 'file')
                loaded = load(results_file);
                if isfield(loaded, 'results_to_save')
                    obj.results.all = loaded.results_to_save;
                    fprintf('Loaded %d results\n', length(loaded.results_to_save));
                elseif isfield(loaded, 'results')
                    obj.results.all = loaded.results;
                    fprintf('Loaded %d results\n', length(loaded.results));
                end
            end

            obj.has_run = true;
        end

        function consolidate(obj, temp_dir)
            % CONSOLIDATE Merge batch results into a single MAT file.
            %
            % Usage:
            %   ps.consolidate()           % Standalone recovery
            %   ps.consolidate(temp_dir)   % Internal call from run()

            standalone_call = (nargin < 2 || isempty(temp_dir));

            if standalone_call
                if isempty(obj.output_dir)
                    error('ParamSpace:NoOutputDir', ...
                        'output_dir is not set.');
                end

                temp_dir = fullfile(obj.output_dir, 'temp_batches');

                if ~exist(temp_dir, 'dir')
                    error('ParamSpace:NoTempDir', ...
                        'No temp_batches directory found in %s.', obj.output_dir);
                end

                % Load summary if it exists
                summary_file = fullfile(obj.output_dir, 'param_space_summary.mat');
                if exist(summary_file, 'file')
                    loaded = load(summary_file);
                    if isfield(loaded, 'summary_data')
                        obj.grid_params = loaded.summary_data.grid_params;
                        obj.param_ranges = loaded.summary_data.param_ranges;
                        obj.n_levels = loaded.summary_data.n_levels;
                        if isfield(loaded.summary_data, 'num_combinations')
                            obj.num_combinations = loaded.summary_data.num_combinations;
                        end
                        if isfield(loaded.summary_data, 'param_vectors')
                            obj.param_vectors = loaded.summary_data.param_vectors;
                        end
                        if isfield(loaded.summary_data, 'model_args')
                            obj.model_args = loaded.summary_data.model_args;
                        end
                    end
                else
                    % Infer from batch files
                    batch_files = dir(fullfile(temp_dir, 'batch_*.mat'));
                    if isempty(batch_files)
                        error('ParamSpace:NoBatchFiles', ...
                            'No batch files found in %s', temp_dir);
                    end

                    all_indices = [];
                    for i = 1:length(batch_files)
                        b = load(fullfile(temp_dir, batch_files(i).name), 'batch_indices');
                        all_indices = [all_indices, b.batch_indices]; %#ok<AGROW>
                    end
                    obj.num_combinations = max(all_indices);

                    warning('ParamSpace:NoSummary', ...
                        'No summary file found. Inferred %d combinations.', obj.num_combinations);
                end

                fprintf('Consolidating results from %s...\n', temp_dir);
            end

            %% Core consolidation logic
            fprintf('\nConsolidating batch results...\n');

            num_batches = ceil(obj.num_combinations / obj.batch_size);

            % Initialize results storage
            obj.results.all = cell(obj.num_combinations, 1);

            % Load and merge batches
            all_found = true;
            for batch_idx = 1:num_batches
                batch_file = fullfile(temp_dir, sprintf('batch_%d.mat', batch_idx));

                if exist(batch_file, 'file')
                    loaded = load(batch_file);
                    batch_results = loaded.batch_results;

                    for k = 1:length(batch_results)
                        res = batch_results{k};
                        if isstruct(res) && isfield(res, 'config_idx')
                            obj.results.all{res.config_idx} = res;
                        end
                    end
                else
                    fprintf('Warning: Batch file %d not found\n', batch_idx);
                    all_found = false;
                end
            end

            % Save results
            results_out_dir = fullfile(obj.output_dir, 'results');
            if ~exist(results_out_dir, 'dir')
                mkdir(results_out_dir);
            end

            results_to_save = obj.results.all;
            save_file = fullfile(results_out_dir, 'param_space_results.mat');
            save(save_file, 'results_to_save', '-v7.3');

            n_success = sum(cellfun(@(r) isstruct(r) && isfield(r, 'success') && r.success, obj.results.all));
            fprintf('%d/%d successful, saved to %s\n', ...
                n_success, obj.num_combinations, save_file);

            % Clean up temp directory
            if all_found
                rmdir(temp_dir, 's');
                fprintf('Temp directory cleaned up.\n');
            else
                fprintf('Temp directory retained due to missing batches.\n');
            end

            %% Finalize for standalone calls
            if standalone_call
                obj.save_summary();
                obj.has_run = true;
                fprintf('Consolidation complete.\n');
            end
        end

        function s = saveobj(obj)
            % SAVEOBJ Convert object to struct for saving.

            s = struct();

            % Configuration
            s.grid_params = obj.grid_params;
            s.param_ranges = obj.param_ranges;
            s.n_levels = obj.n_levels;
            s.integer_params = obj.integer_params;
            s.explicit_vectors = obj.explicit_vectors;
            s.vector_param_config = obj.vector_param_config;
            s.randomize_order = obj.randomize_order;

            % Model
            s.model_args = obj.model_args;
            s.verbose = obj.verbose;

            % Store function handle strings for reference (handles may not
            % serialize across MATLAB sessions with path differences)
            if ~isempty(obj.model_factory)
                s.model_factory_str = func2str(obj.model_factory);
            end
            if ~isempty(obj.metric_extractor)
                s.metric_extractor_str = func2str(obj.metric_extractor);
            end

            % Execution
            s.batch_size = obj.batch_size;
            s.output_dir = obj.output_dir;
            s.note = obj.note;
            s.folder_prefix = obj.folder_prefix;
            s.store_local_lya = obj.store_local_lya;
            s.store_local_lya_dt = obj.store_local_lya_dt;
            s.use_parallel = obj.use_parallel;

            % Results (private)
            s.results = obj.results;
            s.has_run = obj.has_run;
            s.analysis_start_time = obj.analysis_start_time;
            s.param_vectors = obj.param_vectors;
            s.all_configs = obj.all_configs;
            s.shuffled_indices = obj.shuffled_indices;
            s.num_combinations = obj.num_combinations;
            s.vector_param_lookup = obj.vector_param_lookup;
        end
    end

    %% ====================================================================
    %              STATIC METHODS
    % =====================================================================
    methods (Static)
        function obj = loadobj(s)
            % LOADOBJ Reconstruct object from struct when loading.

            if isstruct(s)
                obj = ParamSpace();

                if isfield(s, 'grid_params'), obj.grid_params = s.grid_params; end
                if isfield(s, 'param_ranges'), obj.param_ranges = s.param_ranges; end
                if isfield(s, 'n_levels'), obj.n_levels = s.n_levels; end
                if isfield(s, 'integer_params'), obj.integer_params = s.integer_params; end
                if isfield(s, 'explicit_vectors'), obj.explicit_vectors = s.explicit_vectors; end
                if isfield(s, 'vector_param_config'), obj.vector_param_config = s.vector_param_config; end
                if isfield(s, 'randomize_order'), obj.randomize_order = s.randomize_order; end

                if isfield(s, 'model_args'), obj.model_args = s.model_args; end
                if isfield(s, 'verbose'), obj.verbose = s.verbose; end

                if isfield(s, 'batch_size'), obj.batch_size = s.batch_size; end
                if isfield(s, 'output_dir'), obj.output_dir = s.output_dir; end
                if isfield(s, 'note'), obj.note = s.note; end
                if isfield(s, 'folder_prefix'), obj.folder_prefix = s.folder_prefix; end
                if isfield(s, 'store_local_lya'), obj.store_local_lya = s.store_local_lya; end
                if isfield(s, 'store_local_lya_dt'), obj.store_local_lya_dt = s.store_local_lya_dt; end
                if isfield(s, 'use_parallel'), obj.use_parallel = s.use_parallel; end

                % Warn about function handles
                if isfield(s, 'model_factory_str')
                    warning('ParamSpace:LoadFactory', ...
                        'model_factory was saved as string: ''%s''. You must re-assign the function handle.', ...
                        s.model_factory_str);
                end

                if isfield(s, 'results'), obj.results = s.results; end
                if isfield(s, 'has_run'), obj.has_run = s.has_run; end
                if isfield(s, 'analysis_start_time'), obj.analysis_start_time = s.analysis_start_time; end
                if isfield(s, 'param_vectors'), obj.param_vectors = s.param_vectors; end
                if isfield(s, 'all_configs'), obj.all_configs = s.all_configs; end
                if isfield(s, 'shuffled_indices'), obj.shuffled_indices = s.shuffled_indices; end
                if isfield(s, 'num_combinations'), obj.num_combinations = s.num_combinations; end
                if isfield(s, 'vector_param_lookup'), obj.vector_param_lookup = s.vector_param_lookup; end
            else
                obj = s;
            end
        end

        function result = run_single_job(job, model_factory_local, model_args_local, ...
                grid_params_local, verbose_local, store_local_lya_local, ...
                store_local_lya_dt_local, vector_param_lookup_local, metric_extractor_local)
            % RUN_SINGLE_JOB Execute a single simulation job.
            %
            % Static for parfor compatibility. Creates a model via the
            % factory, builds, runs, and extracts metrics.
            %
            % Argument priority (last wins — MATLAB name-value semantics):
            %   1. model_args (base)
            %   2. Grid parameter overrides
            %   3. rng_seeds from job.network_seed

            run_start = tic;

            try
                % Build model arguments: start with base model_args
                model_args_cell = model_args_local;

                % Add grid parameters (override base args)
                for p_idx = 1:length(grid_params_local)
                    pname = grid_params_local{p_idx};
                    if isfield(vector_param_lookup_local, pname)
                        vec_idx = job.config.(pname);
                        model_args_cell = [model_args_cell, {pname, vector_param_lookup_local.(pname){vec_idx}}]; %#ok<AGROW>
                    else
                        model_args_cell = [model_args_cell, {pname, job.config.(pname)}]; %#ok<AGROW>
                    end
                end

                % Add RNG seeds (last, highest priority)
                model_args_cell = [model_args_cell, {'rng_seeds', [job.network_seed, job.network_seed + 1]}];

                % Create and run model via factory
                model = model_factory_local(model_args_cell);
                model.build();
                model.run();

                % Extract metrics via pluggable extractor
                result = metric_extractor_local(model, store_local_lya_local, store_local_lya_dt_local);

                % Add job metadata
                result.success = true;
                result.config = job.config;
                result.config_idx = job.config_idx;
                result.network_seed = job.network_seed;
                result.run_duration = toc(run_start);

            catch ME
                result = struct();
                result.success = false;
                result.error_message = ME.message;
                result.config = job.config;
                result.config_idx = job.config_idx;
                result.network_seed = job.network_seed;
                result.run_duration = toc(run_start);
                result.LLE = NaN;

                if verbose_local
                    fprintf('  ERROR config %d: %s\n', job.config_idx, ME.message);
                end
            end
        end

        function result = default_metric_extractor(model, store_local_lya, store_local_lya_dt)
            % DEFAULT_METRIC_EXTRACTOR Extract LLE and (optionally) local Lyapunov.
            %
            % Works with any cRNN subclass.

            result = struct();

            % Extract LLE
            if ~isempty(model.lya_results) && isfield(model.lya_results, 'LLE')
                result.LLE = model.lya_results.LLE;
            else
                result.LLE = NaN;
            end

            % Extract decimated local Lyapunov time series if requested
            if store_local_lya && ~isempty(model.lya_results)
                if isfield(model.lya_results, 'local_lya') && isfield(model.lya_results, 't_lya')
                    current_lya_dt = model.lya_results.lya_dt;
                    deci_factor = max(1, round(store_local_lya_dt / current_lya_dt));
                    result.local_lya = model.lya_results.local_lya(1:deci_factor:end);
                    result.t_lya = model.lya_results.t_lya(1:deci_factor:end);
                    result.local_lya_dt = current_lya_dt * deci_factor;
                end
            end
        end

        function result = srnn_metric_extractor(model, store_local_lya, store_local_lya_dt)
            % SRNN_METRIC_EXTRACTOR Extract LLE + SRNN-specific metrics.
            %
            % Adds mean_rate and mean_synaptic_output from SRNN plot_data.

            % Start with default extraction
            result = ParamSpace.default_metric_extractor(model, store_local_lya, store_local_lya_dt);

            % Extract SRNN-specific metrics
            if ~isempty(model.plot_data) && isfield(model.plot_data, 'r')
                r_E = model.plot_data.r.E;
                r_I = model.plot_data.r.I;
                all_rates = [r_E(:); r_I(:)];
                result.mean_rate = mean(all_rates(~isnan(all_rates)));

                if isfield(model.plot_data, 'br')
                    br_E = model.plot_data.br.E;
                    br_I = model.plot_data.br.I;
                    all_br = [br_E(:); br_I(:)];
                    result.mean_synaptic_output = mean(all_br(~isnan(all_br)));
                else
                    result.mean_synaptic_output = NaN;
                end
            else
                result.mean_rate = NaN;
                result.mean_synaptic_output = NaN;
            end
        end
    end

    %% ====================================================================
    %              PRIVATE METHODS
    % =====================================================================
    methods (Access = private)
        function generate_grid(obj)
            % GENERATE_GRID Create the multi-dimensional parameter grid.

            n_params = length(obj.grid_params);
            obj.param_vectors = cell(1, n_params);

            for i = 1:n_params
                param_name = obj.grid_params{i};
                is_int = ismember(param_name, obj.integer_params);

                if isfield(obj.vector_param_config, param_name)
                    obj.param_vectors{i} = 1:obj.n_levels;

                    vpc = obj.vector_param_config.(param_name);

                    if strcmp(vpc.level_spacing, 'log')
                        vary_values = logspace(log10(vpc.vary_range(1)), log10(vpc.vary_range(2)), obj.n_levels);
                    else
                        vary_values = linspace(vpc.vary_range(1), vpc.vary_range(2), obj.n_levels);
                    end
                    if is_int
                        vary_values = round(vary_values);
                    end

                    vectors = cell(obj.n_levels, 1);
                    for lev = 1:obj.n_levels
                        if strcmp(vpc.vary_element, 'last')
                            start_val = vpc.fixed_value;
                            end_val = vary_values(lev);
                        else
                            start_val = vary_values(lev);
                            end_val = vpc.fixed_value;
                        end

                        if strcmp(vpc.spacing, 'log')
                            vectors{lev} = logspace(log10(start_val), log10(end_val), vpc.n_elements);
                        else
                            vectors{lev} = linspace(start_val, end_val, vpc.n_elements);
                        end
                        if is_int
                            vectors{lev} = round(vectors{lev});
                        end
                    end
                    obj.vector_param_lookup.(param_name) = vectors;

                elseif isfield(obj.explicit_vectors, param_name)
                    obj.param_vectors{i} = obj.explicit_vectors.(param_name);
                else
                    param_range = obj.param_ranges.(param_name);
                    if is_int
                        obj.param_vectors{i} = round(linspace(param_range(1), param_range(2), obj.n_levels));
                    else
                        obj.param_vectors{i} = linspace(param_range(1), param_range(2), obj.n_levels);
                    end
                end
            end

            % Generate all combinations using ndgrid
            grid_cells = cell(size(obj.param_vectors));
            [grid_cells{:}] = ndgrid(obj.param_vectors{:});

            obj.num_combinations = numel(grid_cells{1});
            obj.all_configs = cell(obj.num_combinations, 1);

            for i = 1:obj.num_combinations
                config = struct();
                for j = 1:n_params
                    config.(obj.grid_params{j}) = grid_cells{j}(i);
                end
                obj.all_configs{i} = config;
            end

            % Set execution order
            if obj.randomize_order
                rng('shuffle');
                obj.shuffled_indices = randperm(obj.num_combinations);
                fprintf('Generated %d parameter combinations (randomized order)\n', obj.num_combinations);
            else
                obj.shuffled_indices = 1:obj.num_combinations;
                fprintf('Generated %d parameter combinations (sequential order)\n', obj.num_combinations);
            end
        end

        function run_batched_simulation(obj, temp_dir)
            % RUN_BATCHED_SIMULATION Execute simulations in batches with checkpoints.

            num_batches = ceil(obj.num_combinations / obj.batch_size);

            fprintf('Running %d combinations in %d batches...\n', obj.num_combinations, num_batches);

            for batch_idx = 1:num_batches
                batch_file = fullfile(temp_dir, sprintf('batch_%d.mat', batch_idx));

                % Skip if batch already completed (resume capability)
                if exist(batch_file, 'file')
                    fprintf('Batch %d/%d already completed. Skipping.\n', batch_idx, num_batches);
                    continue;
                end

                start_idx = (batch_idx - 1) * obj.batch_size + 1;
                end_idx = min(batch_idx * obj.batch_size, obj.num_combinations);
                batch_indices = obj.shuffled_indices(start_idx:end_idx);
                current_batch_size = length(batch_indices);

                fprintf('\n--- Batch %d/%d (configs %d-%d) ---\n', ...
                    batch_idx, num_batches, start_idx, end_idx);

                % Create jobs (one per grid point — no conditions)
                jobs = cell(current_batch_size, 1);
                for k = 1:current_batch_size
                    config_idx = batch_indices(k);
                    config = obj.all_configs{config_idx};

                    job = struct();
                    job.config = config;
                    job.config_idx = config_idx;
                    job.network_seed = config_idx * 100;
                    jobs{k} = job;
                end

                % Extract values for parfor/for loop
                model_factory_local = obj.model_factory;
                model_args_local = obj.model_args;
                grid_params_local = obj.grid_params;
                verbose_local = obj.verbose;
                store_local_lya_local = obj.store_local_lya;
                store_local_lya_dt_local = obj.store_local_lya_dt;
                vector_param_lookup_local = obj.vector_param_lookup;
                metric_extractor_local = obj.metric_extractor;

                % Determine execution mode
                run_parallel = obj.use_parallel && canUseParallelPool;

                if obj.use_parallel && ~canUseParallelPool && batch_idx == 1
                    warning('ParamSpace:NoParallelPool', ...
                        'Parallel pool not available. Falling back to sequential execution.');
                end

                % Run simulation loop
                parallel_results = cell(current_batch_size, 1);
                batch_start = tic;

                if run_parallel
                    parfor j = 1:current_batch_size
                        parallel_results{j} = ParamSpace.run_single_job(...
                            jobs{j}, model_factory_local, model_args_local, ...
                            grid_params_local, verbose_local, store_local_lya_local, ...
                            store_local_lya_dt_local, vector_param_lookup_local, ...
                            metric_extractor_local);
                    end
                else
                    for j = 1:current_batch_size
                        parallel_results{j} = ParamSpace.run_single_job(...
                            jobs{j}, model_factory_local, model_args_local, ...
                            grid_params_local, verbose_local, store_local_lya_local, ...
                            store_local_lya_dt_local, vector_param_lookup_local, ...
                            metric_extractor_local);
                    end
                end

                batch_elapsed = toc(batch_start);

                % Save batch checkpoint
                batch_results = parallel_results;
                save(batch_file, 'batch_results', 'batch_indices', '-v7.3');

                n_success = sum(cellfun(@(r) r.success, parallel_results));
                fprintf('Batch %d completed in %.1f min (%d/%d successful)\n', ...
                    batch_idx, batch_elapsed/60, n_success, current_batch_size);
            end
        end

        function save_summary(obj)
            % SAVE_SUMMARY Save analysis summary to disk.

            summary_file = fullfile(obj.output_dir, 'param_space_summary.mat');

            summary_data = struct();
            summary_data.grid_params = obj.grid_params;
            summary_data.param_ranges = obj.param_ranges;
            summary_data.param_vectors = obj.param_vectors;
            summary_data.n_levels = obj.n_levels;
            summary_data.num_combinations = obj.num_combinations;
            summary_data.model_args = obj.model_args;
            summary_data.shuffled_indices = obj.shuffled_indices;
            summary_data.analysis_start_time = obj.analysis_start_time;
            summary_data.analysis_completed = char(datetime('now'));

            % Compute statistics
            if isfield(obj.results, 'all')
                results_cell = obj.results.all;
                n_success = sum(cellfun(@(r) isstruct(r) && isfield(r, 'success') && r.success, results_cell));
                summary_data.stats.n_success = n_success;
                summary_data.stats.n_total = length(results_cell);
                summary_data.stats.success_rate = n_success / length(results_cell);
            end

            save(summary_file, 'summary_data', '-v7.3');
            fprintf('Summary saved to: %s\n', summary_file);
        end
    end
end
