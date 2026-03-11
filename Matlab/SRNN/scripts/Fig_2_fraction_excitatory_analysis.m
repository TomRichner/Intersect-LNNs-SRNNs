% Fig_2_fraction_excitatory_analysis.m
% Parameter space analysis: fraction excitatory (f) sweep with adaptation
% variants included as grid parameters.
%
% Sweeps f=[0.4, 0.6] × n_a_E={0,3} × n_b_E={0,1} with reps.
% Since conditions are now grid parameters, the full grid covers all four
% adaptation variants (no_adapt, sfa_only, std_only, sfa+std).
%
% Ported from FractionalReservoir to use ParamSpace / refactored SRNNModel2.
%
% See also: ParamSpace, SRNNModel2, cRNN

%% Configuration
if exist('master_save_figs', 'var')
    if strcmp(master_save_figs, 'save_all_figs')
        save_figs = true;
    elseif strcmp(master_save_figs, 'save_no_figs')
        save_figs = false;
    end
end
if ~exist('save_figs', 'var')
    save_figs = false;
end

%% Setup paths
setup_paths();

script_path = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_path));  % SRNN/scripts -> SRNN -> Matlab project
figs_root = fullfile(project_root, 'figs');

%% Create ParamSpace object
ps = ParamSpace(...
    'n_levels', 5, ...
    'batch_size', 25, ...
    'note', 'frac_exc', ...
    'verbose', true ...
    );
ps.folder_prefix = 'fig2';
if exist('master_output_dir', 'var')
    ps.output_dir = master_output_dir;
end

%% Set model factory and metric extractor
ps.model_factory = @(args) SRNNModel2(args{:});
ps.metric_extractor = @ParamSpace.srnn_metric_extractor;

%% Configure model defaults via model_args
N = 300;
indegree = 100;

ps.model_args = { ...
    'n', N, ...
    'indegree', indegree, ...
    'T_range', [-15, 45], ...
    'tau_d', 0.1, ...
    'level_of_chaos', 1.0, ...
    'lya_method', 'benettin' ...
    };

%% Lyapunov / local storage
ps.store_local_lya = true;
ps.store_local_lya_dt = 0.1;

%% Add parameters to the grid
% Fraction excitatory sweep
ps.add_grid_parameter('f', [0.4, 0.6]);

% Adaptation variants as grid parameters (replaces old conditions)
%   n_a_E: {0, 3}  -> no SFA vs SFA
%   n_b_E: {0, 1}  -> no STD vs STD
ps.add_grid_parameter('n_a_E', [0, 3]);
ps.add_grid_parameter('n_b_E', [0, 1]);

% Repetitions
ps.add_grid_parameter('reps', 1:3);

%% Run
ps.run();

% Save ParamSpace object
save_file = fullfile(ps.output_dir, 'ps_object.mat');
save(save_file, 'ps');
fprintf('ParamSpace object saved to: %s\n', save_file);

% Copy script for reproducibility
copyfile([mfilename('fullpath') '.m'], ps.output_dir);

%% Plot results (requires FractionalReservoir plotting utilities on path)
try
    ps.plot('metric', 'LLE');
    ps.plot('metric', 'mean_rate');
catch ME
    warning('Fig2:PlottingFailed', '%s', ME.message);
end

%% Save figures
if save_figs
    save_dir = fullfile(figs_root, 'fraction_excitatory_analysis');
    try
        save_some_figs_to_folder_2(save_dir, 'fraction_excitatory', [], {'fig', 'svg', 'png', 'jp2'});
        fprintf('Figures saved to %s\n', save_dir);

        data_source_file = fullfile(save_dir, 'data_source.txt');
        fid = fopen(data_source_file, 'w');
        fprintf(fid, 'Figures generated from data in:\n%s\n', ps.output_dir);
        fclose(fid);
    catch ME
        warning('Fig2:FigSaveFailed', '%s', ME.message);
    end
end

%% Summary
fprintf('\n=== Parameter Space Analysis Summary ===\n');
fprintf('Output directory: %s\n', ps.output_dir);
fprintf('Grid parameters: %s\n', strjoin(ps.grid_params, ', '));
fprintf('Levels per parameter: %d\n', ps.n_levels);
fprintf('\nDone! Results saved to: %s\n', ps.output_dir);
