% Fig_2_fraction_excitatory_load_and_plot.m
% Load saved ParamSpace results and regenerate plots + stats without re-running.
%
% Usage:
%   1. Edit 'results_dir' below to point at your param_space_* directory
%   2. Run this script (setup_paths must be called first)
%
% Ported from FractionalReservoir to use ParamSpace / refactored SRNNModel2.
%
% See also: Fig_2_fraction_excitatory_analysis, ParamSpace

%% ========== USER CONFIGURATION ==========

% Point this at your param_space_* output directory
results_dir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
    'data', 'param_space', 'fig2_frac_exc_nLevs_5_EDIT_ME');

% Plotting options
transient_skip = 3;
periods_to_plot = [0 1 1];
metrics_to_hist = {'br', 'lle'};

% Output control
save_figs = false;
save_stats = true;

% ==========================================

%% Master script override support
if exist('master_save_figs', 'var')
    if strcmp(master_save_figs, 'save_all_figs')
        save_figs = true;
    elseif strcmp(master_save_figs, 'save_no_figs')
        save_figs = false;
    end
end

%% Setup paths
setup_paths();

script_path = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_path));
figs_root = fullfile(project_root, 'figs');

%% Validate and load ParamSpace object
if ~exist(results_dir, 'dir')
    error('Results directory not found:\n  %s', results_dir);
end

ps_file = fullfile(results_dir, 'ps_object.mat');
if ~exist(ps_file, 'file')
    error('ps_object.mat not found in:\n  %s\nDid you run the analysis first?', results_dir);
end

fprintf('=== Loading ParamSpace Object ===\n');
fprintf('Directory: %s\n', results_dir);
loaded = load(ps_file);
ps = loaded.ps;

% Re-assign model factory (function handles don't survive save/load)
ps.model_factory = @(args) SRNNModel2(args{:});
ps.metric_extractor = @ParamSpace.srnn_metric_extractor;

fprintf('Loaded ParamSpace object: %d combinations, %d conditions\n\n', ...
    ps.num_combinations, length(ps.conditions));

%% Load results if needed
if ~ps.has_run
    fprintf('has_run is false — loading results from per-condition MAT files...\n');
    ps.load_results(results_dir);
end

%% Plot results (requires FractionalReservoir plotting utilities on path)
try
    [~, figs_hist] = load_and_make_unit_histograms(ps.output_dir, 'Metrics', metrics_to_hist);
    fig_paired_swarm = load_and_plot_lle_by_stim_period(ps.output_dir, ...
        'transient_skip', transient_skip, 'periods_to_plot', periods_to_plot);

    fig_combined = concatenate_figs([figs_hist, fig_paired_swarm], 'vertical', ...
        'HideTitlesAfterFirstRow', true);
catch ME
    fig_combined = [];
    warning('Plotting skipped (external plotting utilities not on path): %s', ME.message);
end

%% Statistical Analysis: Stim vs No-Stim (Wilcoxon signed-rank)
condition_names = cellfun(@(c) c.name, ps.conditions, 'UniformOutput', false);
num_conditions = length(condition_names);

% Step configuration from model_args
n_steps = 3;
no_stim_pattern = false(1, 3);
no_stim_pattern(1:2:end) = true;
T_stim = 45;  % T_range(2) from analysis
step_period = T_stim / n_steps;

stats_lines = {};
stats_lines{end+1} = '=== Statistical Analysis: Stim vs No-Stim ===';
stats_lines{end+1} = sprintf('Data directory: %s', results_dir);
stats_lines{end+1} = sprintf('Test: Wilcoxon signed-rank (paired, non-parametric)');
stats_lines{end+1} = sprintf('Transient skip: %.2f s', transient_skip);
stats_lines{end+1} = sprintf('Step period: %.2f s, No-stim pattern: [%s]', ...
    step_period, strjoin(string(no_stim_pattern), ', '));
stats_lines{end+1} = '';

for c_idx = 1:num_conditions
    cond_name = condition_names{c_idx};
    results_cell = ps.results.(cond_name);

    n_valid = 0;
    stim_means_all = [];
    no_stim_means_all = [];

    for k = 1:length(results_cell)
        res = results_cell{k};
        if ~isstruct(res) || ~isfield(res, 'success') || ~res.success
            continue;
        end
        if ~isfield(res, 'local_lya') || isempty(res.local_lya)
            continue;
        end

        t_lya = res.t_lya;
        local_lya = res.local_lya;

        valid_mask = t_lya >= 0;
        t_lya = t_lya(valid_mask);
        local_lya = local_lya(valid_mask);

        step_means = NaN(1, n_steps);
        for step_idx = 1:n_steps
            step_start = (step_idx - 1) * step_period + transient_skip;
            step_end = step_idx * step_period;
            step_mask = t_lya >= step_start & t_lya < step_end;
            if any(step_mask)
                step_means(step_idx) = mean(local_lya(step_mask), 'omitnan');
            end
        end

        stim_mean = mean(step_means(~no_stim_pattern), 'omitnan');
        no_stim_mean = mean(step_means(no_stim_pattern), 'omitnan');

        if ~isnan(stim_mean) && ~isnan(no_stim_mean)
            n_valid = n_valid + 1;
            stim_means_all(end+1) = stim_mean; %#ok<SAGROW>
            no_stim_means_all(end+1) = no_stim_mean; %#ok<SAGROW>
        end
    end

    if ps.condition_titles.isKey(cond_name)
        display_name = ps.condition_titles(cond_name);
    else
        display_name = strrep(cond_name, '_', ' ');
    end

    if n_valid >= 2
        stim_means_all = stim_means_all(:);
        no_stim_means_all = no_stim_means_all(:);

        [p_value, ~, ~] = signrank(stim_means_all, no_stim_means_all);

        differences = stim_means_all - no_stim_means_all;
        cohens_d = mean(differences) / std(differences);

        stats_lines{end+1} = sprintf('%s (n=%d pairs):', display_name, n_valid); %#ok<SAGROW>
        stats_lines{end+1} = sprintf('  Wilcoxon signed-rank p-value: %.4g', p_value); %#ok<SAGROW>
        stats_lines{end+1} = sprintf('  Cohen''s d: %.4f', cohens_d); %#ok<SAGROW>
        stats_lines{end+1} = sprintf('  Median LLE difference (stim - no-stim): %.4f', ...
            median(stim_means_all) - median(no_stim_means_all)); %#ok<SAGROW>
        stats_lines{end+1} = ''; %#ok<SAGROW>
    else
        stats_lines{end+1} = sprintf('%s: Insufficient valid pairs (n=%d)', ...
            display_name, n_valid); %#ok<SAGROW>
        stats_lines{end+1} = ''; %#ok<SAGROW>
    end
end

% Print stats
fprintf('\n');
for i = 1:length(stats_lines)
    fprintf('%s\n', stats_lines{i});
end

% Write stats to text file
if save_stats
    if save_figs
        stats_dir = fullfile(figs_root, 'fraction_excitatory_analysis');
    else
        stats_dir = results_dir;
    end
    if ~exist(stats_dir, 'dir')
        mkdir(stats_dir);
    end
    stats_file = fullfile(stats_dir, 'stats_results.txt');
    fid = fopen(stats_file, 'w');
    if fid == -1
        warning('Could not open stats file for writing: %s', stats_file);
    else
        fprintf(fid, 'Generated: %s\n\n', string(datetime('now')));
        for i = 1:length(stats_lines)
            fprintf(fid, '%s\n', stats_lines{i});
        end
        fclose(fid);
        fprintf('Stats written to: %s\n', stats_file);
    end
end

%% Panel letters
if ~isempty(fig_combined)
    try
        drawnow
        AddLetters2Plots(fig_combined, ...
            {'(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)'}, ...
            'FontSize', 14, 'FontWeight', 'normal', 'HShift', -0.03, 'VShift', -0.04);
    catch ME
        warning('AddLetters2Plots failed (non-critical): %s', ME.message);
    end
end

%% Save figures
if save_figs
    save_dir = fullfile(figs_root, 'fraction_excitatory_analysis');
    try
        save_some_figs_to_folder_2(save_dir, 'fraction_excitatory', [], {'fig', 'svg', 'png', 'jp2'});
        fprintf('Figures saved to %s\n', save_dir);
    catch ME
        warning('Figure saving failed: %s', ME.message);
    end
end

%% Summary
fprintf('\n=== Summary ===\n');
fprintf('Output directory: %s\n', ps.output_dir);
fprintf('Grid parameters: %s\n', strjoin(ps.grid_params, ', '));
fprintf('Conditions: %s\n', strjoin(cellfun(@(c) c.name, ps.conditions, 'UniformOutput', false), ', '));
fprintf('\nDone! Plots and stats generated from: %s\n', ps.output_dir);
