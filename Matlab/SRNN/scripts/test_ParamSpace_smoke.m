% test_ParamSpace_smoke.m - Quick smoke test of ParamSpace class
%
% Tests object creation, grid setup, and a tiny run (n=10, 1 rep, 1 condition,
% lya_method='none' for speed).

close all; clear; clc;

setup_paths();

fprintf('=== ParamSpace Smoke Test ===\n\n');

%% Test 1: Construction
fprintf('Test 1: Construction...\n');
ps = ParamSpace('n_levels', 2, 'batch_size', 10, 'note', 'smoke_test', 'verbose', true);
assert(isa(ps, 'ParamSpace'), 'Failed: not a ParamSpace instance');
assert(ps.n_levels == 2, 'Failed: n_levels not set');
fprintf('  PASSED\n\n');

%% Test 2: Model factory assignment
fprintf('Test 2: Model factory...\n');
ps.model_factory = @(args) SRNNModel2(args{:});
ps.metric_extractor = @ParamSpace.srnn_metric_extractor;
assert(~isempty(ps.model_factory), 'Failed: model_factory empty');
fprintf('  PASSED\n\n');

%% Test 3: model_args
fprintf('Test 3: model_args...\n');
ps.model_args = {'n', 10, 'T_range', [0, 5], 'lya_method', 'none', 'tau_d', 0.1};
assert(length(ps.model_args) == 8, 'Failed: wrong model_args length');
fprintf('  PASSED\n\n');

%% Test 4: add_grid_parameter
fprintf('Test 4: add_grid_parameter...\n');
ps.add_grid_parameter('f', [0.4, 0.6]);
ps.add_grid_parameter('reps', 1:2);
assert(length(ps.grid_params) == 2, 'Failed: wrong number of grid params');
assert(strcmp(ps.grid_params{1}, 'f'), 'Failed: first grid param not f');
fprintf('  PASSED\n\n');

%% Test 5: set_conditions
fprintf('Test 5: set_conditions...\n');
ps.set_conditions({ ...
    struct('name', 'no_adaptation', 'n_a_E', 0, 'n_b_E', 0) ...
    });
assert(length(ps.conditions) == 1, 'Failed: wrong number of conditions');
fprintf('  PASSED\n\n');

%% Test 6: set condition_titles
fprintf('Test 6: condition_titles...\n');
ps.condition_titles = containers.Map({'no_adaptation'}, {'No Adaptation'});
assert(ps.condition_titles.isKey('no_adaptation'), 'Failed: title not set');
fprintf('  PASSED\n\n');

%% Test 7: Run (tiny config: n=10, T=[0,5], no LLE, 1 condition, 2x2 grid)
fprintf('Test 7: Run (tiny grid)...\n');
ps.output_dir = fullfile(tempdir, 'paramspace_smoke_test');
ps.use_parallel = false;
ps.run();
assert(ps.has_run, 'Failed: has_run not set');
assert(isfield(ps.results, 'no_adaptation'), 'Failed: no results for condition');
fprintf('  PASSED\n\n');

%% Test 8: Results contain expected fields
fprintf('Test 8: Result fields...\n');
res = ps.results.no_adaptation{1};
assert(isstruct(res), 'Failed: result not struct');
assert(isfield(res, 'success'), 'Failed: no success field');
assert(isfield(res, 'config'), 'Failed: no config field');
assert(isfield(res, 'config_idx'), 'Failed: no config_idx field');
assert(isfield(res, 'mean_rate'), 'Failed: no mean_rate field (SRNN extractor)');
fprintf('  PASSED\n\n');

%% Cleanup
rmdir(ps.output_dir, 's');
fprintf('=== All tests passed! ===\n');
