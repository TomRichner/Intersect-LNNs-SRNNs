function setup_paths()
% SETUP_PATHS Add src/ directories to MATLAB path for SRNN scripts.
%
% Adds SRNN/src/ (SRNNModel2) and shared/src/ (cRNN, Activation,
% Stimulus, RMTMatrix) to the MATLAB path.

    script_dir = fileparts(mfilename('fullpath'));
    project_dir = fileparts(script_dir);  % Go up from scripts/ to SRNN/
    matlab_dir = fileparts(project_dir);  % Go up from SRNN/ to Matlab/

    src_dir = fullfile(project_dir, 'src');
    shared_src = fullfile(matlab_dir, 'shared', 'src');

    addpath(genpath(src_dir));
    fprintf('Added to path: %s\n', src_dir);
    addpath(genpath(shared_src));
    fprintf('Added to path: %s\n', shared_src);
end
