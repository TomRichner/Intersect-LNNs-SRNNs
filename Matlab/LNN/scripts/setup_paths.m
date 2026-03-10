function setup_paths()
% SETUP_PATHS Add src/ directories to MATLAB path for LNN scripts.
%
% Adds LNN/src/ (LNN.m), SRNN/src/ (SRNNModel2), and shared/src/
% (cRNN, Activation, Stimulus, RMTMatrix) to the MATLAB path.

    script_dir = fileparts(mfilename('fullpath'));
    project_dir = fileparts(script_dir);  % Go up from scripts/ to LNN/
    matlab_dir = fileparts(project_dir);  % Go up from LNN/ to Matlab/

    lnn_src = fullfile(project_dir, 'src');
    srnn_src = fullfile(matlab_dir, 'SRNN', 'src');
    shared_src = fullfile(matlab_dir, 'shared', 'src');

    addpath(genpath(lnn_src));
    fprintf('Added to path: %s\n', lnn_src);
    addpath(genpath(srnn_src));
    fprintf('Added to path: %s\n', srnn_src);
    addpath(genpath(shared_src));
    fprintf('Added to path: %s\n', shared_src);
end
