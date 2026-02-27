function setup_paths()
% SETUP_PATHS Add src/ directories to MATLAB path for LNN scripts.
%
% Adds the LNN/src/ directory (containing LNN.m) and the SRNN/src/
% directory (containing RMTMatrix.m) to the MATLAB path using the
% location of this script as reference.

    script_dir = fileparts(mfilename('fullpath'));
    project_dir = fileparts(script_dir);  % Go up from scripts/ to LNN/
    matlab_dir = fileparts(project_dir);  % Go up from LNN/ to Matlab/

    lnn_src = fullfile(project_dir, 'src');
    srnn_src = fullfile(matlab_dir, 'SRNN', 'src');

    addpath(genpath(lnn_src));
    fprintf('Added to path: %s\n', lnn_src);
    addpath(genpath(srnn_src));
    fprintf('Added to path: %s\n', srnn_src);
end
