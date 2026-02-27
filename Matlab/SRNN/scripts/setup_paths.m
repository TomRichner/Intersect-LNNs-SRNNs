function setup_paths()
% SETUP_PATHS Add src/ directory to MATLAB path for LNN scripts.
%
% Adds the src/ directory (containing SRNNModel2.m, RMTMatrix.m)
% to the MATLAB path using the location of this script as reference.

    script_dir = fileparts(mfilename('fullpath'));
    project_dir = fileparts(script_dir);  % Go up from scripts/ to LNN/
    src_dir = fullfile(project_dir, 'src');

    addpath(genpath(src_dir));
    fprintf('Added to path: %s\n', src_dir);
end
