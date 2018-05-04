function [B, f_x] = camera_parameters_cityscapes(camera_parameters_file)
%CAMERA_PARAMETERS_CITYSCAPES  Read camera parameters for a Cityscapes image
%from a JSON file.
%
%   INPUTS:
%
%   -|camera_parameters_file|: full path to JSON file where camera parameters
%   are stored.
%
%   OUTPUTS:
%
%   -|B|: scalar of type double, corresponding to the baseline of the stereo
%   pair for the relevant image, measured in meters.
%
%   -|f_x|: scalar of type double, corresponding to the focal length parameter
%   in x-axis (incorporating aspect ratio), measured in pixels.

% addpath('../../../json');
fid = fopen(camera_parameters_file);
raw = fread(fid, inf);
str = char(raw');
fclose(fid);
camera_parameters = JSON.parse(str);
B = camera_parameters.extrinsic.baseline;
f_x = camera_parameters.intrinsic.fx;

end

