function depth_map_in_meters = depth_in_meters_synthia(input_depth_map,...
    parameters)
%DEPTH_IN_METERS_SYNTHIA  Compute depth map in meters from provided SYNTHIA
%depth map.
%   INPUTS:
%
%   -|input_depth_map|: 3-dimensional matrix in uint16 format with depth in cm
%   at each of its pages.
%
%   -|parameters|: dummy input, defined in order to make the function's
%   signature identical to that of other functions implementing the same depth
%   preprocessing interface, e.g. for Cityscapes.
%
%   OUTPUTS:
%
%   -|depth_map_in_meters|: 2-dimensional matrix in double format containing
%   depth in meters.

depth_map_in_meters = double(input_depth_map(:, :, 1)) / 100;

end

