function L_matrix = atmospheric_light_fixed(number_of_images, image_channels,...
    parameters)
%ATMOSPHERIC_LIGHT_FIXED  Generate fixed atmospheric light for a set of images.
%   Inputs:
%       -|number_of_images|: number of images for which atmospheric light is
%       simulated.
%       -|image_channels|: common number of channels shared by all images in the
%       set.
%       -|parameters|: structure containing miscellaneous parameters, such as
%       fixed atmospheric light intensity. Guarantees uniformity with other
%       functions that implement generation of atmospheric light.
%
%   Outputs:
%       -|L_matrix|: 1-by-|number_of_images|-by-|image_channels| matrix
%       containing the fixed value of atmospheric light for every image in the
%       set.

% Determine the fixed intensity of atmospheric light.
c = parameters.c;

% Generate fixed atmospheric light for all images.
L_matrix = repmat(c, 1, number_of_images, image_channels);

end

