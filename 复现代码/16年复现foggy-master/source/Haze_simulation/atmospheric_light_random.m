function L_matrix = atmospheric_light_random(number_of_images,...
    image_channels, parameters)
%ATMOSPHERIC_LIGHT_RANDOM  Generate atmospheric light values for a set of images
%uniformly at random.
%   Inputs:
%       -|number_of_images|: number of images for which atmospheric light is
%       simulated.
%       -|image_channels|: common number of channels shared by all images in the
%       set.
%       -|parameters|: structure containing miscellaneous parameters, such as
%       minimum intensity of atmospheric light and type of random number
%       generator. Guarantees uniformity with other functions that implement
%       generation of atmospheric light.
%
%   Outputs:
%       -|L_matrix|: 1-by-|number_of_images|-by-|image_channels| matrix
%       containing the random values of atmospheric light for every image in the
%       set.

% Determine the range of random values for intensity of atmospheric light.
maximum_intensity = parameters.maximum_intensity;
minimum_intensity = parameters.minimum_intensity;

% Get type of random number generator, e.g. 'default'.
random_generator = parameters.random_generator;

% Get binary flag that indicates whether the random number generator should be
% configured or not.
configure_random_generator = parameters.configure_random_generator;

% Optionally configure random number generation for repeatability.
if configure_random_generator
    rng(random_generator);
end

% Generate random intensity of atmospheric light for each image following a
% uniform distribution inside the specified range.
c = minimum_intensity + (maximum_intensity - minimum_intensity) *...
    rand(1, number_of_images);

% Repeat intensity values in all channels.
L_matrix = repmat(c, 1, 1, image_channels);

end

