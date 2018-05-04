function clean2hazy(input_image_file_names, are_rgb, depth_file_names,...
    scattering_coefficient_method, scattering_coefficient_method_parameters,...
    atmospheric_light_method, atmospheric_light_method_parameters,...
    depth_preprocessing_method, depth_preprocessing_method_parameters,...
    transmission_model, haze_model, output_directory, output_format)
%CLEAN2HAZY  Simulate haze using scene radiance and depth data.
%   INPUTS:
%
%   -|input_image_file_names|: cell array of strings with full paths to clean
%   input images.
%
%   -|are_rgb|: binary flag which is true when input images are of RGB type.
%
%   -|depth_file_names|: cell array of strings with full paths to depth maps for
%   input images.
%
%   -|scattering_coefficient_method|: handle of the function which is specified
%   as the scattering coefficient method.
%
%   -|scattering_coefficient_method_parameters|: structure containing the
%   various parameters of the scattering coefficient method as its fields.
%
%   -|atmospheric_light_method|: handle of the function which is specified as
%   the atmospheric light method.
%
%   -|atmospheric_light_method_parameters|: structure containing the various
%   parameters of the atmospheric light method as its fields.
%
%   -|depth_preprocessing_method|: handle of the function which is specified as
%   the method to preprocess the input depth map.
%
%   -|depth_preprocessing_method_parameters|: extra parameter for specified
%   method for depth preprocessing, apart from basic depth input.
%
%   -|transmission_model|: handle of the function that implements computation of
%   transmission map given a depth map.
%
%   -|haze_model|: handle of the function that implements haze generation from a
%   clean image.
%
%   -|output_directory|: full path to directory where the synthetic hazy output
%   images are saved.
%
%   -|output_format|: string specifying the image format for the hazy output
%   images, e.g. '.png'

% Total number of processed images. Should be equal to number of files with
% corresponding depth data.
number_of_images = length(input_image_file_names);
assert(number_of_images == length(depth_file_names));

% Create output directory where synthetic hazy images will be saved, if it does
% not already exist.
if exist(output_directory) ~= 7
    mkdir(output_directory);
end

% Distinguish between RGB and grayscale images.
if are_rgb
    image_channels = 3;
else
    image_channels = 1;
end

% Generate scattering coefficient values for the set of processed images.
beta_vector = scattering_coefficient_method(number_of_images,...
    scattering_coefficient_method_parameters);

% Generate atmospheric light values for the set of processed images.
L_matrix = atmospheric_light_method(number_of_images, image_channels,...
    atmospheric_light_method_parameters);

% Compute and save hazy images.
for i = 1:number_of_images
    
    % Read depth input.
    % load(depth_file_names{i}, 'depth_map');
    depth_input = imread(depth_file_names{i});
    
    % Process depth input, so that |depth_map| is a 2-dimensional matrix in
    % double format with the same resolution as the input image, containing
    % depth values in meters, ideally without any invalid parts.
    depth_map = depth_preprocessing_method(depth_input,...
        depth_preprocessing_method_parameters{i});
    
    % Compute transmission map using the specified transmission model.
    t = transmission_model(depth_map, beta_vector(i));
    
    % Read clean image and bring it to double precision for subsequent
    % computations.
    R = im2double(imread(input_image_file_names{i}));
    
    % Simulate haze using the specified haze model and the computed transmission
    % map and atmospheric light for the current image.
    I = haze_model(R, t, L_matrix(1, i, :));
    
    % Save the synthetic hazy image in the output directory with a (preferably)
    % lossless format.
    
    % Base name of output image is drawn from input image.
    hazy_image_base_name =...
        file_name_from_path_no_extension(input_image_file_names{i});
    
    % Suffix specifying the parameter values that were used to generate haze.
    % This naming convention assumes homogeneous graylevel atmospheric light.
    parameters_suffix = strcat('_hazy-beta_', num2str(beta_vector(i)),...
        '-c_', num2str(L_matrix(1, i, 1)));
    
    % Build image name from base name, parameters suffix and output format and
    % save it.
    imwrite(I, strcat(output_directory, hazy_image_base_name,...
        parameters_suffix, output_format));
end

end

