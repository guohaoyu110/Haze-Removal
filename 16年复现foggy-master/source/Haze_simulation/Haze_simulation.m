function Haze_simulation(data_root_directory, input_image_data_subdirectory,...
    depth_data_subdirectory, camera_parameters_directory, output_directory)
%HAZE_SIMULATION  Simulate haze for all images in input directory using the
%corresponding depth maps and write results to output directory.
%   INPUTS:
%
%   -|data_root_directory|: path to root directory for input image and depth
%   subdirectories. ATTENTION: Should end with a /
%
%   -|input_image_data_subdirectory|: relative path to input image subdirectory
%   starting from |data_root_directory|. ATTENTION: Should end with a /
%
%   -|depth_data_subdirectory|: relative path to depth subdirectory starting
%   from |data_root_directory|. ATTENTION: Should end with a /
%
%   -|camera_parameters_directory|: full path to directory containing files
%   with camera parameters. ATTENTION: Should end with a /
%
%   -|output_directory|: full path to directory where the synthetic hazy output
%   images are saved. ATTENTION: Should end with a /

% Get all names of input images.
input_image_data_subdirectory_full_path = strcat(data_root_directory,...
    input_image_data_subdirectory);
input_image_file_names =...
    file_full_names_in_directory(input_image_data_subdirectory_full_path);

% Determine whether input images are RGB, assuming that they all have the same
% number of channels.
I_1 = imread(input_image_file_names{1});
are_rgb = ndims(I_1) == 3;

% Get all names of depth map files.
depth_data_subdirectory_full_path = strcat(data_root_directory,...
    depth_data_subdirectory);
depth_file_names =...
    file_full_names_in_directory(depth_data_subdirectory_full_path);

% Select and instantiate the function that computes scattering coefficients.
scattering_coefficient_method = @scattering_coefficient_random;
scattering_coefficient_maximum_value = 0.1;
scattering_coefficient_minimum_value = 0;
scattering_coefficient_random_generator = 'default';
scattering_coefficient_configure_random_generator = 0;
beta = 0.1;
scattering_coefficient_method_parameters =...
    instantiate_scattering_coefficient_method(scattering_coefficient_method,...
    scattering_coefficient_maximum_value,...
    scattering_coefficient_minimum_value,...
    scattering_coefficient_random_generator,...
    scattering_coefficient_configure_random_generator, beta);

% Select and instantiate the function that computes atmospheric light for each
% image.
atmospheric_light_method = @atmospheric_light_random;
atmospheric_light_maximum_intensity = 1;
atmospheric_light_minimum_intensity = 0.8;
atmospheric_light_random_generator = 'default';
atmospheric_light_configure_random_generator = 0;
c = 0.9;
atmospheric_light_method_parameters =...
    instantiate_atmospheric_light_method(atmospheric_light_method,...
    atmospheric_light_maximum_intensity, atmospheric_light_minimum_intensity,...
    atmospheric_light_random_generator,...
    atmospheric_light_configure_random_generator, c);

% Select the function that performs preprocessing on the input depth maps.
addpath('../Input_preprocessing');
depth_preprocessing_method = @depth_in_meters_cityscapes_with_invalid_parts;
depth_preprocessing_method_parameters =...
    instantiate_depth_preprocessing_method(depth_preprocessing_method,...
    depth_file_names, camera_parameters_directory);

% Select the transmission model of the clear scene radiance.
transmission_model = @transmission_exponential;

% Select the haze model which is used for the synthesis of hazy images from the
% clean input images.
haze_model = @haze_linear;

% Specify output format for synthetic hazy images.
output_format = '.png';

% Run haze simulation on input images using the specified parameter values.
clean2hazy(input_image_file_names, are_rgb, depth_file_names,...
    scattering_coefficient_method, scattering_coefficient_method_parameters,...
    atmospheric_light_method, atmospheric_light_method_parameters,...
    depth_preprocessing_method, depth_preprocessing_method_parameters,...
    transmission_model, haze_model, output_directory, output_format);

end

