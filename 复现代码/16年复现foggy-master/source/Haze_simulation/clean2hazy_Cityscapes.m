function clean2hazy_Cityscapes(left_image_file_names, right_image_file_names,...
    disparity_file_names, camera_parameters_file_names,...
    scattering_coefficient_method, atmospheric_light_estimation,...
    transmission_model, haze_model, cityscapes_hazy_output_directory,...
    cityscapes_transmission_output_directory, output_format)
%CLEAN2HAZY_CITYSCAPES  Simulate haze for Cityscapes.
%   INPUTS:
%
%   -|left_image_file_names|: cell array of strings with full paths to clean
%   left input images.
%
%   -|right_image_file_names|: cell array of strings with full paths to clean
%   right input images.
%
%   -|disparity_file_names|: cell array of strings with full paths to disparity
%   maps for input images.
%
%   -|camera_parameters_file_names|: cell array of strings with full paths to
%   JSON files where camera parameters are stored.
%
%   -|scattering_coefficient_method|: handle of the function which is specified
%   as the scattering coefficient method.
%
%   -|atmospheric_light_estimation|: handle of the function which is used to
%   estimate atmospheric light.
%
%   -|transmission_model|: handle of the function that implements computation of
%   transmission map given a depth map.
%
%   -|haze_model|: handle of the function that implements haze generation from a
%   clean image.
%
%   -|cityscapes_hazy_output_directory|: full path to root directory where the
%   synthetic hazy output Cityscapes images are saved.
%
%   -|cityscapes_transmission_output_directory|: full path to root directory
%   where the transmission maps for hazy Cityscapes images are saved.
%
%   -|output_format|: string specifying the image format for the hazy output
%   images, e.g. '.png'

% Total number of processed images. Should be equal to number of files for each
% auxiliary set.
number_of_images = length(left_image_file_names);
assert(number_of_images == length(right_image_file_names));
assert(number_of_images == length(disparity_file_names));
assert(number_of_images == length(camera_parameters_file_names));

% Generate scattering coefficient values for the set of processed images.
beta_parameters.beta = 0.01;
beta_vector = scattering_coefficient_method(number_of_images, beta_parameters);

% Compute and save hazy images and transmission maps.
for i = 1:number_of_images
    
    % Compute the hazy image from its original clean version, the right image of
    % the stereo pair, the disparity map for the left view and the intrinsic
    % parameters of the camera.
    
    % Read input disparity map.
    input_disparity = imread(disparity_file_names{i});
    
    % Read clean images and bring left image to double precision for subsequent
    % computations.
    R_left_uint8 = imread(left_image_file_names{i});
    R_left = im2double(R_left_uint8);
    R_right = im2double(imread(right_image_file_names{i}));
    
    % Process depth input, so that |depth_map| is a 2-dimensional matrix in
    % double format with the same resolution as the input image, containing
    % depth values in meters, with as few invalid parts as possible.
    depth_map =...
        depth_in_meters_cityscapes_stereoscopic_inpainting(input_disparity,...
        R_left, R_left_uint8, R_right, camera_parameters_file_names{i});
    
    % Compute transmission map using the specified transmission model.
    t = transmission_model(depth_map, beta_vector(i));
    
    % Refine transmission map using guided filtering with clean image as
    % guidance.
    window_size = 41;
    mu = 1e-3;
    t = clip_to_unit_range(transmission_guided_filtering(t, R_left,...
        window_size, mu));
    
    % Estimate atmospheric light for the clean image.
    neighborhood_size_dark_channel = 15;
    R_left_dark = get_dark_channel(R_left, neighborhood_size_dark_channel);
    L = atmospheric_light_estimation(R_left_dark, R_left);
    
    % Simulate haze using the specified haze model and the computed transmission
    % map and atmospheric light for the current image.
    I = haze_model(R_left, t, L);
    
    % --------------------------------------------------------------------------
    
    % Save the hazy image and the transmission map in the respective output
    % directories in a (preferably) lossless format.
    
    % The name of and path to the output image is based on the input image.
    [path_to_input, R_left_name] =...
        fileparts(left_image_file_names{i});
    
    % Suffices specifying the parameter values that were used to generate haze.
    parameters_suffix_hazy = strcat('_hazy-beta_', num2str(beta_vector(i)));
    parameters_suffix_transmission = strcat('_transmission-beta_',...
        num2str(beta_vector(i)));
    
    % Full names of output images formed by the name of the input image and the
    % suffices.
    I_name_with_extension = strcat(R_left_name, parameters_suffix_hazy,...
        output_format);
    t_name_with_extension = strcat(R_left_name,...
        parameters_suffix_transmission, output_format);
    
    % Determine output directories based on the directory structure of original
    % Cityscapes dataset: 1) train-val-test directories, 2) city directories.
    path_to_input_split = strsplit(path_to_input, filesep);
    current_hazy_output_directory =...
        fullfile(cityscapes_hazy_output_directory,...
        path_to_input_split{end - 1}, path_to_input_split{end});
    current_transmission_output_directory =...
        fullfile(cityscapes_transmission_output_directory,...
        path_to_input_split{end - 1}, path_to_input_split{end});
    
    % Create output directories where synthetic hazy images and transmission
    % maps will be saved, if they do not already exist.
    if exist(current_hazy_output_directory) ~= 7
        mkdir(current_hazy_output_directory);
    end
    if exist(current_transmission_output_directory) ~= 7
        mkdir(current_transmission_output_directory);
    end
    
    % Build image names from base name, parameters suffices and output format
    % and save them.
    imwrite(I, fullfile(current_hazy_output_directory, I_name_with_extension));
    imwrite(t, fullfile(current_transmission_output_directory,...
        t_name_with_extension));
end

end

