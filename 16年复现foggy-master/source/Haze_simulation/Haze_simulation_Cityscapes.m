function Haze_simulation_Cityscapes(task_id)
%HAZE_SIMULATION_CITYSCAPES  Simulate haze for a batch of images from Cityscapes
%and write results. Structured for execution on a cluster.
%   INPUTS:
%
%   -|task_id|: ID of the task. Used to determine which images out of the entire
%   dataset will form the batch that will be processed by this task.

if(ischar(task_id))
    task_id = str2double(task_id);
end

source_code_root_directory = '/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/foggy/source';
cityscapes_output_root_directory = '/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/data/Cityscapes';
cityscapes_hazy_output_directory = fullfile(cityscapes_output_root_directory,...
    'leftImg8bit_hazy_trainvaltest-beta_0.01');
cityscapes_transmission_output_directory = fullfile(cityscapes_output_root_directory,...
    'leftImg8bit_transmission_trainvaltest-beta_0.01');

% Add paths to functions that are called for haze simulation.
addpath(fullfile(source_code_root_directory, 'Haze_simulation'));
addpath(fullfile(source_code_root_directory, 'Input_preprocessing'));
addpath(fullfile(source_code_root_directory, 'Tools'));
addpath(fullfile(source_code_root_directory, 'Dehazing/Dark_channel_prior'));
addpath('/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/foggy/SLIC_mex');
addpath('/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/json');

% Get all names of left input images.
fid = fopen(fullfile(cityscapes_output_root_directory,...
    'leftImg8bit_orig_trainvaltest_filenames.txt'));
left_image_file_names = textscan(fid, '%s');
fclose(fid);
left_image_file_names = left_image_file_names{1};

% Get all names of right input images.
fid = fopen(fullfile(cityscapes_output_root_directory,...
    'rightImg8bit_orig_trainvaltest_filenames.txt'));
right_image_file_names = textscan(fid, '%s');
fclose(fid);
right_image_file_names = right_image_file_names{1};

% Get all names of disparity map files.
fid = fopen(fullfile(cityscapes_output_root_directory,...
    'disparity_orig_trainvaltest_filenames.txt'));
disparity_file_names = textscan(fid, '%s');
fclose(fid);
disparity_file_names = disparity_file_names{1};

% Get all names of camera parameters files.
fid = fopen(fullfile(cityscapes_output_root_directory,...
    'camera_orig_trainvaltest_filenames.txt'));
camera_parameters_file_names = textscan(fid, '%s');
fclose(fid);
camera_parameters_file_names = camera_parameters_file_names{1};

% Total number of processed images. Should be equal to number of files for each
% auxiliary set.
number_of_images = length(left_image_file_names);
assert(number_of_images == length(right_image_file_names));
assert(number_of_images == length(disparity_file_names));
assert(number_of_images == length(camera_parameters_file_names));

% Number of images handled per task.
% 5000 Cityscapes images = 50 images per task * 100 tasks.
images_per_task = 50;

batch_ind = (task_id - 1) * images_per_task + 1:task_id * images_per_task; 

if batch_ind(1) > number_of_images
    return;
end

% Truncate for last task.
if batch_ind(end) > number_of_images
    batch_ind = batch_ind(1:number_of_images - batch_ind(1) + 1);
end

% Select and instantiate the function that computes scattering coefficients. Set
% attenuation coefficient to meaningful values.
scattering_coefficient_method = @scattering_coefficient_fixed;

% Select and instantiate the function that estimates atmospheric light for each
% image.
atmospheric_light_estimation = @estimate_atmospheric_light_rf;

% Select the transmission model of the clear scene radiance.
transmission_model = @transmission_exponential;

% Select the haze model which is used for the synthesis of hazy images from the
% clean input images.
haze_model = @haze_linear;

% Specify output format for synthetic hazy images.
output_format = '.png';

% Run haze simulation on input images using the specified parameter values.
clean2hazy_Cityscapes(left_image_file_names(batch_ind),...
    right_image_file_names(batch_ind), disparity_file_names(batch_ind),...
    camera_parameters_file_names(batch_ind), scattering_coefficient_method,...
    atmospheric_light_estimation, transmission_model, haze_model,...
    cityscapes_hazy_output_directory,...
    cityscapes_transmission_output_directory, output_format);

end

