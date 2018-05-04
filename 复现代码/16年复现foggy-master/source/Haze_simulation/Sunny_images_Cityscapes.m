% Identify Cityscapes images with very bright atmospheric light, unsuitable for
% synthesis of foggy counterparts, and write a text file that lists all of them.

close all;
clear;

% Read the necessary files.

cityscapes_output_root_directory =...
    '/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/data/Cityscapes';

atmospheric_light_file = 'leftImg8bit_trainvaltest_atmlight.mat';
% Loads variable |L| to workspace.
load(fullfile(cityscapes_output_root_directory, atmospheric_light_file));

image_list_file = 'leftImg8bit_orig_trainvaltest_filenames.txt';
fid = fopen(fullfile(cityscapes_output_root_directory, image_list_file));
image_file_names = textscan(fid, '%s');
fclose(fid);
image_file_names = image_file_names{1};

% Ensure that the number of file names is equal with the number of atmospheric
% light values that have been loaded.
assert(length(image_file_names) == size(L, 1));

% Isolate images with very bright atmospheric light, i.e. "sunny" images.

brightness_threshold = 0.99;
is_L_bright = is_bright(L, brightness_threshold);
sunny_image_file_names = image_file_names(is_L_bright);
number_of_sunny_images = length(sunny_image_file_names);

% Write text file with list of "sunny" images.

sunny_image_list_file = 'leftImg8bit_orig_trainvaltest_sunny_filenames.txt';
fid = fopen(fullfile(cityscapes_output_root_directory,...
    sunny_image_list_file), 'w');
for i = 1:number_of_sunny_images
    fprintf(fid, '%s\n', sunny_image_file_names{i});
end
fclose(fid);
