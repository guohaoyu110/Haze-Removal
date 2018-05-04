close all;
clear;

% Read the necessary files.

cityscapes_output_root_directory =...
    '/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/data/Cityscapes';

sunny_list_file = 'leftImg8bit_orig_trainval_sunny_filenames.txt';
fid = fopen(fullfile(cityscapes_output_root_directory, sunny_list_file));
sunny_file_names = textscan(fid, '%s');
fclose(fid);
sunny_file_names = sunny_file_names{1};

no_sky_list_file =...
    'leftImg8bit_orig_trainval_no_sky_atmospheric_light_filenames.txt';
fid = fopen(fullfile(cityscapes_output_root_directory, no_sky_list_file));
no_sky_file_names = textscan(fid, '%s');
fclose(fid);
no_sky_file_names = no_sky_file_names{1};

% Merge the two lists into one.

sunny_or_no_sky_file_names = [sunny_file_names; no_sky_file_names];
sunny_or_no_sky_file_names = unique(sunny_or_no_sky_file_names);
number_of_sunny_or_no_sky_images = length(sunny_or_no_sky_file_names);

% Write text file with list of "sunny" or "no sky" images.

sunny_or_no_sky_image_list_file = 'leftImg8bit_orig_trainval_sunny_no_sky_filenames.txt';
fid = fopen(fullfile(cityscapes_output_root_directory,...
    sunny_or_no_sky_image_list_file), 'w');
for i = 1:number_of_sunny_or_no_sky_images
    fprintf(fid, '%s\n', sunny_or_no_sky_file_names{i});
end
fclose(fid);
