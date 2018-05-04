close all;
clear;

% Read the necessary files.

cityscapes_output_root_directory =...
    '/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/data/Cityscapes';

good_image_list_file = 'leftImg8bit_orig_trainval_sunny_no_sky_filenames.txt';
fid = fopen(fullfile(cityscapes_output_root_directory, good_image_list_file));
sunny_or_no_sky_file_names = textscan(fid, '%s');
fclose(fid);
sunny_or_no_sky_file_names = sunny_or_no_sky_file_names{1};

complete_list_file =...
    'leftImg8bit_orig_trainval_filenames.txt';
fid = fopen(fullfile(cityscapes_output_root_directory, complete_list_file));
all_file_names = textscan(fid, '%s');
fclose(fid);
all_file_names = all_file_names{1};

% Compute set difference between list with all train+val filenames and list with
% "sunny" or "no sky" filenames.

good_image_file_names = all_file_names(~ismember(all_file_names,...
    sunny_or_no_sky_file_names));
number_of_good_images = length(good_image_file_names);

% Write text file with list of "good" images.

good_image_list_file = 'leftImg8bit_orig_trainval_refined_auto_filenames.txt';
fid = fopen(fullfile(cityscapes_output_root_directory,...
    good_image_list_file), 'w');
for i = 1:number_of_good_images
    fprintf(fid, '%s\n', good_image_file_names{i});
end
fclose(fid);
