% Identify Cityscapes images from train and val set with almost no sky,
% unsuitable for synthesis of foggy counterparts due to inability of correct
% atmospheric light estimation via DCP-based methods, and write a text file that
% lists the left image name for all of them.

close all;
clear;

% Read the necessary files.

cityscapes_output_root_directory =...
    '/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/data/Cityscapes';

gt_list_file = 'gtFine_trainval_filenames.txt';
fid = fopen(fullfile(cityscapes_output_root_directory, gt_list_file));
gt_file_names = textscan(fid, '%s');
fclose(fid);
gt_file_names = gt_file_names{1};
% Total number of processed images.
number_of_images = length(gt_file_names);

% Isolate images for which sky is almost completely invisible.

neighborhood_size_dark_channel = 15;
se = strel('square', neighborhood_size_dark_channel);
sky_label = 23;
brightest_pixels_fraction = 1 / 1000;
no_sky = false(number_of_images, 1);

for i = 1:number_of_images
    
    % Read ground truth image.
    GT = imread(gt_file_names{i});
    [height, width] = size(GT);
    number_of_pixels = height * width;
    
    % Determine whether enough sky pixels participate in atmospheric light
    % estimation with the method of the Regression Forests paper.
    number_of_pure_sky_pixels = sky_pixels_pure_dark_channel(GT, sky_label, se);
    B = brightest_pixels_count_rf(number_of_pixels, brightest_pixels_fraction);
    no_sky(i) = number_of_pure_sky_pixels < ceil(B / 2);
    
end

no_sky_gt_file_names = gt_file_names(no_sky);
no_sky_image_file_names = strrep(no_sky_gt_file_names,...
    'gtFine_trainvaltest/gtFine', 'leftImg8bit');
no_sky_image_file_names = strrep(no_sky_image_file_names,...
    'gtFine_labelIds', 'leftImg8bit');
number_of_no_sky_images = length(no_sky_image_file_names);

% Write text file with list of "no sky" images.

no_sky_image_list_file = 'leftImg8bit_orig_trainval_no_sky_brightest_dark_filenames.txt';
fid = fopen(fullfile(cityscapes_output_root_directory,...
    no_sky_image_list_file), 'w');
for i = 1:number_of_no_sky_images
    fprintf(fid, '%s\n', no_sky_image_file_names{i});
end
fclose(fid);
