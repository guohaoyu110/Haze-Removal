% Identify Cityscapes images from train and val set for which atmospheric light
% has been estimated by selecting a pixel that does not belong to the sky class,
% and write a text file that lists the left image name for all of them.
% These images are unsuitable for synthesis of foggy counterparts due to
% incorrect atmospheric light estimation via DCP-based methods.

close all;
clear;

% Read the necessary files.

cityscapes_output_root_directory =...
    '/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/data/Cityscapes';

indices_L_file = 'leftImg8bit_trainvaltest_atmlight_pixelind.mat';
% Loads variable |indices_L| to workspace.
load(fullfile(cityscapes_output_root_directory, indices_L_file));
% Keep only the values that correspond to the train and val sets.
number_of_trainval_images = 2975 + 500;
indices_L = indices_L(1:number_of_trainval_images);

gt_list_file = 'gtFine_trainval_filenames.txt';
fid = fopen(fullfile(cityscapes_output_root_directory, gt_list_file));
gt_file_names = textscan(fid, '%s');
fclose(fid);
gt_file_names = gt_file_names{1};
% Total number of processed images.
number_of_images = length(gt_file_names);

% Ensure that the number of file names is equal with the number of atmospheric
% light indices that have been loaded.
assert(number_of_images == number_of_trainval_images);

% Isolate images for which atmospheric light has not been selected from sky.

from_sky = false(number_of_images, 1);
sky_label = 23;
out_of_roi_label = 3;
vegetation_label = 21;
cityscapes_image_size = [1024, 2048];
out_of_roi_dims = [5, 5, 6, 5];
patch_radius = 50;

for i = 1:number_of_images
    
    % Read ground truth image.
    GT = imread(gt_file_names{i});
    
    % Atmospheric light is often selected from pixels that are near the image
    % border and are labeled as out of roi, even though they belong to the sky.
    % Take this fact into account.
    switch GT(indices_L(i))
        case sky_label
            from_sky(i) = true;
        case out_of_roi_label
            % Out of roi pixels do not necessarily correspond to sky regions.
            % Use padding of inner part of the image to fill in the out of roi
            % region with labels that correspond to adjacent pixels in the inner
            % part.
            GT_inner = GT(1 + out_of_roi_dims(1):end - out_of_roi_dims(2),...
                1 + out_of_roi_dims(3):end - out_of_roi_dims(4));
            GT_refilled = padarray(GT_inner, [5 5], 'replicate');
            GT_refilled = padarray(GT_refilled, [0 1], 'replicate', 'pre');
            if GT_refilled(indices_L(i)) == sky_label
                from_sky(i) = true;
            end
        case vegetation_label
            % Vegetation usually has very low values in the dark channel.
            % Therefore, if atmospheric light has been selected from a region
            % with vegetation, it is almost certainly the case that the
            % annotation has expanded the vegetation to parts with sky,
            % following the relevant annotation convention of Cityscapes.
            % If there are pixels nearby annotated as sky, deduce that the
            % selected pixel actually belongs to the sky.
            [row, col] = ind2sub(cityscapes_image_size, indices_L(i));
            if any(any(GT(max(row - patch_radius, 1):min(row + patch_radius,...
                    end),...
                    max(col - patch_radius, 1):min(col + patch_radius, end))...
                    == sky_label))
                from_sky(i) = true;
            end
    end
    
end

no_sky_gt_file_names = gt_file_names(~from_sky);
no_sky_image_file_names = strrep(no_sky_gt_file_names,...
    'gtFine_trainvaltest/gtFine', 'leftImg8bit');
no_sky_image_file_names = strrep(no_sky_image_file_names,...
    'gtFine_labelIds', 'leftImg8bit');
number_of_no_sky_images = length(no_sky_image_file_names);

% Write text file with list of "no sky" images.

no_sky_image_list_file =...
    'leftImg8bit_orig_trainval_no_sky_atmospheric_light_filenames.txt';
fid = fopen(fullfile(cityscapes_output_root_directory,...
    no_sky_image_list_file), 'w');
for i = 1:number_of_no_sky_images
    fprintf(fid, '%s\n', no_sky_image_file_names{i});
end
fclose(fid);
