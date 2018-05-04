function cityscapes_label_IDs_from_synthia_IDs(input_directory,...
    output_directory, output_format)

% Define mapping between SYNTHIA labels and Cityscapes IDs. The format that is
% used for Cityscapes IDs is uint8, as in the original Cityscapes dataset.
cityscapes_ids_for_synthia_labels = uint8([0, 23, 11, 7, 8, 13, 21, 17, 26,...
    20, 24, 33, 32, 4, 5, 19, 22, 25, 27, 28, 31, 12, 7]);

% Find all SYNTHIA ground truth images.
file_names = file_full_names_in_directory(input_directory);
number_of_images = length(file_names);

% Create output directory where modified ground truth images will be saved, if
% it does not already exist.
if exist(output_directory) ~= 7
    mkdir(output_directory);
end

% Loop over all input ground truth images to compute and save the modified
% ground truth images in Cityscapes format.
for i = 1:number_of_images
    % Transform the labels.
    gt_labels = imread(file_names{i});
    gt_labels = gt_labels(:, :, 1);
    gt_cityscapes_labels = cityscapes_ids_for_synthia_labels(gt_labels + 1);
    
    % Save the modified ground truth image in the output directory.
    gt_image_name = file_name_from_path_no_extension(file_names{i});
    imwrite(gt_cityscapes_labels, strcat(output_directory, gt_image_name,...
        output_format));
end

end