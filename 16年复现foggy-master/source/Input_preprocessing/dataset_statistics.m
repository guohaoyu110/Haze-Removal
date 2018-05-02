function [mean_dataset, std_dataset] =...
    dataset_statistics(training_data_directory, color_space)
%DATASET_STATISTICS  Compute mean and standard deviation statistics for pixel
%values of a dataset with RGB images.
%
%   INPUTS:
%
%   -|training_data_directory|: full path to directory with training set
%   images. ATTENTION: Should end with a /
%
%   -|color_space|: string flag indicating the color space for which the
%   statistics of the dataset are computed, e.g. RGB or CIELAB.
%   OUTPUTS:
%
%   -|mean_dataset|: 1-by-3 vector with mean RGB value of all pixels in the
%   training set. Computed for RGB values in [0, 1] range.
%
%   -|std_dataset|: 1-by-3 vector with standard deviations for the 3 RGB
%   channels across all pixels in the training set. Computed for RGB values in
%   [0, 1] range.


% IMPORTANT NOTE: In this code, it is assumed that all images in the dataset
% contain the same number of pixels, and therefore most probably share the same
% dimensions.

% Create list of file names with training images.
addpath('../Haze_simulation');
training_file_names = file_full_names_in_directory(training_data_directory);
D = length(training_file_names);

% Initialize variables that hold per-image pixel statistics, used in the final
% computation.
means_per_image = zeros(D, 3);
variances_per_image = zeros(D, 3);

% Main loop over dataset images to compute per-image pixel statistics.
for i = 1:D
    
    % Read image and bring it to double precision for subsequent
    % computations.
    image = im2double(imread(training_file_names{i}));
    switch color_space
        case 'rgb'
            % Do nothing.
        case 'lab'
            image = rgb2lab(image);
    end
    image_1 = image(:, :, 1);
    image_2 = image(:, :, 2);
    image_3 = image(:, :, 3);
    
    % Mean pixel value for processed image.
    means_per_image(i, :) = mean(mean(image));
    
    % Variances of pixel values in each channel separately.
    % NOTE: Variance is computed using the unadjusted formula where the original
    % number of samples appears in the denominator, which corresponds to a
    % biased estimator. However, the total number of pixels in each image is
    % large enough for the resulting bias to be negligible.
    variances_per_image(i, 1) = var(image_1(:), 1);
    variances_per_image(i, 2) = var(image_2(:), 1);
    variances_per_image(i, 3) = var(image_3(:), 1);
    
end

% Final aggregation across the entire dataset.
mean_dataset = mean(means_per_image);

variance_of_means = var(means_per_image, 1);
mean_of_variances = mean(variances_per_image);
variance_dataset = variance_of_means + mean_of_variances;
std_dataset = sqrt(variance_dataset);

end



