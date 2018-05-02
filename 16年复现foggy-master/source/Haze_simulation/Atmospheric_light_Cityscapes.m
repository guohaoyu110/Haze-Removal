function Atmospheric_light_Cityscapes
%ATMOSPHERIC_LIGHT_CITYSCAPES  Estimate atmospheric light for all images in
%Cityscapes and save results. Estimation is performed with dark-channel-based
%methods.

cityscapes_output_root_directory = '/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/data/Cityscapes';

addpath('../Dehazing/Dark_channel_prior');

% Get all names of input images.
fid = fopen(fullfile(cityscapes_output_root_directory,...
    'leftImg8bit_orig_trainvaltest_filenames.txt'));
left_image_file_names = textscan(fid, '%s');
fclose(fid);
left_image_file_names = left_image_file_names{1};

% Total number of processed images.
number_of_images = length(left_image_file_names);

% Initialize matrix with atmospheric light values for each image.
L = zeros(number_of_images, 1, 3);

% Initialize vector with linear indices of pixels selected as atmospheric light.
indices_L = zeros(number_of_images, 1);

% Select and instantiate the function that estimates atmospheric light for each
% image.
atmospheric_light_estimation = @estimate_atmospheric_light_rf;

% Run atmospheric light estimation on input images.
for i = 1:number_of_images
    
    % Read clean image to double precision for subsequent computations.
    R = im2double(imread(left_image_file_names{i}));
    
    % Estimate atmospheric light for the clean image.
    neighborhood_size_dark_channel = 15;
    R_dark = get_dark_channel(R, neighborhood_size_dark_channel);
    [L(i, :, :), indices_L(i)] = atmospheric_light_estimation(R_dark, R);
    
end

% Save as MAT-files.
save(fullfile(cityscapes_output_root_directory,...
    'leftImg8bit_trainvaltest_atmlight.mat'), 'L');
save(fullfile(cityscapes_output_root_directory,...
    'leftImg8bit_trainvaltest_atmlight_pixelind.mat'), 'indices_L');

% Save as text files in comma-separated format.
dlmwrite(fullfile(cityscapes_output_root_directory,...
    'leftImg8bit_trainvaltest_atmlight.txt'), reshape(L, number_of_images, 3));
dlmwrite(fullfile(cityscapes_output_root_directory,...
    'leftImg8bit_trainvaltest_atmlight_pixelind.txt'),...
    indices_L, 'precision', '%7.f');

end

