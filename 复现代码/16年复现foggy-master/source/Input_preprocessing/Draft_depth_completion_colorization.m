close all;
clear;

rgb_input = imread('/srv/glusterfs/daid/data/cityscape/leftImg8bit/train/dusseldorf/dusseldorf_000020_000019_leftImg8bit.png');
% rgb_input = imread('/srv/glusterfs/daid/data/cityscape/leftImg8bit/train/hamburg/hamburg_000000_046078_leftImg8bit.png');
rgb_image = im2double(rgb_input);

disparity_input = imread('/srv/glusterfs/daid/data/cityscape/disparity/train/dusseldorf/dusseldorf_000020_000019_disparity.png');
% disparity_input = imread('/srv/glusterfs/daid/data/cityscape/disparity/train/hamburg/hamburg_000000_046078_disparity.png');

% Flag for downscaling the images.
downscale = 1;

% Optional downscaling.
if downscale
    % Factor of downscaling.
    scale = 0.5;
    
    % Downscale the disparity map using the nearest neighbor method. This method
    % ensures that the value indicating invalid disparities (0) is not modified
    % to a different value.
    disparity_input = imresize(disparity_input, scale, 'nearest');
    
    % Downscale the RGB image using the default bicubic method.
    rgb_image = imresize(rgb_image, scale, 'bicubic');
end

disparity_is_invalid = disparity_input == 0;
disparity_double = (double(disparity_input) - 1) / 256;
disparity_is_zero = disparity_double == 0;

depth = zeros(size(disparity_double));
depth(~disparity_is_invalid & ~disparity_is_zero) = 1 ./ disparity_double(~disparity_is_invalid & ~disparity_is_zero);
depth(disparity_is_zero) = max(depth(:));
% depth(disparity_is_zero) = Inf;

%% Run colorization code provided in the toolbox of NYU Depth v2

tic;
addpath('/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/toolbox_nyu_depth_v2');
filled_depth = fill_depth_colorization(rgb_image, depth);
toc;

%% Plotting raw depth versus denoised + completed depth

figure;
subplot(2, 1, 1);
imshow(depth);
subplot(2, 1, 2);
imshow(filled_depth, [min(depth(:)), max(depth(:))]);

figure;
subplot(2, 1, 1);
imshow(disparity_is_invalid);
subplot(2, 1, 2);
imshow(~disparity_is_invalid | abs(depth - filled_depth) > 0.01);

%% Upscale result to original size if necessary for subsequent haze simulation

if downscale
    % Upscale using bilinear interpolation method (and not the bicubic) to
    % remain inside the range of input values.
    filled_depth = imresize(filled_depth, 1 / scale, 'bilinear');
    
    % Restore the RGB at its original size as well.
    rgb_image = im2double(rgb_input);
end

%% Haze simulation

beta = 8;
c = 0.9;
L = repmat(c, 1, 1, 3);
addpath('../Haze_simulation');
t = transmission_exponential(filled_depth, beta);
R = rgb_image;
I = haze_linear(R, t, L);

%% Show haze synthesis result against original clean image

figure;
imshow(R);
figure;
imshow(I);
% imwrite(I, strcat('/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/results/Cityscapes/hamburg_000000_046078_leftImg8bit_hazy-beta_',...
%     num2str(beta), '-c_', num2str(c), '.png'));