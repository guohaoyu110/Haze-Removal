close all;
clear;

addpath('../../Haze_simulation');
addpath('..');

% Define arrays with input parameters for |dark_channel_prior_dehaze_and_save|.
betas = 0.01:0.01:0.1;
betas_str = cell(size(betas));
for i = 1:length(betas)
    betas_str{i} = num2str(betas(i));
end
input_image_data_directories =...
    strcat({'/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/data/SYNTHIA_RAND_CITYSCAPES/Hazy_daytime_trainvaltest/test/beta_'},...
    betas_str, {'/'});
output_directories =...
    strcat({'/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/data/SYNTHIA_RAND_CITYSCAPES/Dehazed_dark_channel_prior_daytime_trainvaltest/test/beta_'},...
    betas_str, {'/'});
output_format = '.png';

% Main loop for dehazing of all hazy versions of SYNTHIA daytime test set with
% dark channel prior method.
tic;
for i = 1:length(betas_str)
    dark_channel_prior_dehaze_and_save(input_image_data_directories{i},...
        output_directories{i}, output_format);
    fprintf(strcat('Finished dehazing version of test set with beta = ',...
        betas_str{i}, '.\n\n'));
    toc;
end