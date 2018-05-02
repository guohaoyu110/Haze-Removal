% Add path to function with inverse linear haze model.
addpath('..');

% Read input hazy image.
path_to_image = '/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/data/Foggy_Zurich/20161213_082952.jpg';
image = im2double(imread(path_to_image));
I = imresize(image, 0.5);

% Dehaze.
R = dark_channel_prior_dehaze(I);
