clear;close all; clc;

mscnn_dehazing_code_path = '../../../MSCNN_dehazing';

addpath(genpath(mscnn_dehazing_code_path));

% This MatConvNet is compiled under Win7, you can also compile MatConvNet
% under Linux, Mac, and Windows, then run our "demo_MSCNNdehazing.m".

% Path to MatConvNet setup script.
matconvnet_setup_path = '../../../MSCNN_dehazing/matconvnet-1.0-beta23/matlab/vl_setupnn.m';

run(fullfile(fileparts(mfilename('fullpath')), matconvnet_setup_path)) ;

% if the input is very hazy, use large gamma to amend T. (0.8-1.5)

hazy_path = '../../../data/SYNTHIA_RAND_CITYSCAPES/Hazy_daytime/beta_0.03/';
% hazy_path = './testimgs/';

img = '0000228_hazy-beta_0.03-c_0.9.png'; gamma = 1;
% img = 'example-04-haze.png'; gamma = 0.8;
% img = 'newyork.png'; gamma = 1.0;
% img = 'IMG_0752.png'; gamma = 1.3;
% img = 'canyon.png'; gamma = 1.3;
imagename = [hazy_path img];

dehazedImageRGB = mscnndehazing(imagename, gamma);
