function MSCNN_dehaze_and_save(input_image_data_directory, output_directory,...
    output_format)
%MSCNN_DEHAZE_AND_SAVE  Use pretrained MSCNN dehazing network provided by Ren et
%al. to dehaze all hazy images of a directory and save the results in another
%directory.
%   INPUTS:
%
%   -|input_image_data_directory|: full path to input image directory.
%   ATTENTION: Should end with a /
%
%   -|output_directory|: full path to directory where the dehazed output images
%   are saved. ATTENTION: Should end with a /
%
%   -|output_format|: string specifying the image format for the dehazed output
%   images, e.g. '.png'

mscnn_dehazing_code_path = '../../../MSCNN_dehazing';

addpath(genpath(mscnn_dehazing_code_path));

% Path to MatConvNet setup script.
matconvnet_setup_path = '../../../MSCNN_dehazing/matconvnet-1.0-beta23/matlab/vl_setupnn.m';

% Set up MatConvNet used by MSCNN dehazing network.
run(fullfile(fileparts(mfilename('fullpath')), matconvnet_setup_path)) ;

% Use identity to avoid gamma correction.
gamma = 1;

addpath('../Haze_simulation');
input_image_file_names = file_full_names_in_directory(input_image_data_directory);
number_of_images = length(input_image_file_names);

% Create output directory where dehazed images will be saved, if it does not
% already exist.
if exist(output_directory) ~= 7
    mkdir(output_directory);
end

% Main loop to predict and save dehazed images.
for i = 1:number_of_images
    
    % Predict and close the figure that is created as a side-effect.
    R = mscnndehazing(input_image_file_names{i}, gamma);
    close(gcf);
    
    % Name of output image is drawn from input image.
    [~, hazy_image_name] = fileparts(input_image_file_names{i});
    dehazed_image_name = strrep(hazy_image_name, 'hazy', 'dehazed');
        
    % Build image name from base name and output format and save it.
    imwrite(R, strcat(output_directory, dehazed_image_name, output_format));
    
    % Output number of processed images and elapsed time at fixed intervals.
    if ~mod(i, 100)
        fprintf(strcat('Processed images:\t', num2str(i), '\n'));
        toc;
    end

end

end

