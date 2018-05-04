data_parent_dir = '../../data/';
depth_data_source_directories = 'Test_134-depth_maps/';
rgb_data_source_directories = 'Test134/';

current_depth_source_directory = strcat(data_parent_dir,...
    depth_data_source_directories);
depth_source_directory_content = dir(current_depth_source_directory);
depth_file_names = cell(1, length(depth_source_directory_content) - 2);
current_rgb_source_directory = strcat(data_parent_dir,...
    rgb_data_source_directories);
rgb_source_directory_content = dir(current_rgb_source_directory);
input_image_file_names = cell(1, length(rgb_source_directory_content) - 2);
assert(length(depth_source_directory_content) ==...
    length(rgb_source_directory_content));

% Loop over all files.
for j = 1:length(depth_source_directory_content)
    % Ignore if directory.
    if depth_source_directory_content(j).isdir
        continue;
    end
    
    current_depth_file = depth_source_directory_content(j);
    current_depth_file_name = current_depth_file.name;
    depth_file_names{j - 2} =...
        strcat(current_depth_source_directory, current_depth_file_name);
    
    current_rgb_file = rgb_source_directory_content(j);
    current_rgb_file_name = current_rgb_file.name;
    input_image_file_names{j - 2} =...
        strcat(current_rgb_source_directory, current_rgb_file_name);
end