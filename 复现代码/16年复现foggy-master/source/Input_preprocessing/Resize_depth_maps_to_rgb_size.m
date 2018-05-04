data_parent_dir = '../../data/';
depth_data_source_directories = {'Test_134-depth_maps-raw/',...
    'Train_400-depth_maps-raw/'};
rgb_data_source_directories = {'Test134/', 'Train400Img/'};
depth_data_target_directories = {'Test_134-depth_maps/',...
    'Train_400-depth_maps/'};

% Loop over all directories with RGB and corresponding depth data.
for i = 1:length(depth_data_source_directories)
    current_depth_source_directory = strcat(data_parent_dir,...
        depth_data_source_directories{i});
    depth_source_directory_content = dir(current_depth_source_directory);
    current_rgb_source_directory = strcat(data_parent_dir,...
        rgb_data_source_directories{i});
    rgb_source_directory_content = dir(current_rgb_source_directory);
    
    % Check whether current pair of RGB and depth directories have same number
    % of files.
    assert(length(depth_source_directory_content) ==...
        length(rgb_source_directory_content));
    
    current_target_directory = strcat(data_parent_dir,...
        depth_data_target_directories{i});
    
    % Create target directory if it does not already exist.
    if exist(current_target_directory) ~= 7
        mkdir(current_target_directory);
    end
    
    % Loop over all files in each pair of directories, assuming exact
    % correspondence between RGB and depth files.
    for j = 1:length(depth_source_directory_content)
        % Ignore if directory.
        if depth_source_directory_content(j).isdir
            continue;
        end
        
        % Load *.mat file with matrix containing depth map.
        current_depth_file = depth_source_directory_content(j);
        current_depth_file_name = current_depth_file.name;
        load(strcat(current_depth_source_directory, current_depth_file_name));
        
        % Load corresponding image file (e.g. *.jpg) containing RGB data.
        current_rgb_file = rgb_source_directory_content(j);
        current_rgb_file_name = current_rgb_file.name;
        rgb_image = imread(strcat(current_rgb_source_directory,...
            current_rgb_file_name));
        
        % Get size of RGB image.
        rgb_size = size(rgb_image);
        
        % Resize depth map to match size of RGB image.
        depth_map = imresize(depth_map_raw, [rgb_size(1) rgb_size(2)],...
            'method', 'bilinear');
                
        % Save resized depth map in separate file.
        target_file_name = current_depth_file_name;
        save(strcat(current_target_directory, target_file_name),...
            'depth_map');
    end
end