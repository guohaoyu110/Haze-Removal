depth_data_parent_dir = '../../data/';
depth_data_source_directories = {'Test134Depth/', 'Train400Depth/'};
depth_data_target_directories = {'Test_134-depth_maps-raw/',...
    'Train_400-depth_maps-raw/'};

% Loop over all directories with depth data.
for i = 1:length(depth_data_source_directories)
    current_source_directory = strcat(depth_data_parent_dir,...
        depth_data_source_directories{i});
    source_directory_content = dir(current_source_directory);
    current_target_directory = strcat(depth_data_parent_dir,...
        depth_data_target_directories{i});
    
    % Create target directory if it does not already exist.
    if exist(current_target_directory) ~= 7
        mkdir(current_target_directory);
    end
    
    % Loop over all files in each directory.
    for j = 1:length(source_directory_content)
        % Ignore if directory.
        if source_directory_content(j).isdir
            continue;
        end
        
        % Load *.mat file with matrix containing depth map.
        current_file = source_directory_content(j);
        current_file_name = current_file.name;
        load(strcat(current_source_directory, current_file_name));
        
        % Retrieve raw depth map.
        depth_map_raw = Position3DGrid(:, :, 4);
        
        % Save raw depth map on its own in separate file.
        target_file_name = current_file_name;
        save(strcat(current_target_directory, target_file_name),...
            'depth_map_raw');
    end
end