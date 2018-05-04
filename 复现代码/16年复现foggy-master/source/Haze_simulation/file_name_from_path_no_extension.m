function file_name_no_extension =...
    file_name_from_path_no_extension(full_path_to_file)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

split_full_path_to_file = strsplit(full_path_to_file, '/');
file_name_with_extension = split_full_path_to_file{end};
split_file_name_with_extension = strsplit(file_name_with_extension, '.');
file_name_no_extension = strjoin(split_file_name_with_extension(1:(end - 1)),...
    '.');

end

