function [file_names_grouped, number_of_groups] =...
    group_filenames_per_chunk(file_names, chunk)

number_of_images = length(file_names);
file_names_as_numbers = zeros(1, number_of_images);

for i = 1:number_of_images
    [~, file_name] = fileparts(file_names{i});
    file_names_as_numbers(i) = str2num(file_name);
end

group_map = floor(file_names_as_numbers / chunk);

[~, group_start_inds] = unique(group_map, 'stable');

number_of_groups = length(group_start_inds);
file_names_grouped = cell(1, number_of_groups);

for i = 1:number_of_groups - 1
    file_names_grouped{i} =...
        file_names(group_start_inds(i):group_start_inds(i + 1) - 1);
end
file_names_grouped{number_of_groups} =...
    file_names(group_start_inds(number_of_groups):number_of_images);

end