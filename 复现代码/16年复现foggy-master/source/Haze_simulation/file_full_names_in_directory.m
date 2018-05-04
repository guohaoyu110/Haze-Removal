function file_names = file_full_names_in_directory(input_directory)
%FILE_FULL_NAMES_OF_DIRECTORY  Generate full paths to all files of a directory.
%   INPUTS:
%
%   -|input_directory|: full path to input directory. ATTENTION: Should end with
%   a /
%
%   OUTPUTS:
%
%   -|file_names|: cell array with strings corresponding to full paths to files
%   of the input directory.

input_directory_content = dir(input_directory);
file_names = cell(1, length(input_directory_content) - 2);

% Loop over all entries.
for j = 1:length(input_directory_content)
    % Ignore if directory.
    if input_directory_content(j).isdir
        continue;
    end
    
    current_file = input_directory_content(j);
    current_file_name = current_file.name;
    file_names{j - 2} =...
        strcat(input_directory, current_file_name);
end

end

