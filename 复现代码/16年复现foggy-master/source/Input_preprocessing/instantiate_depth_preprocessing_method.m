function depth_preprocessing_method_parameters =...
    instantiate_depth_preprocessing_method(depth_preprocessing_method,...
    depth_file_names, camera_parameters_directory_cityscapes)
%INSTANTIATE_DEPTH_PREPROCESSING_METHOD  Assign values to parameters of the
%specified method for depth preprocessing.
%
%   INPUTS:
%
%   -|depth_preprocessing_method|: handle of the function which is specified as
%   the depth preprocessing method.
%
%   -|depth_file_names|: cell array containing full paths to files with depth
%   input.
%
%   -|camera_parameters_directory_cityscapes|: full path to directory containing
%   files with camera parameters for Cityscapes images. Used by
%   |depth_in_meters_cityscapes_with_invalid_parts|. ATTENTION: Should end with
%   a /
%
%   OUTPUTS:
%
%   -|depth_preprocessing_method_parameters|: extra parameter for specified
%   method for depth preprocessing, apart from basic depth input.

switch func2str(depth_preprocessing_method)
    case 'depth_in_meters_synthia'
        % Dummy parameters.
        depth_preprocessing_method_parameters =...
            cell(1, length(depth_file_names));
        
    case 'depth_in_meters_cityscapes_with_invalid_parts'
        % Cell array with full paths to camera parameters files.
        depth_preprocessing_method_parameters =...
            file_full_names_in_directory(...
            camera_parameters_directory_cityscapes);
        assert(length(depth_preprocessing_method_parameters) ==...
            length(depth_file_names));
end


end

