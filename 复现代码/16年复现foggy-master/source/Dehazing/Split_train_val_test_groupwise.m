% Source and destination directories.
input_data_directory = '/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/data/SYNTHIA_RAND_CITYSCAPES/Depth_daytime/';
output_root_directory = '/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/data/SYNTHIA_RAND_CITYSCAPES/Depth_daytime_trainvaltest/';
output_train_subdirectory = 'train/';
output_val_subdirectory = 'val/';
output_test_subdirectory = 'test/';

addpath('../Haze_simulation');
file_names = file_full_names_in_directory(input_data_directory);

% Group file names based on the chunk they belong.
chunk = 50;
[file_names_grouped, number_of_groups] =...
    group_filenames_per_chunk(file_names, chunk);

% Size of each split, measured in image "groups" (sets of images with similar
% scenes).
training_set_size = 113;
validation_set_size = 19;
test_set_size = number_of_groups - training_set_size - validation_set_size;

% Fix random seed for reproducibility.
rng('default');

% Split the image groups at random into a training, a validation and a test set.
reordering = randperm(number_of_groups);

training_file_names = file_names_grouped(reordering(1:training_set_size));
if exist(strcat(output_root_directory, output_train_subdirectory)) ~= 7
    mkdir(strcat(output_root_directory, output_train_subdirectory));
end
for i = 1:training_set_size
    current_group = training_file_names{i};
    for j = 1:length(current_group)
        copyfile(current_group{j},...
            strcat(output_root_directory, output_train_subdirectory));
    end
end

validation_file_names =...
    file_names_grouped(reordering(training_set_size + (1:validation_set_size)));
if exist(strcat(output_root_directory, output_val_subdirectory)) ~= 7
    mkdir(strcat(output_root_directory, output_val_subdirectory));
end
for i = 1:validation_set_size
    current_group = validation_file_names{i};
    for j = 1:length(current_group)
        copyfile(current_group{j},...
            strcat(output_root_directory, output_val_subdirectory));
    end
end

test_file_names = file_names_grouped(reordering(training_set_size +...
    validation_set_size + (1:test_set_size)));
if exist(strcat(output_root_directory, output_test_subdirectory)) ~= 7
    mkdir(strcat(output_root_directory, output_test_subdirectory));
end
for i = 1:test_set_size
    current_group = test_file_names{i};
    for j = 1:length(current_group)
        copyfile(current_group{j},...
            strcat(output_root_directory, output_test_subdirectory));
    end
end

