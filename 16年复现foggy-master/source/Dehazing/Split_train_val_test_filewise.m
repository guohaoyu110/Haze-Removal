% Source and destination directories.
input_image_data_directory = '/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/data/SYNTHIA_RAND_CITYSCAPES/RGB_daytime/';
output_root_directory = '/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/data/SYNTHIA_RAND_CITYSCAPES/RGB_daytime_trainvaltest/';
output_train_subdirectory = 'train/';
output_val_subdirectory = 'val/';
output_test_subdirectory = 'test/';

addpath('../Haze_simulation');
file_names = file_full_names_in_directory(input_image_data_directory);

% Size of each data split.
number_of_images = length(file_names);
training_set_size = 3000;
validation_set_size = 363;
test_set_size = number_of_images - training_set_size - validation_set_size;

% Fix random seed for reproducibility.
rng('default');

% Split the images at random into a training, a validation and a test set.
reordering = randperm(number_of_images);

training_file_names = file_names(reordering(1:training_set_size));
if exist(strcat(output_root_directory, output_train_subdirectory)) ~= 7
    mkdir(strcat(output_root_directory, output_train_subdirectory));
end
for i = 1:training_set_size
    copyfile(training_file_names{i},...
        strcat(output_root_directory, output_train_subdirectory));
end

validation_file_names =...
    file_names(reordering(training_set_size + (1:validation_set_size)));
if exist(strcat(output_root_directory, output_val_subdirectory)) ~= 7
    mkdir(strcat(output_root_directory, output_val_subdirectory));
end
for i = 1:validation_set_size
    copyfile(validation_file_names{i},...
        strcat(output_root_directory, output_val_subdirectory));
end

test_file_names = file_names(reordering(training_set_size +...
    validation_set_size + (1:test_set_size)));
if exist(strcat(output_root_directory, output_test_subdirectory)) ~= 7
    mkdir(strcat(output_root_directory, output_test_subdirectory));
end
for i = 1:test_set_size
    copyfile(test_file_names{i},...
        strcat(output_root_directory, output_test_subdirectory));
end

