import os
import glob
from shutil import copyfile

cityscapes_root_left_images = '/srv/glusterfs/daid/data/cityscape/leftImg8bit'
cityscapes_root_gt = '/srv/glusterfs/daid/data/cityscape/gtFine_trainvaltest/gtFine'
cityscapes_output_root = '/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/data/Cityscapes'

output_orig = 'leftImg8bit_trainvaltest_refined_clean'
output_gt = 'gtFine_trainvaltest_refined'
input_foggy = 'leftImg8bit_trainvaltest_full_beta_0.01_foggy'
output_foggy = 'leftImg8bit_trainvaltest_refined_beta_0.01_foggy'
input_transmission = 'leftImg8bit_trainvaltest_full_beta_0.01_transmission'
output_transmission = 'leftImg8bit_trainvaltest_refined_beta_0.01_transmission'

suffix_orig = 'leftImg8bit.png'
suffix_gt_color = 'gtFine_color.png'
suffix_gt_instances = 'gtFine_instanceIds.png'
suffix_gt_labels = 'gtFine_labelIds.png'
suffix_gt_labels_train = 'gtFine_labelTrainIds.png'
suffix_gt_polygons = 'gtFine_polygons.json'
suffix_input_foggy = 'leftImg8bit_hazy-beta_0.01.png'
suffix_output_foggy = 'leftImg8bit_foggy-beta_0.01.png'
suffix_transmission = 'leftImg8bit_transmission-beta_0.01.png'

good_images_orig_list = os.path.join(cityscapes_output_root, 'leftImg8bit_orig_trainval_refined_filenames.txt')

with open(good_images_orig_list, 'r') as f:
    good_images_orig_filenames = f.read().splitlines()

# Original leftImg8bit images.
good_images_orig_output_filenames = [fn.replace(cityscapes_root_left_images, os.path.join(cityscapes_output_root, output_orig)) for fn in good_images_orig_filenames]

for dirpath, dirs, files in os.walk(cityscapes_root_left_images):
    for directory in dirs:
        os.makedirs(os.path.join(dirpath, directory).replace(cityscapes_root_left_images, os.path.join(cityscapes_output_root, output_orig)))

# gtFine files.
good_images_gt_help_filenames = [fn.replace(cityscapes_root_left_images, cityscapes_root_gt) for fn in good_images_orig_filenames]

good_images_gt_color_filenames = [fn.replace(suffix_orig, suffix_gt_color) for fn in good_images_gt_help_filenames]
good_images_gt_color_output_filenames = [fn.replace(cityscapes_root_gt, os.path.join(cityscapes_output_root, output_gt)) for fn in good_images_gt_color_filenames]

good_images_gt_instances_filenames = [fn.replace(suffix_orig, suffix_gt_instances) for fn in good_images_gt_help_filenames]
good_images_gt_instances_output_filenames = [fn.replace(cityscapes_root_gt, os.path.join(cityscapes_output_root, output_gt)) for fn in good_images_gt_instances_filenames]

good_images_gt_labels_filenames = [fn.replace(suffix_orig, suffix_gt_labels) for fn in good_images_gt_help_filenames]
good_images_gt_labels_output_filenames = [fn.replace(cityscapes_root_gt, os.path.join(cityscapes_output_root, output_gt)) for fn in good_images_gt_labels_filenames]

good_images_gt_labels_train_filenames = [fn.replace(suffix_orig, suffix_gt_labels_train) for fn in good_images_gt_help_filenames]
good_images_gt_labels_train_output_filenames = [fn.replace(cityscapes_root_gt, os.path.join(cityscapes_output_root, output_gt)) for fn in good_images_gt_labels_train_filenames]

good_images_gt_polygons_filenames = [fn.replace(suffix_orig, suffix_gt_polygons) for fn in good_images_gt_help_filenames]
good_images_gt_polygons_output_filenames = [fn.replace(cityscapes_root_gt, os.path.join(cityscapes_output_root, output_gt)) for fn in good_images_gt_polygons_filenames]

for dirpath, dirs, files in os.walk(cityscapes_root_gt):
    for directory in dirs:
        os.makedirs(os.path.join(dirpath, directory).replace(cityscapes_root_gt, os.path.join(cityscapes_output_root, output_gt)))

# Foggy leftImg8bit images.
good_images_foggy_filenames = [fn.replace(cityscapes_root_left_images, os.path.join(cityscapes_output_root, input_foggy)) for fn in good_images_orig_filenames]
good_images_foggy_filenames = [fn.replace(suffix_orig, suffix_input_foggy) for fn in good_images_foggy_filenames]
good_images_foggy_output_filenames = [fn.replace(input_foggy, output_foggy) for fn in good_images_foggy_filenames]
good_images_foggy_output_filenames = [fn.replace(suffix_input_foggy, suffix_output_foggy) for fn in good_images_foggy_output_filenames]

for dirpath, dirs, files in os.walk(os.path.join(cityscapes_output_root, input_foggy)):
    for directory in dirs:
        os.makedirs(os.path.join(dirpath, directory).replace(input_foggy, output_foggy))

# Transmission for foggy leftImg8bit images.
good_images_transmission_filenames = [fn.replace(cityscapes_root_left_images, os.path.join(cityscapes_output_root, input_transmission)) for fn in good_images_orig_filenames]
good_images_transmission_filenames = [fn.replace(suffix_orig, suffix_transmission) for fn in good_images_transmission_filenames]
good_images_transmission_output_filenames = [fn.replace(input_transmission, output_transmission) for fn in good_images_transmission_filenames]

for dirpath, dirs, files in os.walk(os.path.join(cityscapes_output_root, input_transmission)):
    for directory in dirs:
        os.makedirs(os.path.join(dirpath, directory).replace(input_transmission, output_transmission))

# Copy all sets of image files to output directories.
for i in xrange(len(good_images_orig_filenames)):
    copyfile(good_images_foggy_filenames[i], good_images_foggy_output_filenames[i])
    copyfile(good_images_transmission_filenames[i], good_images_transmission_output_filenames[i])
    copyfile(good_images_orig_filenames[i], good_images_orig_output_filenames[i])
    copyfile(good_images_gt_color_filenames[i], good_images_gt_color_output_filenames[i])
    copyfile(good_images_gt_instances_filenames[i], good_images_gt_instances_output_filenames[i])
    copyfile(good_images_gt_labels_filenames[i], good_images_gt_labels_output_filenames[i])
    copyfile(good_images_gt_labels_train_filenames[i], good_images_gt_labels_train_output_filenames[i])
    copyfile(good_images_gt_polygons_filenames[i], good_images_gt_polygons_output_filenames[i])





