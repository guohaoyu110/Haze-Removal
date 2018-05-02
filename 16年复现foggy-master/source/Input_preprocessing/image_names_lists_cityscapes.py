import os
import glob

cityscapes_root = '/srv/glusterfs/daid/data/cityscape'
cityscapes_camera_root = '/srv/glusterfs/csakarid/data/Cityscapes/camera_trainvaltest'
cityscapes_ouput_root = '/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/data/Cityscapes'

left_images = 'leftImg8bit'
right_images = 'rightImg8bit'
disparity = 'disparity'
camera = 'camera'

train = 'train'
val = 'val'
test = 'test'

cities_train = ['aachen', 'bochum', 'bremen', 'cologne', 'darmstadt', 'dusseldorf', 'erfurt', 'hamburg', 'hanover', 'jena',
                'krefeld', 'monchengladbach', 'strasbourg', 'stuttgart', 'tubingen', 'ulm', 'weimar', 'zurich']
cities_val = ['frankfurt', 'lindau', 'munster']
cities_test = ['berlin', 'bielefeld', 'bonn', 'leverkusen', 'mainz', 'munich']

left_images_file_names = []
right_images_file_names = []
disparity_file_names = []
camera_file_names = []

left_images_list = os.path.join(cityscapes_ouput_root, 'leftImg8bit_orig_trainvaltest_filenames.txt')
right_images_list = os.path.join(cityscapes_ouput_root, 'rightImg8bit_orig_trainvaltest_filenames.txt')
disparity_list = os.path.join(cityscapes_ouput_root, 'disparity_orig_trainvaltest_filenames.txt')
camera_list = os.path.join(cityscapes_ouput_root, 'camera_orig_trainvaltest_filenames.txt')

for city in cities_train:

    curr_left_images_dir = os.path.join(cityscapes_root, left_images, train, city)
    curr_left_images_file_names = sorted(glob.glob(curr_left_images_dir + '/*.png'))
    left_images_file_names.extend(curr_left_images_file_names)

    curr_right_images_dir = os.path.join(cityscapes_root, right_images, train, city)
    curr_right_images_file_names = sorted(glob.glob(curr_right_images_dir + '/*.png'))
    right_images_file_names.extend(curr_right_images_file_names)

    curr_disparity_dir = os.path.join(cityscapes_root, disparity, train, city)
    curr_disparity_file_names = sorted(glob.glob(curr_disparity_dir + '/*.png'))
    disparity_file_names.extend(curr_disparity_file_names)

    curr_camera_dir = os.path.join(cityscapes_camera_root, camera, train, city)
    curr_camera_file_names = sorted(glob.glob(curr_camera_dir + '/*.json'))
    camera_file_names.extend(curr_camera_file_names)


for city in cities_val:

    curr_left_images_dir = os.path.join(cityscapes_root, left_images, val, city)
    curr_left_images_file_names = sorted(glob.glob(curr_left_images_dir + '/*.png'))
    left_images_file_names.extend(curr_left_images_file_names)

    curr_right_images_dir = os.path.join(cityscapes_root, right_images, val, city)
    curr_right_images_file_names = sorted(glob.glob(curr_right_images_dir + '/*.png'))
    right_images_file_names.extend(curr_right_images_file_names)

    curr_disparity_dir = os.path.join(cityscapes_root, disparity, val, city)
    curr_disparity_file_names = sorted(glob.glob(curr_disparity_dir + '/*.png'))
    disparity_file_names.extend(curr_disparity_file_names)

    curr_camera_dir = os.path.join(cityscapes_camera_root, camera, val, city)
    curr_camera_file_names = sorted(glob.glob(curr_camera_dir + '/*.json'))
    camera_file_names.extend(curr_camera_file_names)


for city in cities_test:

    curr_left_images_dir = os.path.join(cityscapes_root, left_images, test, city)
    curr_left_images_file_names = sorted(glob.glob(curr_left_images_dir + '/*.png'))
    left_images_file_names.extend(curr_left_images_file_names)

    curr_right_images_dir = os.path.join(cityscapes_root, right_images, test, city)
    curr_right_images_file_names = sorted(glob.glob(curr_right_images_dir + '/*.png'))
    right_images_file_names.extend(curr_right_images_file_names)

    curr_disparity_dir = os.path.join(cityscapes_root, disparity, test, city)
    curr_disparity_file_names = sorted(glob.glob(curr_disparity_dir + '/*.png'))
    disparity_file_names.extend(curr_disparity_file_names)

    curr_camera_dir = os.path.join(cityscapes_camera_root, camera, test, city)
    curr_camera_file_names = sorted(glob.glob(curr_camera_dir + '/*.json'))
    camera_file_names.extend(curr_camera_file_names)


assert len(left_images_file_names) == len(right_images_file_names)
assert len(left_images_file_names) == len(disparity_file_names)
assert len(left_images_file_names) == len(camera_file_names)

with open(left_images_list, 'a') as f:
    for file_name in left_images_file_names:
        f.write(file_name + '\n')

with open(right_images_list, 'a') as f:
    for file_name in right_images_file_names:
        f.write(file_name + '\n')

with open(disparity_list, 'a') as f:
    for file_name in disparity_file_names:
        f.write(file_name + '\n')

with open(camera_list, 'a') as f:
    for file_name in camera_file_names:
        f.write(file_name + '\n')





