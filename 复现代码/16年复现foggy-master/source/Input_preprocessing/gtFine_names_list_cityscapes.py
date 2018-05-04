import os
import glob

cityscapes_root = '/srv/glusterfs/daid/data/cityscape'
cityscapes_ouput_root = '/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/data/Cityscapes'

gt_fine = 'gtFine_trainvaltest/gtFine'

train = 'train'
val = 'val'

cities_train = ['aachen', 'bochum', 'bremen', 'cologne', 'darmstadt', 'dusseldorf', 'erfurt', 'hamburg', 'hanover', 'jena',
                'krefeld', 'monchengladbach', 'strasbourg', 'stuttgart', 'tubingen', 'ulm', 'weimar', 'zurich']
cities_val = ['frankfurt', 'lindau', 'munster']

gt_fine_file_names = []

gt_fine_list = os.path.join(cityscapes_ouput_root, 'gtFine_trainval_filenames.txt')

for city in cities_train:

    curr_gt_fine_dir = os.path.join(cityscapes_root, gt_fine, train, city)
    curr_gt_fine_file_names = sorted(glob.glob(curr_gt_fine_dir + '/*gtFine_labelIds.png'))
    gt_fine_file_names.extend(curr_gt_fine_file_names)


for city in cities_val:

    curr_gt_fine_dir = os.path.join(cityscapes_root, gt_fine, val, city)
    curr_gt_fine_file_names = sorted(glob.glob(curr_gt_fine_dir + '/*gtFine_labelIds.png'))
    gt_fine_file_names.extend(curr_gt_fine_file_names)


with open(gt_fine_list, 'a') as f:
    for file_name in gt_fine_file_names:
        f.write(file_name + '\n')
