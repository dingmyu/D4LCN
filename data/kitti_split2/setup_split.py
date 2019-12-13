from importlib import import_module
from getopt import getopt
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
import pprint
import sys
import os
import cv2
import math
import shutil
import re
from scipy.io import loadmat

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.util import *

split = 'kitti_split2'

# base paths
base_data = os.path.join(os.getcwd(), 'data')

kitti_raw = dict()
kitti_raw['cal'] = os.path.join(base_data, 'kitti', 'training', 'calib')
kitti_raw['ims'] = os.path.join(base_data, 'kitti', 'training', 'image_2')
kitti_raw['lab'] = os.path.join(base_data, 'kitti', 'training', 'label_2')

kitti_tra = dict()
kitti_tra['cal'] = os.path.join(base_data, split, 'training', 'calib')
kitti_tra['ims'] = os.path.join(base_data, split, 'training', 'image_2')
kitti_tra['lab'] = os.path.join(base_data, split, 'training', 'label_2')

kitti_val = dict()
kitti_val['cal'] = os.path.join(base_data, split, 'validation', 'calib')
kitti_val['ims'] = os.path.join(base_data, split, 'validation', 'image_2')
kitti_val['lab'] = os.path.join(base_data, split, 'validation', 'label_2')

split_data = loadmat(os.path.join(base_data, split, 'kitti_ids_new.mat'))

# mkdirs
mkdir_if_missing(kitti_tra['cal'])
mkdir_if_missing(kitti_tra['ims'])
mkdir_if_missing(kitti_tra['lab'])
mkdir_if_missing(kitti_val['cal'])
mkdir_if_missing(kitti_val['ims'])
mkdir_if_missing(kitti_val['lab'])


print('Linking {} train'.format(split_data['ids_train'][0].shape[0]))

imind = 0

for id_num in split_data['ids_train'][0]:

    id = '{:06d}'.format(id_num)
    new_id = '{:06d}'.format(imind)

    if not os.path.exists(os.path.join(kitti_tra['cal'], str(new_id) + '.txt')):
        os.symlink(os.path.join(kitti_raw['cal'], str(id) + '.txt'), os.path.join(kitti_tra['cal'], str(new_id) + '.txt'))

    if not os.path.exists(os.path.join(kitti_tra['ims'], str(new_id) + '.png')):
        os.symlink(os.path.join(kitti_raw['ims'], str(id) + '.png'), os.path.join(kitti_tra['ims'], str(new_id) + '.png'))

    if not os.path.exists(os.path.join(kitti_tra['lab'], str(new_id) + '.txt')):
        os.symlink(os.path.join(kitti_raw['lab'], str(id) + '.txt'), os.path.join(kitti_tra['lab'], str(new_id) + '.txt'))

    imind += 1

print('Linking {} val'.format(split_data['ids_val'][0].shape[0]))

imind = 0

for id_num in split_data['ids_val'][0]:

    id = '{:06d}'.format(id_num)
    new_id = '{:06d}'.format(imind)

    if not os.path.exists(os.path.join(kitti_val['cal'], str(new_id) + '.txt')):
        os.symlink(os.path.join(kitti_raw['cal'], str(id) + '.txt'), os.path.join(kitti_val['cal'], str(new_id) + '.txt'))

    if not os.path.exists(os.path.join(kitti_val['ims'], str(new_id) + '.png')):
        os.symlink(os.path.join(kitti_raw['ims'], str(id) + '.png'), os.path.join(kitti_val['ims'], str(new_id) + '.png'))

    if not os.path.exists(os.path.join(kitti_val['lab'], str(new_id) + '.txt')):
        os.symlink(os.path.join(kitti_raw['lab'], str(id) + '.txt'), os.path.join(kitti_val['lab'], str(new_id) + '.txt'))

        imind += 1

print('Done')
