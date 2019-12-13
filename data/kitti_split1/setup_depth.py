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

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.util import *

split = 'kitti_split1'

# base paths
base_data = os.path.join(os.getcwd(), 'data')

kitti_raw = dict()
kitti_raw['depth'] = os.path.join(base_data, 'kitti', 'depth_2')

kitti_tra = dict()
kitti_tra['depth'] = os.path.join(base_data, split, 'training', 'depth_2')

kitti_val = dict()
kitti_val['depth'] = os.path.join(base_data, split, 'validation', 'depth_2')

tra_file = os.path.join(base_data, split, 'train.txt')
val_file = os.path.join(base_data, split, 'val.txt')

# mkdirs
mkdir_if_missing(kitti_tra['depth'])
mkdir_if_missing(kitti_val['depth'])

print('Linking train')
text_file = open(tra_file, 'r')

imind = 0

for line in text_file:

    parsed = re.search('(\d+)', line)

    if parsed is not None:

        id = str(parsed[0])
        new_id = '{:06d}'.format(imind)

        if not os.path.exists(os.path.join(kitti_tra['depth'], str(new_id) + '.png')):
            os.symlink(os.path.join(kitti_raw['depth'], str(id) + '.png'), os.path.join(kitti_tra['depth'], str(new_id) + '.png'))

        imind += 1

text_file.close()

print('Linking val')
text_file = open(val_file, 'r')

imind = 0

for line in text_file:

    parsed = re.search('(\d+)', line)

    if parsed is not None:

        id = str(parsed[0])
        new_id = '{:06d}'.format(imind)

        if not os.path.exists(os.path.join(kitti_val['depth'], str(new_id) + '.png')):
            os.symlink(os.path.join(kitti_raw['depth'], str(id) + '.png'), os.path.join(kitti_val['depth'], str(new_id) + '.png'))

        imind += 1

text_file.close()

print('Done')
