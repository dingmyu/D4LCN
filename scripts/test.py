# -----------------------------------------
# python modules
# -----------------------------------------
from importlib import import_module
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
import sys
import numpy as np
import os

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.imdb_util import *

conf_path = 'pretrain/conf.pkl'
weights_path = 'pretrain/model_40000_pkl'

# load config
conf = edict(pickle_read(conf_path))

data_path = os.path.join(os.getcwd(), 'data')
results_path = os.path.join('output', 'pretrain', 'data')

# make directory
mkdir_if_missing(results_path, delete_if_exist=True)

# -----------------------------------------
# torch defaults
# -----------------------------------------

# defaults
init_torch(conf.rng_seed, conf.cuda_seed)

# -----------------------------------------
# setup network
# -----------------------------------------

# net
net = import_module('models.' + conf.model).build(conf)

# load weights
load_weights(net, weights_path, remove_module=True)

# switch modes for evaluation
net.eval()

print(pretty_print('conf', conf))

# -----------------------------------------
# test kitti
# -----------------------------------------

test_kitti_3d(
    conf.dataset_test,  # 'kitti_split1',  #
    'validation',  # 'testing',  # 'validation', #
    net,
    conf,
    results_path,
    data_path,
    use_log=False)
