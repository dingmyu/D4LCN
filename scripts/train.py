# -----------------------------------------
# python modules
# -----------------------------------------
from easydict import EasyDict as edict
from getopt import getopt
import numpy as np
import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.core import *
from lib.imdb_util import *
from lib.loss.rpn_3d import *


def main(argv):

    # -----------------------------------------
    # parse arguments
    # -----------------------------------------
    opts, args = getopt(argv, '', ['config=', 'restore='])

    # defaults
    conf_name = None
    restore = None

    # read opts
    for opt, arg in opts:

        if opt in ('--config'): conf_name = arg
        if opt in ('--restore'): restore = int(arg)

    # required opt
    if conf_name is None:
        raise ValueError('Please provide a configuration file name, e.g., --config=<config_name>')

    # -----------------------------------------
    # basic setup
    # -----------------------------------------

    conf = init_config(conf_name)
    paths = init_training_paths(conf_name, conf.result_dir)

    init_torch(conf.rng_seed, conf.cuda_seed)
    init_log_file(paths.logs)

    vis = init_visdom(conf.result_dir, conf.visdom_port)

    # defaults
    start_iter = 0
    tracker = edict()
    iterator = None
    has_visdom = vis is not None

    dataset = Dataset(conf, paths.data, paths.output)

    generate_anchors(conf, dataset.imdb, paths.output)
    compute_bbox_stats(conf, dataset.imdb, paths.output)

    paths.output = os.path.join(paths.output, conf.result_dir)

    # -----------------------------------------
    # store config
    # -----------------------------------------

    # store configuration
    pickle_write(os.path.join(paths.output, 'conf.pkl'), conf)

    # show configuration
    pretty = pretty_print('conf', conf)
    logging.info(pretty)


    # -----------------------------------------
    # network and loss
    # -----------------------------------------

    # training network
    rpn_net, optimizer, scheduler = init_training_model(conf, paths.output, conf_name)

    # setup loss
    criterion_det = RPN_3D_loss(conf)

    # custom pretrained network
    if 'pretrained' in conf:

        load_weights(rpn_net, conf.pretrained)

    # resume training
    if restore:
        start_iter = (restore - 1)
        resume_checkpoint(optimizer, rpn_net, paths.weights, restore)

    freeze_blacklist = None if 'freeze_blacklist' not in conf else conf.freeze_blacklist
    freeze_whitelist = None if 'freeze_whitelist' not in conf else conf.freeze_whitelist

    freeze_layers(rpn_net, freeze_blacklist, freeze_whitelist)

    optimizer.zero_grad()

    start_time = time()

    # -----------------------------------------
    # train
    # -----------------------------------------

    for iteration in range(start_iter, conf.max_iter):

        # next iteration
        iterator, images, depths, imobjs = next_iteration(dataset.loader, iterator)
        #print(images, images.size())
        #  learning rate
        adjust_lr(conf, optimizer, iteration, scheduler)

        # forward
        if conf.corner_in_3d:
            cls, prob, bbox_2d, bbox_3d, feat_size, bbox_vertices, corners_3d = rpn_net(images.cuda(), depths.cuda())
        elif conf.use_corner:
            cls, prob, bbox_2d, bbox_3d, feat_size, bbox_vertices = rpn_net(images.cuda(), depths.cuda())
        else:
            cls, prob, bbox_2d, bbox_3d, feat_size = rpn_net(images.cuda(), depths.cuda())

        feat_size = feat_size[:2]  # feat_size = [32, 110]
        # print(feat_size)
        # loss
        if conf.corner_in_3d:
            det_loss, det_stats = criterion_det(cls, prob, bbox_2d, bbox_3d, imobjs, feat_size, bbox_vertices, corners_3d)
        elif conf.use_corner:
            det_loss, det_stats = criterion_det(cls, prob, bbox_2d, bbox_3d, imobjs, feat_size, bbox_vertices)
        else:
            det_loss, det_stats = criterion_det(cls, prob, bbox_2d, bbox_3d, imobjs, feat_size)

        total_loss = det_loss
        stats = det_stats

        # backprop
        if total_loss > 0:

            total_loss.backward()

            # batch skip, simulates larger batches by skipping gradient step
            if (not 'batch_skip' in conf) or ((iteration + 1) % conf.batch_skip) == 0:
                optimizer.step()
                optimizer.zero_grad()

        # keep track of stats
        compute_stats(tracker, stats)

        # -----------------------------------------
        # display
        # -----------------------------------------
        if (iteration + 1) % conf.display == 0 and iteration > start_iter:

            # log results
            log_stats(tracker, iteration, start_time, start_iter, conf.max_iter)

            # display results
            if has_visdom:
                display_stats(vis, tracker, iteration, start_time, start_iter, conf.max_iter, conf_name, pretty)

            # reset tracker
            tracker = edict()

        # -----------------------------------------
        # test network
        # -----------------------------------------
        if (iteration + 1) % conf.snapshot_iter == 0 and iteration > start_iter:

            # store checkpoint
            save_checkpoint(optimizer, rpn_net, paths.weights, (iteration + 1))

            if conf.do_test:
                rpn_net = rpn_net.module
                # eval mode
                rpn_net.eval()

                # necessary paths
                results_path = os.path.join(paths.results, 'results_{}'.format((iteration + 1)))

                # -----------------------------------------
                # test kitti
                # -----------------------------------------
                if conf.test_protocol.lower() == 'kitti':

                    # delete and re-make
                    results_path = os.path.join(results_path, 'data')
                    mkdir_if_missing(results_path, delete_if_exist=True)

                    test_kitti_3d(conf.dataset_test, 'validation', rpn_net, conf, results_path, paths.data)

                else:
                    logging.warning('Testing protocol {} not understood.'.format(conf.test_protocol))

                # train mode
                rpn_net = torch.nn.DataParallel(rpn_net)
                rpn_net.train()

                freeze_layers(rpn_net, freeze_blacklist, freeze_whitelist)

    print(conf.result_dir)


# run from command line
if __name__ == "__main__":
    main(sys.argv[1:])
