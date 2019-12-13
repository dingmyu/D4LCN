import torch.nn as nn
import torch.nn.functional as F
import sys

# stop python from writing so much bytecode
sys.dont_write_bytecode = True

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.rpn_util import *


class RPN_3D_loss(nn.Module):

    def __init__(self, conf):

        super(RPN_3D_loss, self).__init__()

        self.num_classes = len(conf.lbls) + 1
        self.num_anchors = conf.anchors.shape[0]
        self.anchors = conf.anchors
        self.bbox_means = conf.bbox_means
        self.bbox_stds = conf.bbox_stds
        self.feat_stride = conf.feat_stride
        self.fg_fraction = conf.fg_fraction
        self.box_samples = conf.box_samples
        self.ign_thresh = conf.ign_thresh
        self.nms_thres = conf.nms_thres
        self.fg_thresh = conf.fg_thresh
        self.bg_thresh_lo = conf.bg_thresh_lo
        self.bg_thresh_hi = conf.bg_thresh_hi
        self.best_thresh = conf.best_thresh
        self.hard_negatives = conf.hard_negatives
        self.focal_loss = conf.focal_loss

        self.crop_size = conf.crop_size

        self.cls_2d_lambda = conf.cls_2d_lambda
        self.iou_2d_lambda = conf.iou_2d_lambda
        self.bbox_2d_lambda = conf.bbox_2d_lambda
        self.bbox_3d_lambda = conf.bbox_3d_lambda
        self.bbox_3d_proj_lambda = conf.bbox_3d_proj_lambda

        self.lbls = conf.lbls
        self.ilbls = conf.ilbls

        self.min_gt_vis = conf.min_gt_vis
        self.min_gt_h = conf.min_gt_h
        self.max_gt_h = conf.max_gt_h

        self.use_corner = conf.use_corner
        self.corner_in_3d = conf.corner_in_3d
        self.use_hill_loss = conf.use_hill_loss


    def forward(self, cls, prob, bbox_2d, bbox_3d, imobjs, feat_size, bbox_vertices=None, corners_3d=None):

        stats = []
        loss = torch.tensor(0).type(torch.cuda.FloatTensor)

        FG_ENC = 1000
        BG_ENC = 2000

        IGN_FLAG = 3000

        batch_size = cls.shape[0]

        prob_detach = prob.cpu().detach().numpy()

        # cls : [B x (W x H) x (Class_num * Anchor_num)] 144
        bbox_x = bbox_2d[:, :, 0]  # [B x (W x H) x Anchor_num] 36
        bbox_y = bbox_2d[:, :, 1]
        bbox_w = bbox_2d[:, :, 2]
        bbox_h = bbox_2d[:, :, 3]

        bbox_x3d = bbox_3d[:, :, 0]  # 3d_proj center
        bbox_y3d = bbox_3d[:, :, 1]
        bbox_z3d = bbox_3d[:, :, 2]  # depth
        bbox_w3d = bbox_3d[:, :, 3]
        bbox_h3d = bbox_3d[:, :, 4]
        bbox_l3d = bbox_3d[:, :, 5]
        bbox_ry3d = bbox_3d[:, :, 6]

        bbox_x3d_proj = torch.zeros(bbox_x3d.shape)
        bbox_y3d_proj = torch.zeros(bbox_x3d.shape)
        bbox_z3d_proj = torch.zeros(bbox_x3d.shape)

        labels = np.zeros(cls.shape[0:2])  # B x (W x H)
        labels_weight = np.zeros(cls.shape[0:2])

        labels_scores = np.zeros(cls.shape[0:2])

        bbox_x_tar = np.zeros(cls.shape[0:2])
        bbox_y_tar = np.zeros(cls.shape[0:2])
        bbox_w_tar = np.zeros(cls.shape[0:2])
        bbox_h_tar = np.zeros(cls.shape[0:2])

        bbox_x3d_tar = np.zeros(cls.shape[0:2])
        bbox_y3d_tar = np.zeros(cls.shape[0:2])
        bbox_z3d_tar = np.zeros(cls.shape[0:2])
        bbox_w3d_tar = np.zeros(cls.shape[0:2])
        bbox_h3d_tar = np.zeros(cls.shape[0:2])
        bbox_l3d_tar = np.zeros(cls.shape[0:2])
        bbox_ry3d_tar = np.zeros(cls.shape[0:2])

        bbox_x3d_proj_tar = np.zeros(cls.shape[0:2])
        bbox_y3d_proj_tar = np.zeros(cls.shape[0:2])
        bbox_z3d_proj_tar = np.zeros(cls.shape[0:2])

        bbox_weights = np.zeros(cls.shape[0:2])

        ious_2d = torch.zeros(cls.shape[0:2])
        ious_3d = torch.zeros(cls.shape[0:2])
        coords_abs_z = torch.zeros(cls.shape[0:2])
        coords_abs_ry = torch.zeros(cls.shape[0:2])

        # get all rois
        rois = locate_anchors(self.anchors, feat_size, self.feat_stride, convert_tensor=True)
        # print('rois:', rois.shape)  # (w * h * anchor_num) * 5, the fifth is the index 0-35
        # [ 1.4635e+03,  3.1150e+02,  2.0395e+03,  6.9550e+02,  3.5000e+01]
        # print('anchors:', self.anchors.shape)  # 36 * 9
        rois = rois.type(torch.cuda.FloatTensor)

        # de-mean std
        bbox_x3d_dn = bbox_x3d * self.bbox_stds[:, 4][0] + self.bbox_means[:, 4][0]
        bbox_y3d_dn = bbox_y3d * self.bbox_stds[:, 5][0] + self.bbox_means[:, 5][0]
        bbox_z3d_dn = bbox_z3d * self.bbox_stds[:, 6][0] + self.bbox_means[:, 6][0]
        bbox_w3d_dn = bbox_w3d * self.bbox_stds[:, 7][0] + self.bbox_means[:, 7][0]
        bbox_h3d_dn = bbox_h3d * self.bbox_stds[:, 8][0] + self.bbox_means[:, 8][0]
        bbox_l3d_dn = bbox_l3d * self.bbox_stds[:, 9][0] + self.bbox_means[:, 9][0]
        bbox_ry3d_dn = bbox_ry3d * self.bbox_stds[:, 10][0] + self.bbox_means[:, 10][0]

        src_anchors = self.anchors[rois[:, 4].type(torch.cuda.LongTensor).cpu(), :]
        src_anchors = torch.tensor(src_anchors, requires_grad=False).type(torch.cuda.FloatTensor)
        if len(src_anchors.shape) == 1: src_anchors = src_anchors.unsqueeze(0)

        # compute 3d transform
        widths = rois[:, 2] - rois[:, 0] + 1.0
        heights = rois[:, 3] - rois[:, 1] + 1.0
        ctr_x = rois[:, 0] + 0.5 * widths
        ctr_y = rois[:, 1] + 0.5 * heights

        # de-normalization
        bbox_x3d_dn = bbox_x3d_dn * widths.unsqueeze(0) + ctr_x.unsqueeze(0)
        bbox_y3d_dn = bbox_y3d_dn * heights.unsqueeze(0) + ctr_y.unsqueeze(0)
        bbox_z3d_dn = src_anchors[:, 4].unsqueeze(0) + bbox_z3d_dn
        bbox_w3d_dn = torch.exp(bbox_w3d_dn) * src_anchors[:, 5].unsqueeze(0)
        bbox_h3d_dn = torch.exp(bbox_h3d_dn) * src_anchors[:, 6].unsqueeze(0)
        bbox_l3d_dn = torch.exp(bbox_l3d_dn) * src_anchors[:, 7].unsqueeze(0)
        bbox_ry3d_dn = src_anchors[:, 8].unsqueeze(0) + bbox_ry3d_dn

        if self.use_hill_loss:
            hill_coords_2d = np.zeros(cls.shape[0:2] + (4,))  # (4, 126720, 4)
            hill_p2 = torch.zeros(torch.Size((cls.shape[0], cls.shape[1], 4, 4)))
            hill_3d = torch.zeros(torch.Size((cls.shape[0], cls.shape[1], 7)))

        for bind in range(0, batch_size):

            imobj = imobjs[bind]
            gts = imobj.gts

            p2_inv = torch.from_numpy(imobj.p2_inv).type(torch.cuda.FloatTensor)
            p2 = torch.from_numpy(imobj.p2).type(torch.cuda.FloatTensor)

            # filter gts
            igns, rmvs = determine_ignores(gts, self.lbls, self.ilbls, self.min_gt_vis, self.min_gt_h)

            # accumulate boxes, Convert from [x,y,w,h] to [x1, y1, x2, y2]
            gts_all = bbXYWH2Coords(np.array([gt.bbox_full for gt in gts]))
            # [cx, cy, cz3d_2d, w3d, h3d, l3d, alpha, cx3d, cy3d, cz3d, rotY]
            gts_3d = np.array([gt.bbox_3d for gt in gts])

            if not ((rmvs == False) & (igns == False)).any():
                continue

            # filter out irrelevant cls, and ignore cls
            gts_val = gts_all[(rmvs == False) & (igns == False), :]
            gts_ign = gts_all[(rmvs == False) & (igns == True), :]
            gts_3d = gts_3d[(rmvs == False) & (igns == False), :]

            # accumulate labels
            box_lbls = np.array([gt.cls for gt in gts])
            box_lbls = box_lbls[(rmvs == False) & (igns == False)]
            box_lbls = np.array([clsName2Ind(self.lbls, cls) for cls in box_lbls])

            if gts_val.shape[0] > 0 or gts_ign.shape[0] > 0:

                rois = rois.cpu()

                # bbox regression
                if self.use_corner:
                    gts_vertices = np.array([gt.vertices for gt in imobj.gts])
                    gts_vertices = gts_vertices[(rmvs == False) & (igns == False), :]
                    gts_corners_3d = np.array([gt.corners_3d for gt in imobj.gts])
                    gts_corners_3d = gts_corners_3d[(rmvs == False) & (igns == False), :]
                    transforms, ols, raw_gt = compute_targets(gts_val, gts_ign, box_lbls, rois.numpy(), self.fg_thresh,
                                                              self.ign_thresh, self.bg_thresh_lo, self.bg_thresh_hi,
                                                              self.best_thresh, anchors=self.anchors, gts_3d=gts_3d,
                                                              gts_vertices=gts_vertices, gts_corners_3d=gts_corners_3d,
                                                              tracker=rois[:, 4].numpy())

                else:
                    transforms, ols, raw_gt = compute_targets(gts_val, gts_ign, box_lbls, rois.numpy(), self.fg_thresh,
                                                      self.ign_thresh, self.bg_thresh_lo, self.bg_thresh_hi,
                                                      self.best_thresh, anchors=self.anchors,  gts_3d=gts_3d,
                                                      tracker=rois[:, 4].numpy())

                if self.use_hill_loss:
                    hill_deltas_2d = transforms[:, 0:4]
                    hill_coords_2d[bind, :, :] = bbox_transform_inv(rois, torch.from_numpy(hill_deltas_2d), means=self.bbox_means[0, :], stds=self.bbox_stds[0, :]).cpu().numpy() / imobj['scale_factor']

                    hill_p2[bind, :, :, :] = p2

                    hill_x3d = bbox_x3d_dn[bind].unsqueeze(0) / imobj['scale_factor']
                    hill_y3d = bbox_y3d_dn[bind].unsqueeze(0) / imobj['scale_factor']
                    hill_z3d = bbox_z3d_dn[bind].unsqueeze(0)
                    hill_w3d = bbox_w3d_dn[bind]
                    hill_h3d = bbox_h3d_dn[bind]
                    hill_l3d = bbox_l3d_dn[bind]
                    hill_ry3d = bbox_ry3d_dn[bind]
                    hill_coord3d = p2_inv.mm(torch.cat((hill_x3d * hill_z3d, hill_y3d * hill_z3d, hill_z3d, torch.ones_like(hill_x3d)), dim=0))  # # (4, 126720) # 36 * 110 * 32
                    hill_cx3d = hill_coord3d[0]
                    hill_cy3d = hill_coord3d[1]
                    hill_cz3d = hill_coord3d[2]
                    hill_ry3d = convertAlpha2Rot_torch(hill_ry3d, hill_cz3d, hill_cx3d)

                    hill_3d_all = torch.cat((hill_cx3d.unsqueeze(1), hill_cy3d.unsqueeze(1), hill_cz3d.unsqueeze(1), hill_w3d.unsqueeze(1), hill_h3d.unsqueeze(1), hill_l3d.unsqueeze(1), hill_ry3d.unsqueeze(1)), dim=1)
                    hill_3d[bind, :, :] = hill_3d_all

                # normalize 2d
                transforms[:, 0:4] -= self.bbox_means[:, 0:4]
                transforms[:, 0:4] /= self.bbox_stds[:, 0:4]

                # normalize 3d
                transforms[:, 5:12] -= self.bbox_means[:, 4:11]
                transforms[:, 5:12] /= self.bbox_stds[:, 4:11]

                if self.use_corner:
                    transforms[:, 16:32] -= self.bbox_means[:, 11:27]
                    transforms[:, 16:32] /= self.bbox_stds[:, 11:27]
                    transforms[:, 32:40] -= self.bbox_means[:, 27:35]
                    transforms[:, 32:40] /= self.bbox_stds[:, 27:35]
                if self.corner_in_3d:
                    transforms[:, 40:56] -= self.bbox_means[:, 35:51]
                    transforms[:, 40:56] /= self.bbox_stds[:, 35:51]
                    transforms[:, 12:14] -= self.bbox_means[:, 51:53]
                    transforms[:, 12:14] /= self.bbox_stds[:, 51:53]

                labels_fg = transforms[:, 4] > 0
                labels_bg = transforms[:, 4] < 0
                labels_ign = transforms[:, 4] == 0

                fg_inds = np.flatnonzero(labels_fg)
                bg_inds = np.flatnonzero(labels_bg)
                ign_inds = np.flatnonzero(labels_ign)

                transforms = torch.from_numpy(transforms)#.cuda()

                labels[bind, fg_inds] = transforms[fg_inds, 4]
                labels[bind, ign_inds] = IGN_FLAG
                labels[bind, bg_inds] = 0

                # GT
                bbox_x_tar[bind, :] = transforms[:, 0]
                bbox_y_tar[bind, :] = transforms[:, 1]
                bbox_w_tar[bind, :] = transforms[:, 2]
                bbox_h_tar[bind, :] = transforms[:, 3]

                bbox_x3d_tar[bind, :] = transforms[:, 5]
                bbox_y3d_tar[bind, :] = transforms[:, 6]
                bbox_z3d_tar[bind, :] = transforms[:, 7]
                bbox_w3d_tar[bind, :] = transforms[:, 8]
                bbox_h3d_tar[bind, :] = transforms[:, 9]
                bbox_l3d_tar[bind, :] = transforms[:, 10]
                bbox_ry3d_tar[bind, :] = transforms[:, 11]

                bbox_x3d_proj_tar[bind, :] = raw_gt[:, 12]
                bbox_y3d_proj_tar[bind, :] = raw_gt[:, 13]
                bbox_z3d_proj_tar[bind, :] = raw_gt[:, 14]

                if self.use_corner:
                    bbox_vertices_depth_tar = np.zeros(cls.shape[0:2] + (24,))
                    bbox_vertices_depth_tar[bind, :, :] = transforms[:, 16:40]
                if self.corner_in_3d:
                    bbox_3d_corners_tar = np.zeros(cls.shape[0:2] + (18,))
                    bbox_3d_corners_tar[bind, :, :16] = transforms[:, 40:56]
                    bbox_3d_corners_tar[bind, :, 16:] = transforms[:, 12:14]
                # ----------------------------------------
                # box sampling
                # ----------------------------------------

                if self.box_samples == np.inf:
                    fg_num = len(fg_inds)
                    bg_num = len(bg_inds)

                else:
                    fg_num = min(round(rois.shape[0]*self.box_samples * self.fg_fraction), len(fg_inds))
                    bg_num = min(round(rois.shape[0]*self.box_samples - fg_num), len(bg_inds))

                if self.hard_negatives:

                    if fg_num > 0 and fg_num != fg_inds.shape[0]:
                        scores = prob_detach[bind, fg_inds, labels[bind, fg_inds].astype(int)]
                        fg_score_ascend = (scores).argsort()
                        fg_inds = fg_inds[fg_score_ascend]
                        fg_inds = fg_inds[0:fg_num]

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        scores = prob_detach[bind, bg_inds, labels[bind, bg_inds].astype(int)]
                        bg_score_ascend = (scores).argsort()
                        bg_inds = bg_inds[bg_score_ascend]
                        bg_inds = bg_inds[0:bg_num]

                else:

                    if fg_num > 0 and fg_num != fg_inds.shape[0]:
                        fg_inds = np.random.choice(fg_inds, fg_num, replace=False)

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        bg_inds = np.random.choice(bg_inds, bg_num, replace=False)

                labels_weight[bind, bg_inds] = BG_ENC  # TODO: set label-weight
                labels_weight[bind, fg_inds] = FG_ENC
                bbox_weights[bind, fg_inds] = 1

                # ----------------------------------------
                # compute IoU stats
                # ----------------------------------------

                if fg_num > 0:

                    # compile deltas pred
                    deltas_2d = torch.cat((bbox_x[bind, :, np.newaxis ], bbox_y[bind, :, np.newaxis],
                                           bbox_w[bind, :, np.newaxis], bbox_h[bind, :, np.newaxis]), dim=1)

                    # compile deltas targets
                    deltas_2d_tar = np.concatenate((bbox_x_tar[bind, :, np.newaxis], bbox_y_tar[bind, :, np.newaxis],
                                                    bbox_w_tar[bind, :, np.newaxis], bbox_h_tar[bind, :, np.newaxis]),
                                                   axis=1)

                    # move to gpu
                    deltas_2d_tar = torch.tensor(deltas_2d_tar, requires_grad=False).type(torch.cuda.FloatTensor)

                    means = self.bbox_means[0, :]
                    stds = self.bbox_stds[0, :]

                    rois = rois.cuda()

                    coords_2d = bbox_transform_inv(rois, deltas_2d, means=means, stds=stds)  # convert to x1, x2, y1, y2
                    coords_2d_tar = bbox_transform_inv(rois, deltas_2d_tar, means=means, stds=stds)

                    # cal IOU
                    ious_2d[bind, fg_inds] = iou(coords_2d[fg_inds, :], coords_2d_tar[fg_inds, :], mode='list')

                    bbox_x3d_dn_fg = bbox_x3d_dn[bind, fg_inds]
                    bbox_y3d_dn_fg = bbox_y3d_dn[bind, fg_inds]

                    src_anchors = self.anchors[rois[fg_inds, 4].type(torch.cuda.LongTensor).cpu(), :]
                    src_anchors = torch.tensor(src_anchors, requires_grad=False).type(torch.cuda.FloatTensor)
                    if len(src_anchors.shape) == 1: src_anchors = src_anchors.unsqueeze(0)

                    # Prediction
                    bbox_x3d_dn_fg = bbox_x3d_dn[bind, fg_inds]
                    bbox_y3d_dn_fg = bbox_y3d_dn[bind, fg_inds]
                    bbox_z3d_dn_fg = bbox_z3d_dn[bind, fg_inds]
                    bbox_w3d_dn_fg = bbox_w3d_dn[bind, fg_inds]
                    bbox_h3d_dn_fg = bbox_h3d_dn[bind, fg_inds]
                    bbox_l3d_dn_fg = bbox_l3d_dn[bind, fg_inds]
                    bbox_ry3d_dn_fg = bbox_ry3d_dn[bind, fg_inds]

                    # re-scale all 2D back to original
                    bbox_x3d_dn_fg /= imobj['scale_factor']
                    bbox_y3d_dn_fg /= imobj['scale_factor']

                    coords_2d = torch.cat((bbox_x3d_dn_fg[np.newaxis,:] * bbox_z3d_dn_fg[np.newaxis,:], bbox_y3d_dn_fg[np.newaxis,:] * bbox_z3d_dn_fg[np.newaxis,:], bbox_z3d_dn_fg[np.newaxis,:]), dim=0)
                    coords_2d = torch.cat((coords_2d, torch.ones([1, coords_2d.shape[1]])), dim=0)

                    coords_3d = torch.mm(p2_inv, coords_2d)
                    # project center to 3d

                    bbox_x3d_proj[bind, fg_inds] = coords_3d[0, :]
                    bbox_y3d_proj[bind, fg_inds] = coords_3d[1, :]
                    bbox_z3d_proj[bind, fg_inds] = coords_3d[2, :]

                    # absolute targets
                    bbox_z3d_dn_tar = bbox_z3d_tar[bind, fg_inds] * self.bbox_stds[:, 6][0] + self.bbox_means[:, 6][0]
                    bbox_z3d_dn_tar = torch.tensor(bbox_z3d_dn_tar, requires_grad=False).type(torch.cuda.FloatTensor)
                    bbox_z3d_dn_tar = src_anchors[:, 4] + bbox_z3d_dn_tar

                    bbox_ry3d_dn_tar = bbox_ry3d_tar[bind, fg_inds] * self.bbox_stds[:, 10][0] + self.bbox_means[:, 10][0]
                    bbox_ry3d_dn_tar = torch.tensor(bbox_ry3d_dn_tar, requires_grad=False).type(torch.cuda.FloatTensor)
                    bbox_ry3d_dn_tar = src_anchors[:, 8] + bbox_ry3d_dn_tar

                    coords_abs_z[bind, fg_inds] = torch.abs(bbox_z3d_dn_tar - bbox_z3d_dn_fg)
                    coords_abs_ry[bind, fg_inds] = torch.abs(bbox_ry3d_dn_tar - bbox_ry3d_dn_fg)

            else:  # No GT.

                bg_inds = np.arange(0, rois.shape[0])

                if self.box_samples == np.inf: bg_num = len(bg_inds)
                else: bg_num = min(round(self.box_samples * (1 - self.fg_fraction)), len(bg_inds))

                if self.hard_negatives:

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        # Use probability prediction and sort
                        scores = prob_detach[bind, bg_inds, labels[bind, bg_inds].astype(int)]
                        bg_score_ascend = (scores).argsort()
                        bg_inds = bg_inds[bg_score_ascend]
                        bg_inds = bg_inds[0:bg_num]

                else:

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        bg_inds = np.random.choice(bg_inds, bg_num, replace=False)


                labels[bind, :] = 0
                labels_weight[bind, bg_inds] = BG_ENC


            # grab label predictions (for weighing purposes)
            active = labels[bind, :] != IGN_FLAG
            labels_scores[bind, active] = prob_detach[bind, active, labels[bind, active].astype(int)]

        # ----------------------------------------
        # useful statistics
        # ----------------------------------------

        fg_inds_all = np.flatnonzero((labels > 0) & (labels != IGN_FLAG))
        bg_inds_all = np.flatnonzero((labels == 0) & (labels != IGN_FLAG))

        fg_inds_unravel = np.unravel_index(fg_inds_all, prob_detach.shape[0:2])  # get position
        bg_inds_unravel = np.unravel_index(bg_inds_all, prob_detach.shape[0:2])

        cls_pred = cls.argmax(dim=2).cpu().detach().numpy()

        # class prediction acc
        if self.cls_2d_lambda and len(fg_inds_all) > 0:
            acc_fg = np.mean(cls_pred[fg_inds_unravel] == labels[fg_inds_unravel])
            stats.append({'name': 'fg', 'val': acc_fg, 'format': '{:0.2f}', 'group': 'acc'})

        if self.cls_2d_lambda and len(bg_inds_all) > 0:
            acc_bg = np.mean(cls_pred[bg_inds_unravel] == labels[bg_inds_unravel])
            stats.append({'name': 'bg', 'val': acc_bg, 'format': '{:0.2f}', 'group': 'acc'})

        # ----------------------------------------
        # box weighting
        # ----------------------------------------

        fg_inds = np.flatnonzero(labels_weight == FG_ENC)
        bg_inds = np.flatnonzero(labels_weight == BG_ENC)
        active_inds = np.concatenate((fg_inds, bg_inds), axis=0)

        fg_num = len(fg_inds)
        bg_num = len(bg_inds)

        labels_weight[...] = 0.0
        box_samples = fg_num + bg_num

        fg_inds_unravel = np.unravel_index(fg_inds, labels_weight.shape)
        bg_inds_unravel = np.unravel_index(bg_inds, labels_weight.shape)
        active_inds_unravel = np.unravel_index(active_inds, labels_weight.shape)

        labels_weight[active_inds_unravel] = 1.0

        if self.fg_fraction is not None:

            if fg_num > 0:

                fg_weight = (self.fg_fraction /(1 - self.fg_fraction)) * (bg_num / fg_num)
                labels_weight[fg_inds_unravel] = fg_weight  # large for foreground
                labels_weight[bg_inds_unravel] = 1.0

            else:
                labels_weight[bg_inds_unravel] = 1.0

        # different method of doing hard negative mining
        # use the scores to normalize the importance of each sample
        # hence, encourages the network to get all "correct" rather than
        # becoming more correct at a decision it is already good at
        # this method is equivelent to the focal loss with additional mean scaling
        if self.focal_loss:

            weights_sum = 0

            # re-weight bg
            if bg_num > 0:
                bg_scores = labels_scores[bg_inds_unravel]  # prob_detach
                bg_weights = (1 - bg_scores) ** self.focal_loss
                weights_sum += np.sum(bg_weights)
                labels_weight[bg_inds_unravel] *= bg_weights

            # re-weight fg
            if fg_num > 0:
                fg_scores = labels_scores[fg_inds_unravel]
                fg_weights = (1 - fg_scores) ** self.focal_loss
                weights_sum += np.sum(fg_weights)
                labels_weight[fg_inds_unravel] *= fg_weights


        # ----------------------------------------
        # classification loss
        # ----------------------------------------
        labels = torch.tensor(labels, requires_grad=False)
        labels = labels.view(-1).type(torch.cuda.LongTensor)

        labels_weight = torch.tensor(labels_weight, requires_grad=False)
        labels_weight = labels_weight.view(-1).type(torch.cuda.FloatTensor)

        cls = cls.view(-1, cls.shape[2])

        if self.cls_2d_lambda:

            # cls loss
            active = labels_weight > 0

            if np.any(active.cpu().numpy()):

                loss_cls = F.cross_entropy(cls[active, :], labels[active], reduction='none', ignore_index=IGN_FLAG)
                loss_cls = (loss_cls * labels_weight[active])

                # simple gradient clipping
                loss_cls = loss_cls.clamp(min=0, max=2000)

                # take mean and scale lambda
                loss_cls = loss_cls.mean()
                loss_cls *= self.cls_2d_lambda

                loss += loss_cls

                stats.append({'name': 'cls', 'val': loss_cls, 'format': '{:0.4f}', 'group': 'loss'})

        # ----------------------------------------
        # bbox regression loss
        # ----------------------------------------

        if np.sum(bbox_weights) > 0:

            bbox_weights = torch.tensor(bbox_weights, requires_grad=False).type(torch.cuda.FloatTensor).view(-1)

            active = bbox_weights > 0

            if self.bbox_2d_lambda:

                # bbox loss 2d
                bbox_x_tar = torch.tensor(bbox_x_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                bbox_y_tar = torch.tensor(bbox_y_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                bbox_w_tar = torch.tensor(bbox_w_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                bbox_h_tar = torch.tensor(bbox_h_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)

                bbox_x = bbox_x[:, :].unsqueeze(2).view(-1)
                bbox_y = bbox_y[:, :].unsqueeze(2).view(-1)
                bbox_w = bbox_w[:, :].unsqueeze(2).view(-1)
                bbox_h = bbox_h[:, :].unsqueeze(2).view(-1)

                loss_bbox_x = F.smooth_l1_loss(bbox_x[active], bbox_x_tar[active], reduction='none')
                loss_bbox_y = F.smooth_l1_loss(bbox_y[active], bbox_y_tar[active], reduction='none')
                loss_bbox_w = F.smooth_l1_loss(bbox_w[active], bbox_w_tar[active], reduction='none')
                loss_bbox_h = F.smooth_l1_loss(bbox_h[active], bbox_h_tar[active], reduction='none')

                loss_bbox_x = (loss_bbox_x * bbox_weights[active]).mean()
                loss_bbox_y = (loss_bbox_y * bbox_weights[active]).mean()
                loss_bbox_w = (loss_bbox_w * bbox_weights[active]).mean()
                loss_bbox_h = (loss_bbox_h * bbox_weights[active]).mean()

                bbox_2d_loss = (loss_bbox_x + loss_bbox_y + loss_bbox_w + loss_bbox_h)
                bbox_2d_loss *= self.bbox_2d_lambda

                loss += bbox_2d_loss
                stats.append({'name': 'bbox_2d', 'val': bbox_2d_loss, 'format': '{:0.4f}', 'group': 'loss'})

            if self.use_hill_loss:
                hill_loss = 0
                # print(hill_3d.view(-1, 7).size(), active.size())
                hill_coords_2d = torch.tensor(hill_coords_2d, requires_grad=False).type(torch.FloatTensor).cuda().view(-1, 4)[active]
                hill_p2 = hill_p2.view(-1, 4, 4)[active]
                hill_3d = hill_3d.view(-1, 7)[active]
                for index in range(hill_3d.size()[0]):
                    p2 = hill_p2[index]
                    c3d = hill_3d[index]
                    R = torch.zeros(torch.Size((3, 3)))
                    R[0, 0] += torch.cos(c3d[6])
                    R[0, 2] += torch.sin(c3d[6])
                    R[2, 0] -= torch.sin(c3d[6])
                    R[2, 2] += torch.cos(c3d[6])
                    R[1, 1] += 1
                    # print(R)
                    corners = torch.zeros(torch.Size((3, 8)))
                    corners[0, 1:5] += c3d[5]/2
                    corners[0, [0, 5, 6, 7]] -= c3d[5]/2
                    corners[1, [2, 3, 6, 7]] += c3d[4]/2
                    corners[1, [0, 1, 4, 5]] -= c3d[4]/2
                    corners[2, 3:7] += c3d[3]/2
                    corners[2, [0, 1, 2, 7]] -= c3d[3]/2
                    corners = R.mm(corners)
                    corners[0, :] += c3d[0]
                    corners[1, :] += c3d[1]
                    corners[2, :] += c3d[2]
                    # print(corners)
                    corners = torch.cat((corners, torch.ones(torch.Size((1, 8)))), dim=0)
                    corners_2d = p2.mm(corners)
                    corners_2d = corners_2d / corners_2d[2]
                    x_new = torch.sum(torch.softmax(-corners_2d[0] * 100, dim=0) * corners_2d[0])
                    y_new = torch.sum(torch.softmax(-corners_2d[1] * 100, dim=0) * corners_2d[1])
                    x2_new = torch.sum(torch.softmax(corners_2d[0] * 100, dim=0) * corners_2d[0])
                    y2_new = torch.sum(torch.softmax(corners_2d[1] * 100, dim=0) * corners_2d[1])
                    # print(x_new, y_new, x2_new, y2_new)
                    # print(hill_coords_2d[index])

                    hill_loss += (F.smooth_l1_loss(x_new, hill_coords_2d[index][0], reduction='none') + \
                                 F.smooth_l1_loss(y_new, hill_coords_2d[index][1], reduction='none') +\
                                 F.smooth_l1_loss(x2_new, hill_coords_2d[index][2], reduction='none') +\
                                 F.smooth_l1_loss(y2_new, hill_coords_2d[index][3], reduction='none'))
                hill_loss = hill_loss / hill_3d.size()[0] / 1000  # 2.5 pixel, 0.01 loss
                loss += hill_loss * self.use_hill_loss
                stats.append({'name': 'hill_loss', 'val': hill_loss, 'format': '{:0.4f}', 'group': 'loss'})

            if self.use_corner:
                bbox_vertices_depth_tar = torch.tensor(bbox_vertices_depth_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1, 24)
                bbox_vertices = bbox_vertices.view(-1, 24)
                loss_vertices = F.smooth_l1_loss(bbox_vertices[active], bbox_vertices_depth_tar[active], reduction='none')
                loss_vertices = (loss_vertices * bbox_weights[active].view(-1, 1)).mean()
                loss += loss_vertices * self.use_corner
                stats.append({'name': 'loss_vertices', 'val': loss_vertices, 'format': '{:0.4f}', 'group': 'loss'})

            if self.corner_in_3d:
                bbox_3d_corners_tar = torch.tensor(bbox_3d_corners_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1, 18)
                corners_3d = corners_3d.view(-1, 18)
                loss_corners_3d = F.smooth_l1_loss(corners_3d[active], bbox_3d_corners_tar[active], reduction='none')
                loss_corners_3d = (loss_corners_3d * bbox_weights[active].view(-1, 1)).mean()
                loss += loss_corners_3d * self.corner_in_3d
                stats.append({'name': 'loss_corners_3d', 'val': loss_corners_3d, 'format': '{:0.4f}', 'group': 'loss'})

            if self.bbox_3d_lambda:

                # bbox loss 3d
                bbox_x3d_tar = torch.tensor(bbox_x3d_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                bbox_y3d_tar = torch.tensor(bbox_y3d_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                bbox_z3d_tar = torch.tensor(bbox_z3d_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                bbox_w3d_tar = torch.tensor(bbox_w3d_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                bbox_h3d_tar = torch.tensor(bbox_h3d_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                bbox_l3d_tar = torch.tensor(bbox_l3d_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                bbox_ry3d_tar = torch.tensor(bbox_ry3d_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)

                bbox_x3d = bbox_x3d[:, :].view(-1)
                bbox_y3d = bbox_y3d[:, :].view(-1)
                bbox_z3d = bbox_z3d[:, :].view(-1)
                bbox_w3d = bbox_w3d[:, :].view(-1)
                bbox_h3d = bbox_h3d[:, :].view(-1)
                bbox_l3d = bbox_l3d[:, :].view(-1)
                bbox_ry3d = bbox_ry3d[:, :].view(-1)

                loss_bbox_x3d = F.smooth_l1_loss(bbox_x3d[active], bbox_x3d_tar[active], reduction='none')
                loss_bbox_y3d = F.smooth_l1_loss(bbox_y3d[active], bbox_y3d_tar[active], reduction='none')
                loss_bbox_z3d = F.smooth_l1_loss(bbox_z3d[active], bbox_z3d_tar[active], reduction='none')
                loss_bbox_w3d = F.smooth_l1_loss(bbox_w3d[active], bbox_w3d_tar[active], reduction='none')
                loss_bbox_h3d = F.smooth_l1_loss(bbox_h3d[active], bbox_h3d_tar[active], reduction='none')
                loss_bbox_l3d = F.smooth_l1_loss(bbox_l3d[active], bbox_l3d_tar[active], reduction='none')
                loss_bbox_ry3d = F.smooth_l1_loss(bbox_ry3d[active], bbox_ry3d_tar[active], reduction='none')

                loss_bbox_x3d = (loss_bbox_x3d * bbox_weights[active]).mean()
                loss_bbox_y3d = (loss_bbox_y3d * bbox_weights[active]).mean()
                loss_bbox_z3d = (loss_bbox_z3d * bbox_weights[active]).mean()
                loss_bbox_w3d = (loss_bbox_w3d * bbox_weights[active]).mean()
                loss_bbox_h3d = (loss_bbox_h3d * bbox_weights[active]).mean()
                loss_bbox_l3d = (loss_bbox_l3d * bbox_weights[active]).mean()
                loss_bbox_ry3d = (loss_bbox_ry3d * bbox_weights[active]).mean()

                bbox_3d_loss = (loss_bbox_x3d + loss_bbox_y3d + loss_bbox_z3d)
                bbox_3d_loss += (loss_bbox_w3d + loss_bbox_h3d + loss_bbox_l3d + loss_bbox_ry3d)

                bbox_3d_loss *= self.bbox_3d_lambda

                loss += bbox_3d_loss
                stats.append({'name': 'bbox_3d', 'val': bbox_3d_loss, 'format': '{:0.4f}', 'group': 'loss'})

            if self.bbox_3d_proj_lambda:

                # bbox loss 3d
                bbox_x3d_proj_tar = torch.tensor(bbox_x3d_proj_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                bbox_y3d_proj_tar = torch.tensor(bbox_y3d_proj_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                bbox_z3d_proj_tar = torch.tensor(bbox_z3d_proj_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)

                bbox_x3d_proj = bbox_x3d_proj[:, :].view(-1)
                bbox_y3d_proj = bbox_y3d_proj[:, :].view(-1)
                bbox_z3d_proj = bbox_z3d_proj[:, :].view(-1)

                loss_bbox_x3d_proj = F.smooth_l1_loss(bbox_x3d_proj[active], bbox_x3d_proj_tar[active], reduction='none')
                loss_bbox_y3d_proj = F.smooth_l1_loss(bbox_y3d_proj[active], bbox_y3d_proj_tar[active], reduction='none')
                loss_bbox_z3d_proj = F.smooth_l1_loss(bbox_z3d_proj[active], bbox_z3d_proj_tar[active], reduction='none')

                loss_bbox_x3d_proj = (loss_bbox_x3d_proj * bbox_weights[active]).mean()
                loss_bbox_y3d_proj = (loss_bbox_y3d_proj * bbox_weights[active]).mean()
                loss_bbox_z3d_proj = (loss_bbox_z3d_proj * bbox_weights[active]).mean()

                bbox_3d_proj_loss = (loss_bbox_x3d_proj + loss_bbox_y3d_proj + loss_bbox_z3d_proj)

                bbox_3d_proj_loss *= self.bbox_3d_proj_lambda

                loss += bbox_3d_proj_loss
                stats.append({'name': 'bbox_3d_proj', 'val': bbox_3d_proj_loss, 'format': '{:0.4f}', 'group': 'loss'})

            coords_abs_z = coords_abs_z.view(-1)
            stats.append({'name': 'z', 'val': coords_abs_z[active].mean(), 'format': '{:0.2f}', 'group': 'misc'})

            coords_abs_ry = coords_abs_ry.view(-1)
            stats.append({'name': 'ry', 'val': coords_abs_ry[active].mean(), 'format': '{:0.2f}', 'group': 'misc'})

            ious_2d = ious_2d.view(-1)
            stats.append({'name': 'iou', 'val': ious_2d[active].mean(), 'format': '{:0.2f}', 'group': 'acc'})

            # use a 2d IoU based log loss
            if self.iou_2d_lambda:
                iou_2d_loss = -torch.log(ious_2d[active])
                iou_2d_loss = (iou_2d_loss * bbox_weights[active])
                iou_2d_loss = iou_2d_loss.mean()

                iou_2d_loss *= self.iou_2d_lambda
                loss += iou_2d_loss

                stats.append({'name': 'iou', 'val': iou_2d_loss, 'format': '{:0.4f}', 'group': 'loss'})


        return loss, stats
