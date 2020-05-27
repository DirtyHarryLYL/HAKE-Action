# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# --------------------------------------------------------
# 
# Modified from Multitask Network Cascade
# Copyright (c) 2017, Haoshu Fang
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import caffe
import numpy as np
import yaml


DEBUG = False
PRINT_GRADIENT = 1


class ProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        #self._feat_stride = layer_params['feat_stride']
        #self._anchors = generate_anchors()
        #self._num_anchors = self._anchors.shape[0]
        self._use_clip = layer_params.get('use_clip', 0)
        #self._clip_denominator = float(layer_params.get('clip_base', 256))
        #self._clip_thresh = 1.0 / self._clip_denominator
        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        self._top_name_map = {}
        assert(bottom[0].data.shape[0] == bottom[1].data.shape[0])
        top[0].reshape(bottom[0].data.shape[0], 5)
        self._top_name_map['rois'] = 0
        # For MNC, we force the output proposals will also be used to train RPN
        # this is achieved by passing proposal_index to anchor_target_layer
        # if str(self.phase) == 'TRAIN':
        #     if cfg.TRAIN.MIX_INDEX:
        #         top[1].reshape(1, 1)
        #         self._top_name_map['proposal_index'] = 1

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted transform deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)


        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        orig_bbox = bottom[0].data
        bbox_deltas = bottom[1].data
        im_info = bottom[2].data
        self._ind = np.arange(0,bottom[0].data.shape[0])
        # 1. Generate proposals from transform deltas and shifted anchors
        # height, width = scores.shape[-2:]
        # self._height = height
        # self._width = width
        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        # A = self._num_anchors
        # K = shifts.shape[0]
        # anchors = self._anchors.reshape((1, A, 4)) + \
        #           shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        # anchors = anchors.reshape((K * A, 4))

        _, keep = clip_boxes(orig_bbox, (im_info.shape[2],im_info.shape[3]))
        self._orig_bbox_index_before_clip = keep

        # Transpose and reshape predicted transform transformations to get them
        # into the same order as the anchors:
        #
        # transform deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        # bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        # scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via transform transformations

        proposals = bbox_transform_inv(orig_bbox, bbox_deltas)
        self._orig_box = orig_bbox

        # 2. clip predicted boxes to image
        proposals, keep = clip_boxes(proposals, (im_info.shape[2],im_info.shape[3]))
        # Record the cooresponding index before and after clip
        # This step doesn't need unmap
        # We need it to decide whether do back propagation
        self._proposal_index_before_clip = keep

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        # keep = filter_small_boxes(proposals, min_size * im_info[2])
        # proposals = proposals[keep, :]
        # scores = scores[keep]
        # self._ind_after_filter = keep

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        # order = scores.ravel().argsort()[::-1]

        # if pre_nms_topN > 0:
        #     order = order[:pre_nms_topN]
        # proposals = proposals[order, :]
        # scores = scores[order]
        # self._ind_after_sort = order
        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        # keep = nms(np.hstack((proposals, scores)), nms_thresh)

        # if post_nms_topN > 0:
        #     keep = keep[:post_nms_topN]
        # proposals = proposals[keep, :]

        # scores = scores[keep]
        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        # batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        # proposals = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        self._proposal_index = keep

        blobs = {
            'rois': proposals
        }

        # if str(self.phase) == 'TRAIN':
        #     if cfg.TRAIN.MIX_INDEX:
        #         all_rois_index = self._ind_after_filter[self._ind_after_sort[self._proposal_index]].reshape(1, len(keep))
        #         blobs['proposal_index'] = all_rois_index

        # Copy data to forward to top layer
        for blob_name, blob in blobs.iteritems():
            top[self._top_name_map[blob_name]].reshape(*blob.shape)
            top[self._top_name_map[blob_name]].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):

        if propagate_down[1]:
            bottom[1].diff.fill(0.0)

            # first count only non-zero top gradient to accelerate computing
            #top_non_zero_ind = np.unique(np.where(abs(top[0].diff[:, :]) > 0)[0])
            #proposal_index = np.asarray(self._proposal_index)
            # unmap indexes to the original scale
            #unmap_val = self._ind_after_filter[self._ind_after_sort[proposal_index[top_non_zero_ind]]]

            # not back propagate gradient if proposals/anchors are out of image boundary
            # this is a 0/1 mask so we just multiply them when calculating bottom gradient
            weight_out_proposal = np.in1d(self._ind,self._proposal_index_before_clip)
            weight_out_anchor = np.in1d(self._ind,self._orig_bbox_index_before_clip)

            # unmap_val are arranged as (H * W * A) as stated in forward comment
            # with A as the fastest dimension (which is different from caffe)
            c = np.arange(0,bottom[0].data.shape[0])
            # w = (unmap_val / self._num_anchors) % self._width
            # h = (unmap_val / self._num_anchors / self._width) % self._height

            # width and height should be in feature map scale
            anchor_w = (self._orig_box[c, 3] - self._orig_box[c, 1])
            anchor_h = (self._orig_box[c, 4] - self._orig_box[c, 2])
            dfdx1 = top[0].diff[c, 1]
            dfdy1 = top[0].diff[c, 2]
            dfdx2 = top[0].diff[c, 3]
            dfdy2 = top[0].diff[c, 4]

            dfdxc = dfdx1 + dfdx2
            dfdyc = dfdy1 + dfdy2
            dfdw = 0.5 * (dfdx2 - dfdx1)
            dfdh = 0.5 * (dfdy2 - dfdy1)

            bottom[1].diff[c, 0] = \
                dfdxc * anchor_w * weight_out_proposal * weight_out_anchor
            bottom[1].diff[c, 1] = \
                dfdyc * anchor_h * weight_out_proposal * weight_out_anchor
            bottom[1].diff[c, 2] = \
                dfdw * np.exp(bottom[1].data[c, 2]) * anchor_w * weight_out_proposal * weight_out_anchor
            bottom[1].diff[c, 3] = \
                dfdh * np.exp(bottom[1].data[c, 3]) * anchor_h * weight_out_proposal * weight_out_anchor

            # if use gradient clip, constraint gradient inside [-thresh, thresh]
            if self._use_clip:
                bottom[1].diff[c, 0] = np.minimum(np.maximum(
                    bottom[1].diff[c, 0], -self._clip_thresh), self._clip_thresh)
                bottom[1].diff[c, 1] = np.minimum(np.maximum(
                    bottom[1].diff[c, 1], -self._clip_thresh), self._clip_thresh)
                bottom[1].diff[c, 2] = np.minimum(np.maximum(
                    bottom[1].diff[c, 2], -self._clip_thresh), self._clip_thresh)
                bottom[1].diff[c, 3] = np.minimum(np.maximum(
                    bottom[1].diff[c, 3], -self._clip_thresh), self._clip_thresh)


def bbox_transform_inv(boxes, deltas):
    """
    invert bounding box transform
    apply delta on anchors to get transformed proposals
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    #print deltas.shape
    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    pred_ctr_x = dx * widths[:] + ctr_x[:]
    pred_ctr_y = dy * heights[:] + ctr_y[:]
    pred_w = np.exp(dw) * widths[:]
    pred_h = np.exp(dh) * heights[:]

    pred_boxes = np.zeros(boxes.shape, dtype=boxes.dtype)
    pred_boxes[:, 0] = boxes[:, 0]
    # x1
    pred_boxes[:, 1] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 2] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 3] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    Clip boxes inside image boundaries
    """
    x1 = boxes[:, 1]
    y1 = boxes[:, 2]
    x2 = boxes[:, 3]
    y2 = boxes[:, 4]
    keep = np.where((x1 >= 0) & (x2 <= im_shape[1] - 1) & (y1 >= 0) & (y2 <= im_shape[0] - 1))[0]
    clipped_boxes = np.zeros(boxes.shape, dtype=boxes.dtype)
    clipped_boxes[:, 0] = boxes[:, 0]
    # x1 >= 0
    clipped_boxes[:, 1] = np.maximum(np.minimum(boxes[:, 1], im_shape[1] - 1), 0)
    # y1 >= 0
    clipped_boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    clipped_boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    clipped_boxes[:, 4] = np.maximum(np.minimum(boxes[:, 4], im_shape[0] - 1), 0)
    return clipped_boxes, keep
