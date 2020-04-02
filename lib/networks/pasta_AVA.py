from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.framework import ops

from ult.config import cfg
from ult.ava_loss_weight import verb80
from ult.visualization import draw_bounding_boxes_HOI

import numpy as np

def resnet_arg_scope(is_training=True,
                     weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params = {
        'is_training': False,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': ops.GraphKeys.UPDATE_OPS
    }
    with arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY),
        weights_initializer = slim.variance_scaling_initializer(),
        biases_regularizer  = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), 
        biases_initializer  = tf.constant_initializer(0.0),
        trainable           = is_training,
        activation_fn       = tf.nn.relu,
        normalizer_fn       = slim.batch_norm,
        normalizer_params   = batch_norm_params):
        with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

"""Contains definitions for the original form of Residual Networks.
    The 'v1' residual networks (ResNets) implemented in this module were proposed
    by:
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Deep Residual Learning for Image Recognition. arXiv:1512.03385
    Other variants were introduced in:
    [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Identity Mappings in Deep Residual Networks. arXiv: 1603.05027
    The networks defined in this module utilize the bottleneck building block of
    [1] with projection shortcuts only for increasing depths. They employ batch
    normalization *after* every weight layer. This is the architecture used by
    MSRA in the Imagenet and MSCOCO 2016 competition models ResNet-101 and
    ResNet-152. See [2; Fig. 1a] for a comparison between the current 'v1'
    architecture and the alternative 'v2' architecture of [2] which uses batch
    normalization *before* every weight layer in the so-called full pre-activation
    units.
    Typical use:
        from tensorflow.contrib.slim.python.slim.nets import resnet_v1
    ResNet-101 for image classification into 1000 classes:
        # inputs has shape [batch, 224, 224, 3]
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = resnet_v1.resnet_v1_101(inputs, 1000, is_training=False)
    ResNet-101 for semantic segmentation into 21 classes:
    # inputs has shape [batch, 513, 513, 3]
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = resnet_v1.resnet_v1_101(inputs,
                                                    21,
                                                    is_training=False,
                                                    global_pool=False,
                                                    output_stride=16)
"""
"""
    Helper function for creating a resnet_v1 bottleneck block.
    Args:
        scope: The scope of the block.
        base_depth: The depth of the bottleneck layer for each unit.
        num_units: The number of units in the block.
        stride: The stride of the block, implemented as a stride in the last unit.
                All other units have stride=1.
    Returns:
        A resnet_v1 bottleneck block.
""" 
"""Generator for v1 ResNet models.
    This function generates a family of ResNet v1 models. See the resnet_v1_*()
    methods for specific model instantiations, obtained by selecting different
    block instantiations that produce ResNets of various depths.
    Training for image classification on Imagenet is usually done with [224, 224]
    inputs, resulting in [7, 7] feature maps at the output of the last ResNet
    block for the ResNets defined in [1] that have nominal stride equal to 32.
    However, for dense prediction tasks we advise that one uses inputs with
    spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
    this case the feature maps at the ResNet output will have spatial shape
    [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
    and corners exactly aligned with the input image corners, which greatly
    facilitates alignment of the features to the image. Using as input [225, 225]
    images results in [8, 8] feature maps at the output of the last ResNet block.
    For dense prediction tasks, the ResNet needs to run in fully-convolutional
    (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
    have nominal stride equal to 32 and a good choice in FCN mode is to use
    output_stride=16 in order to increase the density of the computed features at
    small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.

    Args:
        inputs: A tensor of size [batch, height_in, width_in, channels].
        blocks: A list of length equal to the number of ResNet blocks. Each element
                is a resnet_utils.Block object describing the units in the block.
        num_classes: Number of predicted classes for classification tasks. If None
                    we return the features before the logit layer.
        is_training: whether batch_norm layers are in training mode.
        global_pool: If True, we perform global average pooling before computing the
                    logits. Set to True for image classification, False for dense prediction.
        output_stride: If None, then the output will be computed at the nominal
                        network stride. If output_stride is not None, it specifies the requested
                        ratio of input to output spatial resolution.
        include_root_block: If True, include the initial convolution followed by
                            max-pooling, if False excludes it.
        reuse: whether or not the network and its variables should be reused. To be
                able to reuse 'scope' must be given.
        scope: Optional variable_scope.
    Returns:
        net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
            If global_pool is False, then height_out and width_out are reduced by a
            factor of output_stride compared to the respective height_in and width_in,
            else both height_out and width_out equal one. If num_classes is None, then
            net is the output of the last ResNet block, potentially after global
            average pooling. If num_classes is not None, net contains the pre-softmax
            activations.
        end_points: A dictionary from components of the network to the corresponding
                    activation.
    Raises:
        ValueError: If the target output_stride is not valid.
"""

class ResNet50(): # 64--128--256--512--512
    def __init__(self):
        self.visualize = {}
        self.intermediate = {}
        self.predictions = {}
        self.score_summaries = {}
        self.event_summaries = {}
        self.train_summaries = []
        self.losses = {}
        self.image       = tf.placeholder(tf.float32, shape=[1, None, None, 3], name = 'image') #scene stream
        self.H_boxes     = tf.placeholder(tf.float32, shape=[None, 5], name = 'H_boxes') # Human stream
        self.P_boxes     = tf.placeholder(tf.float32, shape=[None, 10, 5], name = 'P_boxes') # PaSta stream
        self.gt_verb     = tf.placeholder(tf.float32, shape=[None, 80], name = 'gt_class_verb') # target verb
        self.H_num       = tf.placeholder(tf.int32)
        self.verb_weight = np.array(verb80, dtype='float32').reshape(1, -1)
        self.num_classes = 80 # HOI
        self.num_pasta0    = 12 # pasta0 ankle
        self.num_pasta1    = 10 # pasta1 knee
        self.num_pasta2    = 5 # pasta2 hip
        self.num_pasta3    = 31 # pasta3 hand
        self.num_pasta4    = 5 # pasta4 shoulder
        self.num_pasta5    = 13 # pasta5 head
        self.num_fc      = 1024
        self.scope       = 'resnet_v1_50'
        self.stride      = [16, ]
        self.lr          = tf.placeholder(tf.float32)
        if tf.__version__ == '1.1.0':
            self.blocks     = [resnet_utils.Block('block1', resnet_v1.bottleneck,[(256,   64, 1)] * 2 + [(256,   64, 2)]),
                               resnet_utils.Block('block2', resnet_v1.bottleneck,[(512,  128, 1)] * 3 + [(512,  128, 2)]),
                               resnet_utils.Block('block3', resnet_v1.bottleneck,[(1024, 256, 1)] * 5 + [(1024, 256, 1)]),
                               resnet_utils.Block('block4', resnet_v1.bottleneck,[(2048, 512, 1)] * 3),
                               resnet_utils.Block('block5', resnet_v1.bottleneck,[(2048, 512, 1)] * 3)]
        else: # we use tf 1.2.0 here, Resnet-50
            from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
            self.blocks = [resnet_v1_block('block1', base_depth=64,  num_units=3, stride=2), # a resnet_v1 bottleneck block
                           resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                           resnet_v1_block('block3', base_depth=256, num_units=6, stride=1), # feature former
                           resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
                           resnet_v1_block('block5', base_depth=512, num_units=3, stride=1)]
    # Args:
    #   scope: The scope of the block.
    #   base_depth: The depth of the bottleneck layer for each unit.
    #   num_units: The number of units in the block.
    #   stride: The stride of the block, implemented as a stride in the last unit.
    #     All other units have stride=1.

    def build_base(self):
        with tf.variable_scope(self.scope, self.scope):
            net = resnet_utils.conv2d_same(self.image, 64, 7, stride=2, scope='conv1') # conv2d + subsample
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')
        return net

    # Number of fixed blocks during training, by default 
    # __C.RESNET.FIXED_BLOCKS = 1

    # feature extractor
    def image_to_head(self, is_training):
        with slim.arg_scope(resnet_arg_scope(is_training=False)):
            net    = self.build_base()
            net, _ = resnet_v1.resnet_v1(net,
                                         self.blocks[0:cfg.RESNET.FIXED_BLOCKS], 
                                         global_pool=False,
                                         include_root_block=False,
                                         scope=self.scope)
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            head, _ = resnet_v1.resnet_v1(net,
                                          self.blocks[cfg.RESNET.FIXED_BLOCKS:-2], 
                                          global_pool=False,
                                          include_root_block=False,
                                          scope=self.scope)
        return head

    def res5(self, pool5_H, is_training, name):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):

            pool5_H, _ = resnet_v1.resnet_v1(pool5_H, 
                                           self.blocks[-2:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           reuse=False,
                                           scope=self.scope)

            fc5_H = tf.reduce_mean(pool5_H, axis=[1, 2])
        
        return fc5_H

    def crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name) as scope:

            batch_ids    = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            bottom_shape = tf.shape(bottom)
            height       = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self.stride[0])
            width        = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self.stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height

            bboxes        = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            if cfg.RESNET.MAX_POOL:
                pre_pool_size = cfg.POOLING_SIZE * 2
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")
                crops = slim.max_pool2d(crops, [2, 2], padding='SAME')
            else:
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [cfg.POOLING_SIZE, cfg.POOLING_SIZE], name="crops")
        return crops

    def region_classification(self, fc7_P, is_training, initializer, name):
        with tf.variable_scope(name) as scope:

            cls_score_P = slim.fully_connected(fc7_P, self.num_classes, 
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None, scope='cls_score_P')
            cls_prob_P  = tf.nn.sigmoid(cls_score_P, name='cls_prob_P') 
            tf.reshape(cls_prob_P, [1, self.num_classes]) 
            self.predictions["cls_score_P"] = cls_score_P
            self.predictions["cls_prob_P"]  = cls_prob_P

            self.predictions["cls_prob_HO"]  = cls_prob_P 

        return

    def part_classification(self, pool5_P, is_training, initializer, num_state, name):
        with tf.variable_scope(name) as scope:
            fc6_P    = slim.fully_connected(pool5_P, 512)
            fc6_P    = slim.dropout(fc6_P, keep_prob=0.5, is_training=is_training)
            fc7_P    = slim.fully_connected(fc6_P, 512)
            fc7_P    = slim.dropout(fc7_P, keep_prob=0.5, is_training=is_training)

        return fc7_P

    def pasta_classification(self, pool5_P0, pool5_P1, pool5_P2, pool5_P3, pool5_P4, pool5_P5, is_training, initializer, name):
        with tf.variable_scope(name) as scope:
            fc7_P0 = self.part_classification(pool5_P0, is_training, initializer, self.num_pasta0, 'cls_pasta_0')
            fc7_P1 = self.part_classification(pool5_P1, is_training, initializer, self.num_pasta1, 'cls_pasta_1')
            fc7_P2 = self.part_classification(pool5_P2, is_training, initializer, self.num_pasta2, 'cls_pasta_2')
            fc7_P3 = self.part_classification(pool5_P3, is_training, initializer, self.num_pasta3, 'cls_pasta_3')
            fc7_P4 = self.part_classification(pool5_P4, is_training, initializer, self.num_pasta4, 'cls_pasta_4')
            fc7_P5 = self.part_classification(pool5_P5, is_training, initializer, self.num_pasta5, 'cls_pasta_5')
            fc7_P  = tf.concat([fc7_P0, fc7_P1, fc7_P2, fc7_P3, fc7_P4, fc7_P5], axis=1)
        return fc7_P

    def ROI_for_parts(self, head, fc5_H, fc5_S, P_boxes, name):
        with tf.variable_scope(name) as scope:
            pool5_P0 = tf.reduce_mean(self.crop_pool_layer(head, P_boxes[:, 0, :], 'crop_P0'), axis=[1, 2]) # RAnk
            pool5_P1 = tf.reduce_mean(self.crop_pool_layer(head, P_boxes[:, 1, :], 'crop_P1'), axis=[1, 2]) # RKnee
            pool5_P2 = tf.reduce_mean(self.crop_pool_layer(head, P_boxes[:, 2, :], 'crop_P2'), axis=[1, 2]) # LKnee
            pool5_P3 = tf.reduce_mean(self.crop_pool_layer(head, P_boxes[:, 3, :], 'crop_P3'), axis=[1, 2]) # LAnk
            pool5_P4 = tf.reduce_mean(self.crop_pool_layer(head, P_boxes[:, 4, :], 'crop_P4'), axis=[1, 2]) # Hip
            pool5_P5 = tf.reduce_mean(self.crop_pool_layer(head, P_boxes[:, 5, :], 'crop_P5'), axis=[1, 2]) # Head
            pool5_P6 = tf.reduce_mean(self.crop_pool_layer(head, P_boxes[:, 6, :], 'crop_P6'), axis=[1, 2]) # RHand
            pool5_P7 = tf.reduce_mean(self.crop_pool_layer(head, P_boxes[:, 7, :], 'crop_P7'), axis=[1, 2]) # RSho
            pool5_P8 = tf.reduce_mean(self.crop_pool_layer(head, P_boxes[:, 8, :], 'crop_P8'), axis=[1, 2]) # LSho
            pool5_P9 = tf.reduce_mean(self.crop_pool_layer(head, P_boxes[:, 9, :], 'crop_P9'), axis=[1, 2]) # LHand
            
            fc5_S    = tf.tile(fc5_S, [tf.shape(pool5_P0)[0], 1])
            fc5_P0 = tf.concat([pool5_P0, pool5_P3, fc5_H, fc5_S], axis=1)
            fc5_P1 = tf.concat([pool5_P1, pool5_P2, fc5_H, fc5_S], axis=1)
            fc5_P2 = tf.concat([pool5_P4, fc5_H, fc5_S], axis=1)
            fc5_P3 = tf.concat([pool5_P6, pool5_P9, fc5_H, fc5_S], axis=1)
            fc5_P4 = tf.concat([pool5_P7, pool5_P8, fc5_H, fc5_S], axis=1)
            fc5_P5 = tf.concat([pool5_P5, fc5_H, fc5_S], axis=1)
            
        return fc5_P0, fc5_P1, fc5_P2, fc5_P3, fc5_P4, fc5_P5

    def build_network(self, is_training):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)

        head       = self.image_to_head(is_training) 
        pool5_H    = self.crop_pool_layer(head, self.H_boxes, 'Crop_H') 

        fc5_H = self.res5(pool5_H, is_training, name='res5_HR')
        fc5_S = tf.reduce_mean(head, axis=[1, 2])
        fc5_P0, fc5_P1, fc5_P2, fc5_P3, fc5_P4, fc5_P5 = self.ROI_for_parts(head, fc5_H, fc5_S, self.P_boxes, 'ROI_for_parts')

        fc7_P = self.pasta_classification(fc5_P0, fc5_P1, fc5_P2, fc5_P3, fc5_P4, fc5_P5, is_training, initializer, 'pasta_classification')

        self.region_classification(fc7_P, is_training, initializer, 'region_classification')

        self.score_summaries.update(self.predictions)
        return

    def create_architecture(self, is_training):

        self.build_network(is_training)

        for var in tf.trainable_variables():
            self.train_summaries.append(var)

        self.add_loss()
        layers_to_output = {}
        layers_to_output.update(self.losses)

        val_summaries = []
        with tf.device("/cpu:0"):
            for key, var in self.event_summaries.items():
                val_summaries.append(tf.summary.scalar(key, var))
        
        val_summaries.append(tf.summary.scalar('lr', self.lr))
        self.summary_op     = tf.summary.merge_all()
        self.summary_op_val = tf.summary.merge(val_summaries)

        return layers_to_output

    def add_loss(self):

        with tf.variable_scope('LOSS') as scope:
            cls_score_P = self.predictions["cls_score_P"]

            label_verb   = self.gt_verb

            # P stream
            P_cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=label_verb, logits=cls_score_P, pos_weight=self.verb_weight)
            P_cross_entropy = tf.reduce_mean(P_cross_entropy)

            self.losses['P_cross_entropy']  = P_cross_entropy

            loss = P_cross_entropy

            self.losses['total_loss'] = loss
            self.event_summaries.update(self.losses)

        return loss

    def add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)

    def train_step(self, sess, blobs, lr, train_op):
        feed_dict = {self.image: blobs['image'], 
                     self.H_boxes: blobs['H_boxes'], self.P_boxes: blobs['P_boxes'],self.gt_verb: blobs['gt_verb'],  
                     self.lr: lr, self.H_num: blobs['H_num']}
        
        loss, _ = sess.run([self.losses['total_loss'], 
                            train_op],
                            feed_dict=feed_dict)
        return loss

    def train_step_with_summary(self, sess, blobs, lr, train_op):
        feed_dict = {self.image: blobs['image'], 
                     self.H_boxes: blobs['H_boxes'], self.P_boxes: blobs['P_boxes'], self.gt_verb: blobs['gt_verb'], 
                     self.lr: lr, self.H_num: blobs['H_num']}

        loss, summary, _ = sess.run([self.losses['total_loss'], 
                                     self.summary_op, 
                                     train_op], 
                                     feed_dict=feed_dict)
        return loss, summary

    def test_image_HO(self, sess, image, blobs):
        feed_dict = {self.image: image, 
                     self.H_boxes: blobs['H_boxes'], self.P_boxes: blobs['P_boxes'], self.H_num: blobs['H_num']}
        cls_prob_HO = sess.run([self.predictions["cls_prob_HO"]], feed_dict=feed_dict)
        return cls_prob_HO
        
