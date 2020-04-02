
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ult.config import cfg
from ult.timer import Timer
from ult.ult_AVA import Generate_part_bbox

import cPickle as pickle
import numpy as np
import os
import sys
import glob
import time
import ipdb
import cv2
import pickle
import csv

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

action = [1, 3, 4, 5, 6, 7, 8, 9, 10, 
          11, 12, 13, 14, 15, 17, 20, 22, 24, 26, 
          27, 28, 29, 30, 34, 36, 37, 38, 41, 43, 
          45, 46, 47, 48, 49, 51, 52, 54, 56, 57, 
          58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 
          68, 69, 70, 72, 73, 74, 76, 77, 78, 79, 80]

def im_detect(sess, net, image_id, Test_RCNN, detection):
    
    pair = Test_RCNN[image_id]
    if pair[1][0] == '0':
        im_file      = '/Disk6/AVA/input/val/' + pair[0] + '/' + pair[1][1:] + '_0.jpg'
    else:
        im_file      = '/Disk6/AVA/input/val/' + pair[0] + '/' + pair[1] + '_0.jpg'
    # print(im_file)
    im           = cv2.imread(im_file)
    im_orig      = im.astype(np.float32, copy=True)
    im_orig     -= cfg.PIXEL_MEANS
    im_orig      = im_orig.reshape(1, im_orig.shape[0], im_orig.shape[1], 3)
    blobs        = {}
    h, w = im_orig.shape[1], im_orig.shape[2]

    blobs['H_num'] = 1
    blobs['H_boxes'] = np.array([0, w * float(pair[2]),  h * float(pair[3]),  w * float(pair[4]),  h * float(pair[5])]).reshape(1,5)
    # blobs['spatial'] = Get_next_sp_only_pose(blobs['H_boxes'][0, 1:], pair[-1]).reshape(1, 64, 64, 1)
    if (len(pair) >= 10):
        P_box = Generate_part_bbox(pair[8], [w * float(pair[2]),  h * float(pair[3]),  w * float(pair[4]),  h * float(pair[5])])
    else:
        P_box = Generate_part_bbox(0, [w * float(pair[2]),  h * float(pair[3]),  w * float(pair[4]),  h * float(pair[5])])
    blobs['P_boxes'] = P_box

    # cls_prob_verb, cls_prob_PVP, cls_prob_vec = net.test_image_HO(sess, im_orig, blobs)
    pred = net.test_image_HO(sess, im_orig, blobs)[0]
    if len(pred) == 3:
        cls_prob_verb = pred[0]
    else:
        cls_prob_verb = pred
    for i in range(60):
        pre = [pair[0], pair[1], pair[2], pair[3], pair[4], pair[5], action[i], cls_prob_verb[0, action[i] - 1]]
        detection.append(pre)
    return

def test_net(sess, net, Test_RCNN, output_dir):
    
    np.random.seed(cfg.RNG_SEED)
    detection = []
    count = 0
    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    
    for i in range(len(Test_RCNN)):

        _t['im_detect'].tic()
        
        im_detect(sess, net, i, Test_RCNN, detection)

        _t['im_detect'].toc()

        print('im_detect: {:d}/{:d} {:.3f}s'.format(count + 1, len(Test_RCNN), _t['im_detect'].average_time))
        count += 1

    with open(output_dir, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(detection)
