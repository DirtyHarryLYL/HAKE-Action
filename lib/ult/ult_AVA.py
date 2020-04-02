
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
import cPickle as pickle
import random
from random import randint
import tensorflow as tf
import cv2

from ult import config

def Generate_part_bbox(joint_bbox, Human_bbox=None):
    part_bbox = np.zeros([1, 10, 5], dtype=np.float64)
    if joint_bbox is None or isinstance(joint_bbox, int):
        if Human_bbox is None:
            raise ValueError
        for i in range(10):
            part_bbox[0, i, :] = np.array([0, Human_bbox[0], Human_bbox[1], Human_bbox[2], Human_bbox[3]], dtype=np.float64)
    else:
        for i in range(10):
            part_bbox[0, i, :] = np.array([0, max(0, joint_bbox[i]['x1']), max(0,joint_bbox[i]['y1']), max(0,joint_bbox[i]['x2']), max(0,joint_bbox[i]['y2'])], dtype=np.float64)
    return part_bbox


def Generate_action_AVA(idx):
    action_verb = np.zeros([1, 80], dtype=np.float64)
    if isinstance(idx, int):
        if idx != -1:
            action_verb[:, idx - 1] = 1
    else:
        if -1 not in idx:
            tmp = np.array(idx) - 1
            action_verb[:, list(tmp)] = 1
    return action_verb


def Get_Next_Instance_Verb_AVA_transfer(Trainval_GT, image_id, Pos_augment, fake=False):

    im_file = '/Disk6/AVA/input/train/' + str(image_id)
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)
    Human_augmented, Part_bbox, num_pos, gt_verb = Augmented_Verb_AVA_transfer(image_id, Trainval_GT, Pos_augment)

    shape = np.array([0, 0, 0, im_shape[1], im_shape[0]]).astype(np.float64)
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['P_boxes']     = Part_bbox
    blobs['H_num']       = num_pos
    blobs['gt_verb']     = gt_verb

    return blobs

def Augmented_Verb_AVA_transfer(image_id, GT, Pos_augment):
    pair_info = GT[image_id]
    pair_num = len(pair_info)

    # if not sufficient, repeat data
    if pair_num >= Pos_augment:
        GT = []
        for i in range(pair_num):
            GT.append(pair_info[i])
    else:
        GT = []
        for i in range(Pos_augment):
            index = random.randint(0, pair_num - 1)
            GT.append(pair_info[index])

    Human_augmented = np.empty((0, 5), dtype=np.float64)
    part_bbox   = np.empty((0, 10, 5), dtype=np.float64)
    action_verb = np.empty((0, 80), dtype=np.float64)
    for i in range(Pos_augment):
        Human    = GT[i][2]
        Human_augmented  = np.concatenate((Human_augmented, np.array([0, Human[0],  Human[1],  Human[2],  Human[3]]).reshape(1,5).astype(np.float64)), axis=0)
        part_bbox        = np.concatenate((part_bbox, Generate_part_bbox(GT[i][3], Human)), axis=0)
        action_verb      = np.concatenate((action_verb, Generate_action_AVA(GT[i][1])), axis=0)
    num_pos = Pos_augment
    Human_augmented   = Human_augmented.reshape(num_pos, 5) 

    return Human_augmented, part_bbox, num_pos, action_verb
