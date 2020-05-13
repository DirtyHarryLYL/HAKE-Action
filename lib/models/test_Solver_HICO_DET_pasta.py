
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ult.config import cfg
from ult.timer import Timer
from ult.ult_HICO_DET import Get_next_sp_with_pose, Generate_part_bbox, Generate_relation_bbox, Generate_action_object
from ult.HICO_DET_utils import obj_range, get_map

import cPickle as pickle
import numpy as np
import os
import sys
import glob
import time
import cv2

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

cnt = np.array([
    71,113,38,204,125,52,260,278,9,106,30,1380,124,150,26,17,116,89,1447,1183,1502,182,6,168,12,93,113,12,15,131,28,47,189,28,81,11,103,35,1382,339,182,997,594,34,8,106,159,200,541,55,13,20,27,195,137,12,532,30,29,43,789,431,10,5,107,31,6,393,67,79,5,28,35,484,149,306,9,8,197,101,13,95,65,10,6,40,43,117,92,1005,6,97,19,253,658,39,16,73,90,10,10,54,34,52,12,67,108,10,789,993,79,119,7,17,12,377,22,211,21,28,188,25,90,23,47,122,25,8,113,25,38,750,36,181,19,5,13,96,141,1253,440,1001,124,125,28,33,923,65,141,8,99,17,408,1131,1419,1313,278,18,8,138,110,23,552,314,81,10,6,58,11,249,14,14,11,61,25,54,53,171,39,7,30,6,119,34,6,92,77,137,8,7,292,163,10,143,24,9,132,88,10,26,61,239,41,30,35,7,7,22,914,57,23,18,983,89,14,139,10,129,321,117,14,24,7,104,53,90,810,6,407,7,716,7,194,194,6,23,33,51,8,7,345,156,37,19,107,1463,107,118,882,666,769,64,81,33,10,7,40,6,42,63,6,6,6,17,345,119,235,80,166,32,15,87,48,73,12,29,46,188,28,6,6,8,25,215,586,198,6,278,151,6,58,51,11,16,148,157,145,353,21,29,105,34,54,6,67,12,58,138,262,26,86,6,40,46,125,8,21,7,18,91,261,135,41,287,11,6,29,17,11,122,157,122,36,10,7,21,112,348,503,24,267,41,176,17,28,7,266,42,63,59,7,6,80,144,10,23,139,79,6,109,217,39,29,57,8,59,243,56,53,220,642,515,109,138,472,39,268,651,44,6,657,9,28,257,44,582,19,270,58,6,12,12,20,51,57,3,25,6,7,7,58,7,8,8,11,7,13,9,16,80,10,24,35,56,38,25,6,113,5,84,37,61,110,180,48,55,13,6,69,7,45,9,27,30,216,77,8,23,39,6,8,132,214,18,51,158,252,104,27,8,15,14,63,137,593,337,394,1185,58,1842,25,1480,50,11,52,83,25,141,34,11,546,581,716,13,10,42,74,180,344,604,602,657,10,290,115,6,48,17,64,220,109,139,251,381,105,321,27,83,6,6,168,14,28,14,4,32,227,130,228,6,14,13,34,31,6,97,231,11,304,21,9,29,6,240,181,18,6,61,47,240,68,8,35,186,241,6,112,14,71,6,108,26,42,42,659,67,8,7,6,6,11,6,8,19,68,8,10,108,252,351,6,17,15,19,16,15,121,25,183,47,79,25,206,154,17,74,889,1222,10,19,14,15,943,113,23,25,6,61,26,395,62,49,6,6,191,11,8,8,41,8
])

def im_detect(sess, net, image_id, Test_RCNN, keys, scores_P, scores_A, scores_L, bboxes, hdet, odet):
    This_image   = []
    im_file      = cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/HICO_test2015_' + (str(image_id)).zfill(8) + '.jpg'
    im           = cv2.imread(im_file)
    im_orig      = im.astype(np.float32, copy=True)
    im_orig     -= cfg.PIXEL_MEANS
    im_orig      = im_orig.reshape(1, im_orig.shape[0], im_orig.shape[1], 3)
    blobs        = {}
    
    for i in range(len(Test_RCNN[image_id])):
    # for Human_out in Test_RCNN[image_id]:
        Human_out = Test_RCNN[image_id][i]
        if Human_out[2] == 1:
            H_box = np.array([0, Human_out[1][0],  Human_out[1][1],  Human_out[1][2],  Human_out[1][3]]).reshape(1,5)
            P_box = Human_out[6][None, ...]

            blobs['H_num'] = 0
            blobs['H_boxes'] = np.empty((0, 5), dtype=np.float64)
            blobs['O_boxes'] = np.empty((0, 5), dtype=np.float64)
            blobs['R_boxes'] = np.empty((0, 5), dtype=np.float64)
            blobs['P_boxes'] = np.empty((0, 10, 5), dtype=np.float64)
            blobs['gt_object'] = np.empty((0, 80), dtype=np.float64)

            index = []
            for j in range(len(Test_RCNN[image_id])):
                if i != j: 
                    Object = Test_RCNN[image_id][j]
                    blobs['H_num'] += 1
                    # This is a valid object
                    # 1.the object detection result should > thres  2.the bbox detected is n object
                    index.append(j)
                    blobs['H_boxes'] = np.concatenate([blobs['H_boxes'], H_box], axis=0)
                    blobs['O_boxes'] = np.concatenate([blobs['O_boxes'], np.array([0, Object[1][0], Object[1][1], Object[1][2], Object[1][3]], dtype='float64').reshape(1, -1)])
                    blobs['P_boxes'] = np.concatenate([blobs['P_boxes'], P_box], axis=0)
                    blobs['R_boxes'] = np.concatenate([blobs['R_boxes'], Generate_relation_bbox(Human_out[1], Object[1])], axis=0)
                    blobs['gt_object'] = np.concatenate([blobs['gt_object'], Generate_action_object(Object[2]-1, 80)], axis=0)
            
            if blobs['H_num'] > 0:
                P, A, L = net.test_image_HO(sess, im_orig, blobs)
                for j in range(blobs['H_num']):
                    Object = Test_RCNN[image_id][index[j]]
                    classid = Object[2] - 1
                    keys[classid].append(image_id)
                    scores_P[classid].append(P[j][obj_range[classid][0] - 1:obj_range[classid][1]].reshape(1, -1))
                    scores_A[classid].append(A[j][obj_range[classid][0] - 1:obj_range[classid][1]].reshape(1, -1))
                    scores_L[classid].append(L[j][obj_range[classid][0] - 1:obj_range[classid][1]].reshape(1, -1))
                    hbox = np.array(Human_out[1]).reshape(1, -1)
                    obox = np.array(Object[1]).reshape(1, -1)
                    bboxes[classid].append(np.concatenate([hbox, obox], axis=1))
                    hdet[classid].append(Human_out[3])
                    odet[classid].append(Object[3])


def test_net(sess, net, Test_RCNN, output_dir):
    
    np.random.seed(cfg.RNG_SEED)
    detection = {}
    count = 0
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    keys, scores_P, scores_A, scores_L, bboxes, hdet, odet = [], [], [], [], [], [], []
    for i in range(80):
        keys.append([])
        scores_P.append([])
        scores_A.append([])
        scores_L.append([])
        bboxes.append([])
        hdet.append([])
        odet.append([])

    for line in glob.iglob(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/*.jpg'):

        _t['im_detect'].tic()
 
        image_id   = int(line[-9:-4])
        
        im_detect(sess, net, image_id, Test_RCNN, keys, scores_P, scores_A, scores_L, bboxes, hdet, odet)

        _t['im_detect'].toc()

        print('im_detect: {:d}/{:d} {:.3f}s'.format(count + 1, 9658, _t['im_detect'].average_time))
        count += 1
    
    for i in range(80):
        scores_P[i] = np.concatenate(scores_P[i], axis=0)
        scores_A[i] = np.concatenate(scores_A[i], axis=0)
        scores_L[i] = np.concatenate(scores_L[i], axis=0)
        bboxes[i]   = np.concatenate(bboxes[i], axis=0)
        keys[i]     = np.array(keys[i])
        hdet[i]     = np.array(hdet[i])
        odet[i]     = np.array(odet[i])
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    pickle.dump(scores_P, open(os.path.join(output_dir, 'scores_P.pkl'), 'wb'))
    pickle.dump(scores_A, open(os.path.join(output_dir, 'scores_A.pkl'), 'wb'))
    pickle.dump(scores_L, open(os.path.join(output_dir, 'scores_L.pkl'), 'wb'))
    pickle.dump(bboxes, open(os.path.join(output_dir, 'bboxes.pkl'), 'wb'))
    pickle.dump(keys, open(os.path.join(output_dir, 'keys.pkl'), 'wb'))
    pickle.dump(hdet, open(os.path.join(output_dir, 'hdet.pkl'), 'wb'))
    pickle.dump(odet, open(os.path.join(output_dir, 'odet.pkl'), 'wb'))
    

