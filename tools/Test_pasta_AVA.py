from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import tensorflow as tf
import numpy as np
import argparse
import cPickle as pickle
import os

from ult.config import cfg
from models.test_Solver_AVA_pasta import test_net
from networks.pasta_AVA import ResNet50

def parse_args():
    parser = argparse.ArgumentParser(description='Test an iCAN on AVP')
    parser.add_argument('--iteration', dest='iteration',
            help='Number of iterations to load',
            default=1800000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='', type=str)
    parser.add_argument('--Restore_flag', dest='Restore_flag',
            help='How many ResNet blocks are there?',
            default=5, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    cfg.TRAIN_MODULE = 6
    args = parse_args()
    np.random.seed(cfg.RNG_SEED)
    Test_RCNN = pickle.load( open( cfg.DATA_DIR + '/' + 'ava_val_fixed.pkl', "rb" ) ) # test detections
    
    # change this to trained model for finetune, 1800000, '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'
    # pretrain model
    weight = cfg.ROOT_DIR + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'

    # output directory where the logs are saved
    print ('iter = ' + str(args.iteration) + ', path = ' + weight ) 
  
    output_file = cfg.ROOT_DIR + '/Results/' + str(args.iteration) + '_' + args.model +  '.csv'

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    sess = tf.Session(config=tfconfig)

    net = ResNet50()
    net.create_architecture(False)
    
    saver = tf.train.Saver()
    saver.restore(sess, weight)

    print('Pre-trained weights loaded.')

    test_net(sess, net, Test_RCNN, output_file)
    sess.close()
    os.chdir(cfg.ROOT_DIR + '/eval')
    os.system("python -O get_ava_performance.py -l ava/ava_action_list_v2.1_for_activitynet_2018.pbtxt.txt -g ava_val_v2.1.csv -e ava_val_excluded_timestamps_v2.1.csv -d " + output_file + "> " + output_file[:-4] + '.txt')
