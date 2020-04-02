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
from models.test_Solver_HICO_DET_pasta import test_net
from networks.pasta_HICO_DET import ResNet50

def parse_args():
    parser = argparse.ArgumentParser(description='Test an pastanet on HICO DET')
    parser.add_argument('--iteration', dest='iteration',
            help='Number of iterations to load',
            default=1800000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='pasta_HICO_DET', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    cfg.TRAIN_MODULE = 2

    args = parse_args()
    Test_RCNN = pickle.load( open( cfg.DATA_DIR + '/' + 'Test_all_part.pkl', "rb" ) ) # test detections
    np.random.seed(cfg.RNG_SEED)
    # pretrain model
    weight = cfg.ROOT_DIR + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'

    # output directory where the logs are saved
    print ('iter = ' + str(args.iteration) + ', path = ' + weight ) 
  
    output_file = cfg.ROOT_DIR + '/-Results/' + str(args.iteration) + '_' + args.model + '/'
    
    if not os.path.exists(output_file):
        os.mkdir(output_file)

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