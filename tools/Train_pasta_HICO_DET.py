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
from models.train_Solver_HICO_DET_pasta import train_net
from networks.pasta_HICO_DET import ResNet50

def parse_args():
    parser = argparse.ArgumentParser(description='Train an iCAN on HICO')
    parser.add_argument('--num_iteration', dest='max_iters',
            help='Number of iterations to perform',
            default=1800000, type=int)
    parser.add_argument('--iter', dest='iter', help='iter to load', default=1800000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='', type=str)
    parser.add_argument('--Pos_augment', dest='Pos_augment',
            help='Number of augmented detection for each one.',
            default=15, type=int)
    parser.add_argument('--Neg_select', dest='Neg_select',
            help='Number of Negative example selected for each image',
            default=60, type=int)
    parser.add_argument('--Restore_flag', dest='Restore_flag',
            help='How many ResNet blocks are there?',
            default=5, type=int)
    parser.add_argument('--train_continue', dest='train_continue',
            help='Whether to continue from previous ckpt',
            default=cfg.TRAIN_MODULE_CONTINUE, type=int)
    parser.add_argument('--init_weight', dest='init_weight',
            help='How to init weight',
            default=cfg.TRAIN_INIT_WEIGHT, type=int)
    parser.add_argument('--module_update', dest='module_update',
            help='How to update modules',
            default=cfg.TRAIN_MODULE_UPDATE, type=int)
    parser.add_argument('--train_module', dest='train_module',
            help='How to compute loss',
            default=cfg.TRAIN_MODULE, type=int)
    parser.add_argument('--data', dest='--data',
            help='which data to choose',
            default=0, type=int)
    args = parser.parse_args()
    return args



if __name__ == '__main__':

    args = parse_args()
    cfg.TRAIN.SNAPSHOT_ITERS    = 100000
    cfg.TRAIN_MODULE_CONTINUE   = args.train_continue
    cfg.TRAIN_INIT_WEIGHT       = args.init_weight
    cfg.TRAIN_MODULE_UPDATE     = args.module_update
    cfg.TRAIN_MODULE            = args.train_module

    if args.data == 0:   # pretrain/train on HICO-DET
        Trainval_GT       = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_all_part.pkl', "rb"))
        Trainval_N        = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_Neg_all_part.pkl', "rb")) 
    elif args.data == 1: # pretrain on full PaStaNet
        Trainval_GT       = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_10w.pkl', "rb"))
        Trainval_N        = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_Neg_10w.pkl', "rb"))

    np.random.seed(cfg.RNG_SEED)
    # change this to trained model for finetune, 1800000, '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'
    if cfg.TRAIN_MODULE_CONTINUE == 1:
        weight    = cfg.ROOT_DIR + '/Weights/' + args.model + '/HOI_iter_%d.ckpt' % args.iter # from previous_ckpt
    else:
        if cfg.TRAIN_INIT_WEIGHT == 1:   # pretrain on full PaStaNet or train on HICO-DET
            weight    = cfg.ROOT_DIR + '/Weights/faster_rcnn_resnet101_coco_2018_01_28/model.ckpt' # from faster R-CNN
        elif cfg.TRAIN_INIT_WEIGHT == 2: # finetune on HICO-DET
            weight    = cfg.ROOT_DIR + '/Weights/pretrain_for_HICO_DET/model.ckpt'
    
    # output directory where the logs are saved
    tb_dir     = cfg.ROOT_DIR + '/logs/' + args.model + '/'

    # output directory where the models are saved
    output_dir = cfg.ROOT_DIR + '/Weights/' + args.model + '/'

    net = ResNet50()
 
    train_net(net, Trainval_GT, Trainval_N, output_dir, tb_dir, args.Pos_augment, args.Neg_select, args.Restore_flag, weight, max_iters=args.max_iters)
