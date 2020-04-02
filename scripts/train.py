import argparse
import os
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument('--pasta-mode', type=str, default='linear', help='linear/mlp/seq/gcn/tree')

args = parser.parse_args()

# PaSta-Models
if args.pasta_mode == 'linear':
    if not osp.exists('snaps/PaSta-Linear.caffemodel.h5'):
        os.system("build/tools/caffe train -solver models/PaSta-Models/PaSta-Linear/solver_PaSta-Linear.prototxt -weights snaps/pretrain_model.caffemodel -gpu 0 2>&1 |tee log/pasta_linear.log")
elif args.pasta_mode == 'mlp':
    if not osp.exists('snaps/PaSta-MLP.caffemodel.h5'):
        os.system("build/tools/caffe train -solver models/PaSta-Models/PaSta-MLP/solver_PaSta-MLP.prototxt -weights snaps/pretrain_model.caffemodel -gpu 0 2>&1 |tee log/pasta_mlp.log")
elif args.pasta_mode == 'seq':
    if not osp.exists('snaps/PaSta-Seq.caffemodel.h5'):
        os.system("build/tools/caffe train -solver models/PaSta-Models/PaSta-Seq/solver_PaSta-Seq.prototxt -weights snaps/pretrain_model.caffemodel -gpu 0 2>&1 |tee log/pasta_seq.log")
elif args.pasta_mode == 'gcn':
    if not osp.exists('snaps/PaSta-GCN.caffemodel.h5'):
        os.system("build/tools/caffe train -solver models/PaSta-Models/PaSta-GCN/solver_PaSta-GNN.prototxt -weights snaps/pretrain_model.caffemodel -gpu 0 2>&1 |tee log/pasta_gcn.log")
elif args.pasta_mode == 'tree':
    if not osp.exists('snaps/PaSta-Tree.caffemodel.h5'):
        os.system("build/tools/caffe train -solver models/PaSta-Models/PaSta-Tree/solver_PaSta-Tree.prototxt -weights snaps/pretrain_model.caffemodel -gpu 0 2>&1 |tee log/pasta_tree.log")

# 10v_attention
if not osp.exists('snaps/10v_attention.caffemodel.h5'):
    os.system("build/tools/caffe train -solver models/10v_attention/solver_10v_attention.prototxt -weights snaps/pretrain_model.caffemodel -gpu 0 2>&1 |tee log/10v_attention.log")

# Language-Models
if not osp.exists('snaps/language_model.caffemodel.h5'):
    os.system("build/tools/caffe train -solver models/Language-Models/solver_language-model.prototxt -weights snaps/pretrain_model.caffemodel -gpu 0 2>&1 |tee log/language_model.log")
