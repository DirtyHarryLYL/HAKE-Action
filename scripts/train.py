import argparse
import os
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument('--pasta-mode', type=str, default='linear', help='linear/mlp/seq/gcn/tree')
parser.add_argument('--data', type=str, default='hico', help='hico/large')

args = parser.parse_args()

# PaSta-Models
if args.pasta_mode == 'linear':
    if args.data == 'hico':
        if not osp.exists('snaps/PaSta-Linear.caffemodel.h5'):
            os.system("build/tools/caffe train -solver models/PaSta-Models/PaSta-Linear/solver_PaSta-Linear.prototxt -weights snaps/pretrain_model.caffemodel -gpu 0 2>&1 |tee log/pasta_linear.log")
    elif args.data == 'large':
        if not osp.exists('snaps/PaSta-Linear_large.caffemodel.h5'):
            os.system("build/tools/caffe train -solver models/PaSta-Models/PaSta-Linear/solver_PaSta-Linear_large.prototxt -weights snaps/pretrain_model.caffemodel -gpu 0 2>&1 |tee log/pasta_linear_large.log")
elif args.pasta_mode == 'mlp':
    if args.data == 'hico':
        if not osp.exists('snaps/PaSta-MLP.caffemodel.h5'):
            os.system("build/tools/caffe train -solver models/PaSta-Models/PaSta-MLP/solver_PaSta-MLP.prototxt -weights snaps/pretrain_model.caffemodel -gpu 0 2>&1 |tee log/pasta_mlp.log")
    elif args.data == 'large':
        if not osp.exists('snaps/PaSta-MLP_large.caffemodel.h5'):
            os.system("build/tools/caffe train -solver models/PaSta-Models/PaSta-MLP/solver_PaSta-MLP_large.prototxt -weights snaps/pretrain_model.caffemodel -gpu 0 2>&1 |tee log/pasta_mlp_large.log")
elif args.pasta_mode == 'seq':
    if args.data == 'hico':
        if not osp.exists('snaps/PaSta-Seq.caffemodel.h5'):
            os.system("build/tools/caffe train -solver models/PaSta-Models/PaSta-Seq/solver_PaSta-Seq.prototxt -weights snaps/pretrain_model.caffemodel -gpu 0 2>&1 |tee log/pasta_seq.log")
    elif args.data == 'large':
        if not osp.exists('snaps/PaSta-Seq_large.caffemodel.h5'):
            os.system("build/tools/caffe train -solver models/PaSta-Models/PaSta-Seq/solver_PaSta-Seq_large.prototxt -weights snaps/pretrain_model.caffemodel -gpu 0 2>&1 |tee log/pasta_seq_large.log")
elif args.pasta_mode == 'gcn':
    if args.data == 'hico':
        if not osp.exists('snaps/PaSta-GCN.caffemodel.h5'):
            os.system("build/tools/caffe train -solver models/PaSta-Models/PaSta-GCN/solver_PaSta-GNN.prototxt -weights snaps/pretrain_model.caffemodel -gpu 0 2>&1 |tee log/pasta_gcn.log")
    elif args.data == 'large':
        if not osp.exists('snaps/PaSta-GCN_large.caffemodel.h5'):
            os.system("build/tools/caffe train -solver models/PaSta-Models/PaSta-GCN/solver_PaSta-GNN_large.prototxt -weights snaps/pretrain_model.caffemodel -gpu 0 2>&1 |tee log/pasta_gcn_large.log")
elif args.pasta_mode == 'tree':
    if args.data == 'hico':
        if not osp.exists('snaps/PaSta-Tree.caffemodel.h5'):
            os.system("build/tools/caffe train -solver models/PaSta-Models/PaSta-Tree/solver_PaSta-Tree.prototxt -weights snaps/pretrain_model.caffemodel -gpu 0 2>&1 |tee log/pasta_tree.log")
    elif args.data == 'large':
        if not osp.exists('snaps/PaSta-Tree_large.caffemodel.h5'):
            os.system("build/tools/caffe train -solver models/PaSta-Models/PaSta-Tree/solver_PaSta-Tree_large.prototxt -weights snaps/pretrain_model.caffemodel -gpu 0 2>&1 |tee log/pasta_tree_large.log")

# 10v_attention
if args.data == 'hico':
    if not osp.exists('snaps/10v_attention.caffemodel.h5'):
        os.system("build/tools/caffe train -solver models/10v_attention/solver_10v_attention.prototxt -weights snaps/pretrain_model.caffemodel -gpu 0 2>&1 |tee log/10v_attention.log")
elif args.data == 'large':
    if not osp.exists('snaps/10v_attention_large.caffemodel.h5'):
        os.system("build/tools/caffe train -solver models/10v_attention/solver_10v_attention_large.prototxt -weights snaps/pretrain_model.caffemodel -gpu 0 2>&1 |tee log/10v_attention_large.log")

# Language-Models
if args.data == 'hico':
    if not osp.exists('snaps/language_model.caffemodel.h5'):
        os.system("build/tools/caffe train -solver models/Language-Models/solver_language-model.prototxt -weights snaps/pretrain_model.caffemodel -gpu 0 2>&1 |tee log/language_model.log")
elif args.data == 'large':
    if not osp.exists('snaps/language_model_large.caffemodel.h5'):
        os.system("build/tools/caffe train -solver models/Language-Models/solver_language-model_large.prototxt -weights snaps/pretrain_model.caffemodel -gpu 0 2>&1 |tee log/language_model_large.log")

