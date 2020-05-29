import argparse
import os
import os.path as osp
from test_before_mil import test_one_caffemodel
from obj_mask import obj_mask_func
from obj_mask import prior_info

parser = argparse.ArgumentParser()
parser.add_argument('--pasta-mode', type=str, default='linear', help='linear/mlp/seq/gcn/tree')

args = parser.parse_args()

# PaSta-Models
if args.pasta_mode == 'linear':
    if not osp.exists('results/Pasta-Linear/Pasta-Linear.csv'):
        test_one_caffemodel('models/PaSta-Models/PaSta-Linear/deploy_PaSta-Linear.prototxt', 'snaps/PaSta-Linear.caffemodel.h5', 'results/Pasta-Linear/', 'pvp', 'Pasta-Linear')
        obj_mask_func('results/Pasta-Linear/Pasta-Linear.csv')
elif args.pasta_mode == 'mlp':
    if not osp.exists('results/Pasta-MLP/Pasta-MLP.csv'):
        test_one_caffemodel('models/PaSta-Models/PaSta-MLP/deploy_PaSta-MLP.prototxt', 'snaps/PaSta-MLP.caffemodel.h5', 'results/Pasta-MLP/', 'pvp', 'Pasta-MLP')
        obj_mask_func('results/Pasta-MLP/Pasta-MLP.csv')
elif args.pasta_mode == 'seq':
    if not osp.exists('results/Pasta-Seq/Pasta-Seq.csv'):
        test_one_caffemodel('models/PaSta-Models/PaSta-Seq/deploy_PaSta-Seq.prototxt', 'snaps/PaSta-Seq.caffemodel.h5', 'results/Pasta-Seq/', 'pvp', 'Pasta-Seq')
        obj_mask_func('results/Pasta-Seq/Pasta-Seq.csv')
elif args.pasta_mode == 'gcn':
    if not osp.exists('results/Pasta-GCN/Pasta-GCN.csv'):
        test_one_caffemodel('models/PaSta-Models/PaSta-GCN/deploy_PaSta-GNN.prototxt', 'snaps/PaSta-GCN.caffemodel.h5', 'results/Pasta-GCN/', 'pvp', 'Pasta-GCN')
        obj_mask_func('results/Pasta-GCN/Pasta-GCN.csv')
elif args.pasta_mode == 'tree':
    if not osp.exists('results/Pasta-Tree/Pasta-Tree.csv'):
        test_one_caffemodel('models/PaSta-Models/PaSta-Tree/deploy_PaSta-Tree.prototxt', 'snaps/PaSta-Tree.caffemodel.h5', 'results/Pasta-Tree/', 'pvp', 'Pasta-Tree')
        obj_mask_func('results/Pasta-Tree/Pasta-Tree.csv')

# 10v_attention
if not osp.exists('results/10v-attention/10v-attention.csv'):
    test_one_caffemodel('models/10v_attention/deploy_10v_attention.prototxt', 'snaps/10v_attention.caffemodel.h5', 'results/10v-attention/', '10v', '10v-attention')
    obj_mask_func('results/10v-attention/10v-attention.csv')

# Language-Models
if not osp.exists('results/language-model/language-model.csv'):
    test_one_caffemodel('models/Language-Models/deploy_language-model.prototxt', 'snaps/language_model.caffemodel.h5', 'results/language-model/', 'pvp', 'language-model')
    obj_mask_func('results/language-model/language-model.csv')


# Fuse the results
if args.pasta_mode == 'linear':
    os.system('python scripts/fuse.py results/Pasta-Linear/Pasta-Linear.csv results/10v-attention/10v-attention.csv results/pairwise.csv results/language-model/language-model.csv')
    prior_info('./results/fused.csv', './data/hico/object80.csv', './data/hico/verb117.csv', 800, 50, './results/linear_final.csv')
elif args.pasta_mode == 'mlp':
    os.system('python scripts/fuse.py results/Pasta-MLP/Pasta-MLP.csv results/10v-attention/10v-attention.csv results/pairwise.csv results/language-model/language-model.csv')
    prior_info('./results/fused.csv', './data/hico/object80.csv', './data/hico/verb117.csv', 800, 50, './results/mlp_final.csv')
elif args.pasta_mode == 'seq':
    os.system('python scripts/fuse.py results/Pasta-Seq/Pasta-Seq.csv results/10v-attention/10v-attention.csv results/pairwise.csv results/language-model/language-model.csv')
    prior_info('./results/fused.csv', './data/hico/object80.csv', './data/hico/verb117.csv', 800, 50, './results/seq_final.csv')
elif args.pasta_mode == 'gcn':
    os.system('python scripts/fuse.py results/Pasta-GCN/Pasta-GCN.csv results/10v-attention/10v-attention.csv results/pairwise.csv results/language-model/language-model.csv')
    prior_info('./results/fused.csv', './data/hico/object80.csv', './data/hico/verb117.csv', 800, 50, './results/gcn_final.csv')
elif args.pasta_mode == 'tree':
    os.system('python scripts/fuse.py results/Pasta-GCN/Pasta-GCN.csv results/10v-attention/10v-attention.csv results/pairwise.csv results/language-model/language-model.csv')
    prior_info('./results/fused.csv', './data/hico/object80.csv', './data/hico/verb117.csv', 800, 50, './results/tree_final.csv')
