
import cPickle as pickle
import numpy as np
import os
import argparse
import torch
import h5py
from HICO_DET_utils import calc_ap, obj_range, rare, getSigmoid, hoi_no_inter_all
from HICO_DET_utils import obj_range, getSigmoid, hoi_no_inter_all

def parse_args():
    parser = argparse.ArgumentParser(description='Generate detection file')
    parser.add_argument('--model', dest='model',
            help='Select model to generate',
            default='', type=str)
    args = parser.parse_args()
    return args

args = parse_args()

im_index  = np.zeros(10000).astype(np.int32)
mapping = pickle.load(open('key_mapping.pkl', 'rb'))
for key in mapping.keys():
    im_index[key] = mapping[key]
with h5py.File('hico_caffe600.h5', 'r') as f:
    score_I = f['w'][:, :]

score_H  = pickle.load(open('TIN/score_H.pkl', 'rb'))
score_O  = pickle.load(open('TIN/score_O.pkl', 'rb'))
score_sp = pickle.load(open('TIN/score_sp.pkl', 'rb'))
hdet     = pickle.load(open('TIN/hdet.pkl', 'rb'))
odet     = pickle.load(open('TIN/odet.pkl', 'rb'))
keys     = pickle.load(open('TIN/keys.pkl', 'rb'))
pos      = pickle.load(open('TIN/pos.pkl', 'rb'))
neg      = pickle.load(open('TIN/neg.pkl', 'rb'))
bboxes   = pickle.load(open('TIN/bboxes.pkl', 'rb'))

score_P  = pickle.load(open(args.model + '/scores_P.pkl', 'rb'))
score_A  = pickle.load(open(args.model + '/scores_A.pkl', 'rb'))
score_L  = pickle.load(open(args.model + '/scores_L.pkl', 'rb'))

h_fac, o_fac, sp_fac, P_fac, A_fac, L_fac, hthresh, othresh, athresh, bthresh, P_weight, A_weight, L_weight = pickle.load(open('generation_args.pkl', 'rb'))

detection = {}
detection['bboxes'] = []
detection['scores'] = []
detection['index']  = []
detection['keys']   = []

for i in range(600):
    detection['index'].append([])
    detection['scores'].append([])
    detection['bboxes'].append([])
    detection['keys'].append([])

for obj_index in range(80):
    x, y = obj_range[obj_index]
    x -= 1
    
    inter_det_mask = (hdet[obj_index] > hthresh[x]) * (odet[obj_index] > othresh[x])
    no_inter_det_mask = (hdet[obj_index] > hthresh[y-1]) * (odet[obj_index] > othresh[y-1])

    for hoi_index in range(x, y):
        score_H[obj_index][:, hoi_index - x]  /= h_fac[hoi_index]
        score_O[obj_index][:, hoi_index - x]  /= o_fac[hoi_index]
        score_sp[obj_index][:, hoi_index - x] /= sp_fac[hoi_index]
        score_P[obj_index][:, hoi_index - x]  /= P_fac[hoi_index]
        score_A[obj_index][:, hoi_index - x]  /= A_fac[hoi_index]
        score_L[obj_index][:, hoi_index - x]  /= L_fac[hoi_index]

    hod  = getSigmoid(9, 1, 3, 0, hdet[obj_index].reshape(-1, 1)) * getSigmoid(9, 1, 3, 0, odet[obj_index].reshape(-1, 1))
    sH  = torch.sigmoid(torch.from_numpy(score_H[obj_index]).cuda()).cpu().numpy()
    sO  = torch.sigmoid(torch.from_numpy(score_O[obj_index]).cuda()).cpu().numpy()
    ssp = torch.sigmoid(torch.from_numpy(score_sp[obj_index]).cuda()).cpu().numpy()
    sP  = torch.sigmoid(torch.from_numpy(score_P[obj_index]).cuda()).cpu().numpy() * P_weight
    sA  = torch.sigmoid(torch.from_numpy(score_A[obj_index]).cuda()).cpu().numpy() * A_weight
    sL  = torch.sigmoid(torch.from_numpy(score_L[obj_index]).cuda()).cpu().numpy() * L_weight
    sHO = (((sH + sO) * ssp + sP + sA + sL) * score_I[im_index[keys[obj_index]], x:y]) * hod

    for hoi_index in range(x, y):
        at, bt = athresh[hoi_index], bthresh[hoi_index]
        if hoi_index + 1 in hoi_no_inter_all:
            nis_mask = 1 - (pos[obj_index] > at) * (neg[obj_index] < bt)
            mask   = no_inter_det_mask * nis_mask
        else:
            nis_mask = 1 - (pos[obj_index] < at) * (neg[obj_index] > bt)
            mask = inter_det_mask * nis_mask
        select        = np.where(mask > 0)[0]
        detection['scores'][hoi_index] = sHO[select, hoi_index - x]
        detection['bboxes'][hoi_index] = bboxes[obj_index][select]
        detection['keys'][hoi_index]   = keys[obj_index][select]

pickle.dump(detection, open('Detection_' + args.model + '.pkl', 'wb'))
        