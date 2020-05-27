# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# --------------------------------------------------------
# 
# Modified from Multitask Network Cascade
# Copyright (c) 2017, Haoshu Fang
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import caffe
import numpy as np
import yaml


DEBUG = False
PRINT_GRADIENT = 1


class SelectLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)

        self._top_k = layer_params.get('top_k', 0)
        self._ind = np.zeros((bottom[0].data.shape[0],self._top_k),dtype=int)
   
        for i in range(self._top_k):
            top[i].reshape(*bottom[1].data.shape)

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def forward(self, bottom, top):
        score_vec = bottom[0].data
        for i in range(score_vec.shape[0]):
            #self._ind[i] = np.argpartition(score_vec[i,...], -self._top_k)[0:self._top_k]
            self._ind[i] = np.array(score_vec[i,...]).argsort()[::-1][:self._top_k]
        for i in range(score_vec.shape[0]):
            for j in range(self._top_k):
                top[j].data[i,...]=bottom[self._ind[i,j]+1].data[i,...]
        

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff.fill(0.0)
        for i in range(bottom[0].data.shape[1]):
            bottom[i+1].diff.fill(0.0)
        for i in range(bottom[0].data.shape[0]):
            for j in range(self._top_k):
                bottom[self._ind[i,j]+1].diff[i,...]=top[j].diff[i,...]
