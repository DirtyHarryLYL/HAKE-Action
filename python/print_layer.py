from __future__ import print_function
import caffe
import numpy as np
class PrintLayer(caffe.Layer):
  def setup(self, bottom, top):
    if len(bottom) != 1:
      raise Exception("Need one inputs.")


    
  def reshape(self, bottom, top):
    top[0].reshape(2)
    

  def forward(self, bottom, top):
    top[0].data[...] = bottom[0].data.shape
    
    

  def backward(self, top, propagate_down, bottom):
    pass
    # for i in range(len(propagate_down)):
    #   if not propagate_down[i]:
    #     continue
    #   bottom[i].diff[...] = top[i].diff[:]
