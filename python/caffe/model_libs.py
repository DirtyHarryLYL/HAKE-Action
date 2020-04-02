import os

import math
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

def check_if_exist(path):
    return os.path.exists(path)

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def UnpackVariable(var, num):
  assert len > 0
  if type(var) is list and len(var) == num:
    return var
  else:
    ret = []
    if type(var) is list:
      assert len(var) == 1
      for i in xrange(0, num):
        ret.append(var[0])
    else:
      for i in xrange(0, num):
        ret.append(var)
    return ret

def ConvBNLayer(net, from_layer, out_layer, use_bn, use_relu, num_output,
    kernel_size, pad, stride, for_HG_module=False, use_scale=True, eps=0.001, conv_prefix='', conv_postfix='',
    bn_prefix='', bn_postfix='_bn', scale_prefix='', scale_postfix='_scale',
    bias_prefix='', bias_postfix='_bias'):

  if use_bn:
    if for_HG_module:
      kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_filler': dict(type='constant', value=0),
        'bias_term': True,
        }
    # parameters for convolution layer with batchnorm.
    else:
      kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        }
    # parameters for batchnorm layer.
    bn_kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        'eps': eps,
        }
    # parameters for scale bias layer after batchnorm.
    if use_scale:
      sb_kwargs = {
          'bias_term': True,
          'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=0)],
          'filler': dict(type='constant', value=1.0),
          'bias_filler': dict(type='constant', value=0.0),
          }
    else:
      bias_kwargs = {
          'param': [dict(lr_mult=1, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          }
  else:
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
        }

  conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
  [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
  [pad_h, pad_w] = UnpackVariable(pad, 2)
  [stride_h, stride_w] = UnpackVariable(stride, 2)
  if kernel_h == kernel_w:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)
  else:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
        stride_h=stride_h, stride_w=stride_w, **kwargs)
  if use_bn:
    bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
    net[bn_name] = L.BatchNorm(net[conv_name], in_place=True,**bn_kwargs)
    if use_scale:
      sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
      if for_HG_module:
        net[sb_name] = L.EltwiseAffine(net[bn_name], in_place=True, channel_shared=False, param=[dict(lr_mult=1, decay_mult=0),dict(lr_mult=1, decay_mult=0)])
      else:
        net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
    else:
      bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
      net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
  if use_relu:
    relu_name = '{}_relu'.format(conv_name)
    net[relu_name] = L.ReLU(net[conv_name], in_place=True)

def ResBody(net, from_layer, block_name, out2a, out2b, out2c, stride, use_branch1, for_HG_module=False):
  # ResBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

  conv_prefix = 'res{}_'.format(block_name)
  conv_postfix = ''
  bn_prefix = 'bn{}_'.format(block_name)
  bn_postfix = ''
  scale_prefix = 'scale{}_'.format(block_name)
  scale_postfix = ''
  use_scale = True

  if use_branch1:
    branch_name = 'branch1'
    if for_HG_module:
      ConvBNLayer(net, from_layer, branch_name, use_bn=False, use_relu=False,
        num_output=out2c, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,for_HG_module=for_HG_module,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    else:
      ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=False,
        num_output=out2c, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    branch1 = '{}{}'.format(conv_prefix, branch_name)
  else:
    branch1 = from_layer

  if for_HG_module:
    bn_kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        'eps': 0.001,
      }
    bn_name = '{}_bn'.format(from_layer)
    net[bn_name] = L.BatchNorm(net[from_layer], **bn_kwargs)
    sb_name = '{}_scale'.format(bn_name)
    net[sb_name] = L.EltwiseAffine(net[bn_name], in_place=True, channel_shared=False,param=[dict(lr_mult=1, decay_mult=0),dict(lr_mult=1, decay_mult=0)])
    relu_name = '{}_relu'.format(sb_name)
    net[relu_name] = L.ReLU(net[sb_name],in_place=True)
    from_layer = relu_name


  branch_name = 'branch2a'
  ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=True,
      num_output=out2a, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,for_HG_module=for_HG_module,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2b'
  ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
      num_output=out2b, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,for_HG_module=for_HG_module,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2c'
  if for_HG_module:
    ConvBNLayer(net, out_name, branch_name, use_bn=False, use_relu=False,
      num_output=out2c, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,for_HG_module=for_HG_module,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  else:
    ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=False,
      num_output=out2c, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  branch2 = '{}{}'.format(conv_prefix, branch_name)

  res_name = 'res{}'.format(block_name)
  net[res_name] = L.Eltwise(net[branch1], net[branch2])
  if not for_HG_module:
    relu_name = '{}_relu'.format(res_name)
    net[relu_name] = L.ReLU(net[res_name], in_place=True)


def InceptionTower(net, from_layer, tower_name, layer_params):
  use_scale = False
  for param in layer_params:
    tower_layer = '{}/{}'.format(tower_name, param['name'])
    del param['name']
    if 'pool' in tower_layer:
      net[tower_layer] = L.Pooling(net[from_layer], **param)
    else:
      ConvBNLayer(net, from_layer, tower_layer, use_bn=True, use_relu=True,
          use_scale=use_scale, **param)
    from_layer = tower_layer
  return net[from_layer]

def CreateAnnotatedDataLayer(source, batch_size=32, backend=P.Data.LMDB,
        output_label=True, train=True, label_map_file='',
        transform_param={}, batch_sampler=[{}]):
    if train:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                'transform_param': transform_param,
                }
    else:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
                'transform_param': transform_param,
                }
    if output_label:
        data, label = L.AnnotatedData(name="data",
            annotated_data_param=dict(label_map_file=label_map_file,
                batch_sampler=batch_sampler),
            data_param=dict(batch_size=batch_size, backend=backend, source=source),
            ntop=2, **kwargs)
        return [data, label]
    else:
        data = L.AnnotatedData(name="data",
            annotated_data_param=dict(label_map_file=label_map_file,
                batch_sampler=batch_sampler),
            data_param=dict(batch_size=batch_size, backend=backend, source=source),
            ntop=1, **kwargs)
        return data

def CreateHeatmapDataLayer(output_label=True, train=True, visualise=False, transform_param={}, heatmap_data_param={}):
    if train:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                'heatmap_data_param': heatmap_data_param,
                'transform_param': transform_param,
                }
    else:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
                'heatmap_data_param': heatmap_data_param,
                'transform_param': transform_param,
                }
    if output_label:
        data, label = L.DataHeatmap(name="data", visualise=visualise, ntop=2, **kwargs)
        return [data, label]
    else:
        data = L.DataHeatmap(name="data", visualise=visualise, ntop=1, **kwargs)
        return data

def VGGNetBody(net, from_layer, need_fc=True, fully_conv=False, reduced=False,
        dilated=False, nopool=False, dropout=True, _layers=[]):
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}

    assert from_layer in net.keys()
    net.conv1_1 = L.Convolution(net[from_layer], num_output=64, pad=1, kernel_size=3, **kwargs)

    net.relu1_1 = L.ReLU(net.conv1_1, in_place=True)
    net.conv1_2 = L.Convolution(net.relu1_1, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.relu1_2 = L.ReLU(net.conv1_2, in_place=True)

    if nopool:
        name = 'conv1_3'
        net[name] = L.Convolution(net.relu1_2, num_output=64, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool1'
        net.pool1 = L.Pooling(net.relu1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2_1 = L.Convolution(net[name], num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_1 = L.ReLU(net.conv2_1, in_place=True)
    net.conv2_2 = L.Convolution(net.relu2_1, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_2 = L.ReLU(net.conv2_2, in_place=True)

    if nopool:
        name = 'conv2_3'
        net[name] = L.Convolution(net.relu2_2, num_output=128, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool2'
        net[name] = L.Pooling(net.relu2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv3_1 = L.Convolution(net[name], num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_1 = L.ReLU(net.conv3_1, in_place=True)
    net.conv3_2 = L.Convolution(net.relu3_1, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_2 = L.ReLU(net.conv3_2, in_place=True)
    net.conv3_3 = L.Convolution(net.relu3_2, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_3 = L.ReLU(net.conv3_3, in_place=True)

    if nopool:
        name = 'conv3_4'
        net[name] = L.Convolution(net.relu3_3, num_output=256, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool3'
        net[name] = L.Pooling(net.relu3_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv4_1 = L.Convolution(net[name], num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_1 = L.ReLU(net.conv4_1, in_place=True)
    net.conv4_2 = L.Convolution(net.relu4_1, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_2 = L.ReLU(net.conv4_2, in_place=True)
    net.conv4_3 = L.Convolution(net.relu4_2, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_3 = L.ReLU(net.conv4_3, in_place=True)

    if nopool:
        name = 'conv4_4'
        net[name] = L.Convolution(net.relu4_3, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool4'
        net[name] = L.Pooling(net.relu4_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv5_1 = L.Convolution(net[name], num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_1 = L.ReLU(net.conv5_1, in_place=True)
    net.conv5_2 = L.Convolution(net.relu5_1, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_2 = L.ReLU(net.conv5_2, in_place=True)
    net.conv5_3 = L.Convolution(net.relu5_2, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_3 = L.ReLU(net.conv5_3, in_place=True)

    if need_fc:
        if dilated:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(net.relu5_3, num_output=512, pad=1, kernel_size=3, stride=1, **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=1)
        else:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(net.relu5_3, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        if fully_conv:
            if dilated:
                if reduced:
                    net.fc6 = L.Convolution(net[name], num_output=1024, pad=6, kernel_size=3, dilation=6, **kwargs)
                else:
                    net.fc6 = L.Convolution(net[name], num_output=4096, pad=6, kernel_size=7, dilation=2, **kwargs)
            else:
                if reduced:
                    net.fc6 = L.Convolution(net[name], num_output=1024, pad=3, kernel_size=3, dilation=3, **kwargs)
                else:
                    net.fc6 = L.Convolution(net[name], num_output=4096, pad=3, kernel_size=7, **kwargs)

            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)

            if reduced:
                net.fc7 = L.Convolution(net.relu6, num_output=1024, kernel_size=1, **kwargs)
            else:
                net.fc7 = L.Convolution(net.relu6, num_output=4096, kernel_size=1, **kwargs)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)
        else:
            net.fc6 = L.InnerProduct(net.pool5, num_output=4096)
            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)
            net.fc7 = L.InnerProduct(net.relu6, num_output=4096)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)

    # Update  layers.
    kwargs['param'] = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    layers = net.keys()
    for _layer in _layers:
        if _layer in layers:
            net.update(_layer, kwargs)

    return net

def VGGNetBodyBN(net, from_layer, need_fc=True, fully_conv=False, reduced=False,
        dilated=False, nopool=False, dropout=True, _layers=[]):

    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}
    # parameters for batchnorm layer.
    bn_kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        'eps': 0.001,
        }
    # parameters for scale bias layer after batchnorm.

    sb_kwargs = {
          'bias_term': True,
          'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=0)],
          'filler': dict(type='constant', value=1.0),
          'bias_filler': dict(type='constant', value=0.0),
          }

    assert from_layer in net.keys()
    net.conv1_1 = L.Convolution(net[from_layer], num_output=64, pad=1, kernel_size=3, **kwargs)

    net.conv1_1_bn = L.BatchNorm(net.conv1_1, in_place=True, **bn_kwargs)
    net.conv1_1_sb = L.Scale(net.conv1_1_bn, in_place=True, **sb_kwargs)
    net.relu1_1 = L.ReLU(net.conv1_1, in_place=True)

    net.conv1_2 = L.Convolution(net.relu1_1, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.conv1_2_bn = L.BatchNorm(net.conv1_2, in_place=True, **bn_kwargs)
    net.conv1_2_sb = L.Scale(net.conv1_2_bn, in_place=True, **sb_kwargs)

    net.relu1_2 = L.ReLU(net.conv1_2, in_place=True)

    if nopool:
        name = 'conv1_3'
        net[name] = L.Convolution(net.relu1_2, num_output=32, pad=1, kernel_size=3, stride=2, **kwargs)
        net.conv1_3_bn = L.BatchNorm(net.conv1_3, in_place=True, **bn_kwargs)
        net.conv1_3_sb = L.Scale(net.conv1_3_bn, in_place=True, **sb_kwargs)
    else:
        name = 'pool1'
        net.pool1 = L.Pooling(net.relu1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2_1 = L.Convolution(net[name], num_output=32, pad=1, kernel_size=3, **kwargs)
    net.conv2_1_bn = L.BatchNorm(net.conv2_1, in_place=True, **bn_kwargs)
    net.conv2_1_sb = L.Scale(net.conv2_1_bn, in_place=True, **sb_kwargs)
    net.relu2_1 = L.ReLU(net.conv2_1, in_place=True)

    net.conv2_2 = L.Convolution(net.relu2_1, num_output=32, pad=1, kernel_size=3, **kwargs)
    net.conv2_2_bn = L.BatchNorm(net.conv2_2, in_place=True, **bn_kwargs)
    net.conv2_2_sb = L.Scale(net.conv2_2_bn, in_place=True, **sb_kwargs)
    net.relu2_2 = L.ReLU(net.conv2_2, in_place=True)

    if nopool:
        name = 'conv2_3'
        net[name] = L.Convolution(net.relu2_2, num_output=64, pad=1, kernel_size=3, stride=2, **kwargs)
        net.conv2_3_bn = L.BatchNorm(net.conv2_3, in_place=True, **bn_kwargs)
        net.conv2_3_sb = L.Scale(net.conv2_3_bn, in_place=True, **sb_kwargs)
    else:
        name = 'pool2'
        net[name] = L.Pooling(net.relu2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv3_1 = L.Convolution(net[name], num_output=64, pad=1, kernel_size=3, **kwargs)
    net.conv3_1_bn = L.BatchNorm(net.conv3_1, in_place=True, **bn_kwargs)
    net.conv3_1_sb = L.Scale(net.conv3_1_bn, in_place=True, **sb_kwargs)
    net.relu3_1 = L.ReLU(net.conv3_1, in_place=True)
    net.conv3_2 = L.Convolution(net.relu3_1, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.conv3_2_bn = L.BatchNorm(net.conv3_2, in_place=True, **bn_kwargs)
    net.conv3_2_sb = L.Scale(net.conv3_2_bn, in_place=True, **sb_kwargs)
    net.relu3_2 = L.ReLU(net.conv3_2, in_place=True)
    net.conv3_3 = L.Convolution(net.relu3_2, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.conv3_3_bn = L.BatchNorm(net.conv3_3, in_place=True, **bn_kwargs)
    net.conv3_3_sb = L.Scale(net.conv3_3_bn, in_place=True, **sb_kwargs)
    net.relu3_3 = L.ReLU(net.conv3_3, in_place=True)

    if nopool:
        name = 'conv3_4'
        net[name] = L.Convolution(net.relu3_3, num_output=64, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool3'
        net[name] = L.Pooling(net.relu3_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv4_1 = L.Convolution(net[name], num_output=128, pad=1, kernel_size=3, **kwargs)
    net.conv4_1_bn = L.BatchNorm(net.conv4_1, in_place=True, **bn_kwargs)
    net.conv4_1_sb = L.Scale(net.conv4_1_bn, in_place=True, **sb_kwargs)
    net.relu4_1 = L.ReLU(net.conv4_1, in_place=True)
    net.conv4_2 = L.Convolution(net.relu4_1, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.conv4_2_bn = L.BatchNorm(net.conv4_2, in_place=True, **bn_kwargs)
    net.conv4_2_sb = L.Scale(net.conv4_2_bn, in_place=True, **sb_kwargs)
    net.relu4_2 = L.ReLU(net.conv4_2, in_place=True)
    net.conv4_3 = L.Convolution(net.relu4_2, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.conv4_3_bn = L.BatchNorm(net.conv4_3, in_place=True, **bn_kwargs)
    net.conv4_3_sb = L.Scale(net.conv4_3_bn, in_place=True, **sb_kwargs)
    net.relu4_3 = L.ReLU(net.conv4_3, in_place=True)

    if nopool:
        name = 'conv4_4'
        net[name] = L.Convolution(net.relu4_3, num_output=128, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool4'
        net[name] = L.Pooling(net.relu4_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv5_1 = L.Convolution(net[name], num_output=128, pad=1, kernel_size=3, **kwargs)
    net.conv5_1_bn = L.BatchNorm(net.conv5_1, in_place=True, **bn_kwargs)
    net.conv5_1_sb = L.Scale(net.conv5_1_bn, in_place=True, **sb_kwargs)
    net.relu5_1 = L.ReLU(net.conv5_1, in_place=True)
    net.conv5_2 = L.Convolution(net.relu5_1, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.conv5_2_bn = L.BatchNorm(net.conv5_2, in_place=True, **bn_kwargs)
    net.conv5_2_sb = L.Scale(net.conv5_2_bn, in_place=True, **sb_kwargs)
    net.relu5_2 = L.ReLU(net.conv5_2, in_place=True)
    net.conv5_3 = L.Convolution(net.relu5_2, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.conv5_3_bn = L.BatchNorm(net.conv5_3, in_place=True, **bn_kwargs)
    net.conv5_3_sb = L.Scale(net.conv5_3_bn, in_place=True, **sb_kwargs)
    net.relu5_3 = L.ReLU(net.conv5_3, in_place=True)

    if need_fc:
        if dilated:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(net.relu5_3, num_output=128, pad=1, kernel_size=3, stride=1, **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=1)
        else:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(net.relu5_3, num_output=128, pad=1, kernel_size=3, stride=2, **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        if fully_conv:
            if dilated:
                if reduced:
                    net.fc6 = L.Convolution(net[name], num_output=1024, pad=6, kernel_size=3, dilation=6, **kwargs)
                    net.fc6_bn = L.BatchNorm(net.fc6, in_place=True, **bn_kwargs)
                    net.fc6_sb = L.Scale(net.fc6_bn, in_place=True, **sb_kwargs)
                else:
                    net.fc6 = L.Convolution(net[name], num_output=2048, pad=6, kernel_size=7, dilation=2, **kwargs)
            else:
                if reduced:
                    net.fc6 = L.Convolution(net[name], num_output=1024, pad=3, kernel_size=3, dilation=3, **kwargs)
                    net.fc6_bn = L.BatchNorm(net.fc6, in_place=True, **bn_kwargs)
                    net.fc6_sb = L.Scale(net.fc6_bn, in_place=True, **sb_kwargs)
                else:
                    net.fc6 = L.Convolution(net[name], num_output=2048, pad=3, kernel_size=7, **kwargs)

            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)

            if reduced:
                net.fc7 = L.Convolution(net.relu6, num_output=1024, kernel_size=1, **kwargs)
                net.fc7_bn = L.BatchNorm(net.fc7, in_place=True, **bn_kwargs)
                net.fc7_sb = L.Scale(net.fc7_bn, in_place=True, **sb_kwargs)
            else:
                net.fc7 = L.Convolution(net.relu6, num_output=2048, kernel_size=1, **kwargs)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)
        else:
            net.fc6 = L.InnerProduct(net.pool5, num_output=2048)
            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)
            net.fc7 = L.InnerProduct(net.relu6, num_output=2048)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)

    # Update  layers.
    kwargs['param'] = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    layers = net.keys()
    for _layer in _layers:
        if _layer in layers:
            net.update(_layer, kwargs)

    return net

def VGGNetBodyBNkernel1(net, from_layer, need_fc=True, fully_conv=False, reduced=False,
        dilated=False, nopool=False, dropout=True, _layers=[]):

    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}
    # parameters for batchnorm layer.
    bn_kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        'eps': 0.001,
        }
    # parameters for scale bias layer after batchnorm.

    sb_kwargs = {
          'bias_term': True,
          'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=0)],
          'filler': dict(type='constant', value=1.0),
          'bias_filler': dict(type='constant', value=0.0),
          }

    assert from_layer in net.keys()
    net.conv1_1 = L.Convolution(net[from_layer], num_output=64, pad=1, kernel_size=3, **kwargs)

    net.conv1_1_bn = L.BatchNorm(net.conv1_1, in_place=True, **bn_kwargs)
    net.conv1_1_sb = L.Scale(net.conv1_1_bn, in_place=True, **sb_kwargs)
    net.relu1_1 = L.ReLU(net.conv1_1, in_place=True)

    net.conv1_2 = L.Convolution(net.relu1_1, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.conv1_2_bn = L.BatchNorm(net.conv1_2, in_place=True, **bn_kwargs)
    net.conv1_2_sb = L.Scale(net.conv1_2_bn, in_place=True, **sb_kwargs)

    net.relu1_2 = L.ReLU(net.conv1_2, in_place=True)

    if nopool:
        name = 'conv1_3'
        net[name] = L.Convolution(net.relu1_2, num_output=64, pad=1, kernel_size=3, stride=2, **kwargs)
        net.conv1_3_bn = L.BatchNorm(net.conv1_3, in_place=True, **bn_kwargs)
        net.conv1_3_sb = L.Scale(net.conv1_3_bn, in_place=True, **sb_kwargs)
    else:
        name = 'pool1'
        net.pool1 = L.Pooling(net.relu1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2_1 = L.Convolution(net[name], num_output=64, pad=1, kernel_size=3, **kwargs)
    net.conv2_1_bn = L.BatchNorm(net.conv2_1, in_place=True, **bn_kwargs)
    net.conv2_1_sb = L.Scale(net.conv2_1_bn, in_place=True, **sb_kwargs)
    net.relu2_1 = L.ReLU(net.conv2_1, in_place=True)

    net.conv2_2 = L.Convolution(net.relu2_1, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.conv2_2_bn = L.BatchNorm(net.conv2_2, in_place=True, **bn_kwargs)
    net.conv2_2_sb = L.Scale(net.conv2_2_bn, in_place=True, **sb_kwargs)
    net.relu2_2 = L.ReLU(net.conv2_2, in_place=True)

    if nopool:
        name = 'conv2_3'
        net[name] = L.Convolution(net.relu2_2, num_output=128, pad=1, kernel_size=3, stride=2, **kwargs)
        net.conv2_3_bn = L.BatchNorm(net.conv2_3, in_place=True, **bn_kwargs)
        net.conv2_3_sb = L.Scale(net.conv2_3_bn, in_place=True, **sb_kwargs)
    else:
        name = 'pool2'
        net[name] = L.Pooling(net.relu2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv3_1 = L.Convolution(net[name], num_output=128, pad=1, kernel_size=3, **kwargs)
    net.conv3_1_bn = L.BatchNorm(net.conv3_1, in_place=True, **bn_kwargs)
    net.conv3_1_sb = L.Scale(net.conv3_1_bn, in_place=True, **sb_kwargs)
    net.relu3_1 = L.ReLU(net.conv3_1, in_place=True)
    net.conv3_2 = L.Convolution(net.relu3_1, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.conv3_2_bn = L.BatchNorm(net.conv3_2, in_place=True, **bn_kwargs)
    net.conv3_2_sb = L.Scale(net.conv3_2_bn, in_place=True, **sb_kwargs)
    net.relu3_2 = L.ReLU(net.conv3_2, in_place=True)
    net.conv3_3 = L.Convolution(net.relu3_2, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.conv3_3_bn = L.BatchNorm(net.conv3_3, in_place=True, **bn_kwargs)
    net.conv3_3_sb = L.Scale(net.conv3_3_bn, in_place=True, **sb_kwargs)
    net.relu3_3 = L.ReLU(net.conv3_3, in_place=True)

    if nopool:
        name = 'conv3_4'
        net[name] = L.Convolution(net.relu3_3, num_output=128, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool3'
        net[name] = L.Pooling(net.relu3_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv4_1 = L.Convolution(net[name], num_output=256, pad=1, kernel_size=3, **kwargs)
    net.conv4_1_bn = L.BatchNorm(net.conv4_1, in_place=True, **bn_kwargs)
    net.conv4_1_sb = L.Scale(net.conv4_1_bn, in_place=True, **sb_kwargs)
    net.relu4_1 = L.ReLU(net.conv4_1, in_place=True)
    net.conv4_2 = L.Convolution(net.relu4_1, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.conv4_2_bn = L.BatchNorm(net.conv4_2, in_place=True, **bn_kwargs)
    net.conv4_2_sb = L.Scale(net.conv4_2_bn, in_place=True, **sb_kwargs)
    net.relu4_2 = L.ReLU(net.conv4_2, in_place=True)
    net.conv4_3 = L.Convolution(net.relu4_2, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.conv4_3_bn = L.BatchNorm(net.conv4_3, in_place=True, **bn_kwargs)
    net.conv4_3_sb = L.Scale(net.conv4_3_bn, in_place=True, **sb_kwargs)
    net.relu4_3 = L.ReLU(net.conv4_3, in_place=True)

    if nopool:
        name = 'conv4_4'
        net[name] = L.Convolution(net.relu4_3, num_output=256, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool4'
        net[name] = L.Pooling(net.relu4_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv5_1 = L.Convolution(net[name], num_output=256, pad=1, kernel_size=3, **kwargs)
    net.conv5_1_bn = L.BatchNorm(net.conv5_1, in_place=True, **bn_kwargs)
    net.conv5_1_sb = L.Scale(net.conv5_1_bn, in_place=True, **sb_kwargs)
    net.relu5_1 = L.ReLU(net.conv5_1, in_place=True)
    net.conv5_2 = L.Convolution(net.relu5_1, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.conv5_2_bn = L.BatchNorm(net.conv5_2, in_place=True, **bn_kwargs)
    net.conv5_2_sb = L.Scale(net.conv5_2_bn, in_place=True, **sb_kwargs)
    net.relu5_2 = L.ReLU(net.conv5_2, in_place=True)
    net.conv5_3 = L.Convolution(net.relu5_2, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.conv5_3_bn = L.BatchNorm(net.conv5_3, in_place=True, **bn_kwargs)
    net.conv5_3_sb = L.Scale(net.conv5_3_bn, in_place=True, **sb_kwargs)
    net.relu5_3 = L.ReLU(net.conv5_3, in_place=True)

    if need_fc:
        if dilated:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(net.relu5_3, num_output=256, pad=1, kernel_size=3, stride=1, **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=1)
        else:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(net.relu5_3, num_output=256, pad=1, kernel_size=3, stride=2, **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        if fully_conv:
            if dilated:
                if reduced:
                    net.fc6 = L.Convolution(net[name], num_output=1024, pad=6, kernel_size=1, dilation=6, **kwargs)
                    net.fc6_bn = L.BatchNorm(net.fc6, in_place=True, **bn_kwargs)
                    net.fc6_sb = L.Scale(net.fc6_bn, in_place=True, **sb_kwargs)
                else:
                    net.fc6 = L.Convolution(net[name], num_output=2048, pad=6, kernel_size=7, dilation=2, **kwargs)
            else:
                if reduced:
                    net.fc6 = L.Convolution(net[name], num_output=1024, pad=3, kernel_size=1, dilation=3, **kwargs)
                    net.fc6_bn = L.BatchNorm(net.fc6, in_place=True, **bn_kwargs)
                    net.fc6_sb = L.Scale(net.fc6_bn, in_place=True, **sb_kwargs)
                else:
                    net.fc6 = L.Convolution(net[name], num_output=2048, pad=3, kernel_size=7, **kwargs)

            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)

            if reduced:
                net.fc7 = L.Convolution(net.relu6, num_output=1024, kernel_size=1, **kwargs)
                net.fc7_bn = L.BatchNorm(net.fc7, in_place=True, **bn_kwargs)
                net.fc7_sb = L.Scale(net.fc7_bn, in_place=True, **sb_kwargs)
            else:
                net.fc7 = L.Convolution(net.relu6, num_output=2048, kernel_size=1, **kwargs)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)
        else:
            net.fc6 = L.InnerProduct(net.pool5, num_output=2048)
            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)
            net.fc7 = L.InnerProduct(net.relu6, num_output=2048)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)

    # Update  layers.
    kwargs['param'] = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    layers = net.keys()
    for _layer in _layers:
        if _layer in layers:
            net.update(_layer, kwargs)

    return net

def ResNet152Body(net, from_layer, use_pool5=True):
    conv_prefix = ''
    conv_postfix = ''
    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''
    ConvBNLayer(net, from_layer, 'conv1', use_bn=True, use_relu=True,
        num_output=64, kernel_size=7, pad=3, stride=2,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)

    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    ResBody(net, 'pool1', '2a', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=True)
    ResBody(net, 'res2a', '2b', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False)
    ResBody(net, 'res2b', '2c', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False)

    ResBody(net, 'res2c', '3a', out2a=128, out2b=128, out2c=512, stride=2, use_branch1=True)

    from_layer = 'res3a'
    for i in xrange(1, 8):
      block_name = '3b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=128, out2b=128, out2c=512, stride=1, use_branch1=False)
      from_layer = 'res{}'.format(block_name)

    ResBody(net, from_layer, '4a', out2a=256, out2b=256, out2c=1024, stride=2, use_branch1=True)

    from_layer = 'res4a'
    for i in xrange(1, 36):
      block_name = '4b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=256, out2b=256, out2c=1024, stride=1, use_branch1=False)
      from_layer = 'res{}'.format(block_name)

    ResBody(net, from_layer, '5a', out2a=512, out2b=512, out2c=2048, stride=2, use_branch1=True)
    ResBody(net, 'res5a', '5b', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False)
    ResBody(net, 'res5b', '5c', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False)

    if use_pool5:
      net.pool5 = L.Pooling(net.res5c, pool=P.Pooling.AVE, global_pooling=True)

    return net



def InceptionV3Body(net, from_layer, output_pred=False):
  # scale is fixed to 1, thus we ignore it.
  use_scale = False

  out_layer = 'conv'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=32, kernel_size=3, pad=0, stride=2, use_scale=use_scale)
  from_layer = out_layer

  out_layer = 'conv_1'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=32, kernel_size=3, pad=0, stride=1, use_scale=use_scale)
  from_layer = out_layer

  out_layer = 'conv_2'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=64, kernel_size=3, pad=1, stride=1, use_scale=use_scale)
  from_layer = out_layer

  out_layer = 'pool'
  net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
      kernel_size=3, stride=2, pad=0)
  from_layer = out_layer

  out_layer = 'conv_3'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=80, kernel_size=1, pad=0, stride=1, use_scale=use_scale)
  from_layer = out_layer

  out_layer = 'conv_4'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=192, kernel_size=3, pad=0, stride=1, use_scale=use_scale)
  from_layer = out_layer

  out_layer = 'pool_1'
  net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
      kernel_size=3, stride=2, pad=0)
  from_layer = out_layer

  # inceptions with 1x1, 3x3, 5x5 convolutions
  for inception_id in xrange(0, 3):
    if inception_id == 0:
      out_layer = 'mixed'
      tower_2_conv_num_output = 32
    else:
      out_layer = 'mixed_{}'.format(inception_id)
      tower_2_conv_num_output = 64
    towers = []
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
        ])
    towers.append(tower)
    tower_name = '{}/tower'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=48, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=64, kernel_size=5, pad=2, stride=1),
        ])
    towers.append(tower)
    tower_name = '{}/tower_1'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=96, kernel_size=3, pad=1, stride=1),
        dict(name='conv_2', num_output=96, kernel_size=3, pad=1, stride=1),
        ])
    towers.append(tower)
    tower_name = '{}/tower_2'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='pool', pool=P.Pooling.AVE, kernel_size=3, pad=1, stride=1),
        dict(name='conv', num_output=tower_2_conv_num_output, kernel_size=1, pad=0, stride=1),
        ])
    towers.append(tower)
    out_layer = '{}/join'.format(out_layer)
    net[out_layer] = L.Concat(*towers, axis=1)
    from_layer = out_layer

  # inceptions with 1x1, 3x3(in sequence) convolutions
  out_layer = 'mixed_3'
  towers = []
  tower_name = '{}'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='conv', num_output=384, kernel_size=3, pad=0, stride=2),
      ])
  towers.append(tower)
  tower_name = '{}/tower'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
      dict(name='conv_1', num_output=96, kernel_size=3, pad=1, stride=1),
      dict(name='conv_2', num_output=96, kernel_size=3, pad=0, stride=2),
      ])
  towers.append(tower)
  tower_name = '{}'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='pool', pool=P.Pooling.MAX, kernel_size=3, pad=0, stride=2),
      ])
  towers.append(tower)
  out_layer = '{}/join'.format(out_layer)
  net[out_layer] = L.Concat(*towers, axis=1)
  from_layer = out_layer

  # inceptions with 1x1, 7x1, 1x7 convolutions
  for inception_id in xrange(4, 8):
    if inception_id == 4:
      num_output = 128
    elif inception_id == 5 or inception_id == 6:
      num_output = 160
    elif inception_id == 7:
      num_output = 192
    out_layer = 'mixed_{}'.format(inception_id)
    towers = []
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ])
    towers.append(tower)
    tower_name = '{}/tower'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=num_output, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
        dict(name='conv_2', num_output=192, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
        ])
    towers.append(tower)
    tower_name = '{}/tower_1'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=num_output, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
        dict(name='conv_2', num_output=num_output, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
        dict(name='conv_3', num_output=num_output, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
        dict(name='conv_4', num_output=192, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
        ])
    towers.append(tower)
    tower_name = '{}/tower_2'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='pool', pool=P.Pooling.AVE, kernel_size=3, pad=1, stride=1),
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ])
    towers.append(tower)
    out_layer = '{}/join'.format(out_layer)
    net[out_layer] = L.Concat(*towers, axis=1)
    from_layer = out_layer

  # inceptions with 1x1, 3x3, 1x7, 7x1 filters
  out_layer = 'mixed_8'
  towers = []
  tower_name = '{}/tower'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
      dict(name='conv_1', num_output=320, kernel_size=3, pad=0, stride=2),
      ])
  towers.append(tower)
  tower_name = '{}/tower_1'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
      dict(name='conv_1', num_output=192, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
      dict(name='conv_2', num_output=192, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
      dict(name='conv_3', num_output=192, kernel_size=3, pad=0, stride=2),
      ])
  towers.append(tower)
  tower_name = '{}'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='pool', pool=P.Pooling.MAX, kernel_size=3, pad=0, stride=2),
      ])
  towers.append(tower)
  out_layer = '{}/join'.format(out_layer)
  net[out_layer] = L.Concat(*towers, axis=1)
  from_layer = out_layer

  for inception_id in xrange(9, 11):
    num_output = 384
    num_output2 = 448
    if inception_id == 9:
      pool = P.Pooling.AVE
    else:
      pool = P.Pooling.MAX
    out_layer = 'mixed_{}'.format(inception_id)
    towers = []
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=320, kernel_size=1, pad=0, stride=1),
        ])
    towers.append(tower)

    tower_name = '{}/tower'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
        ])
    subtowers = []
    subtower_name = '{}/mixed'.format(tower_name)
    subtower = InceptionTower(net, '{}/conv'.format(tower_name), subtower_name, [
        dict(name='conv', num_output=num_output, kernel_size=[1, 3], pad=[0, 1], stride=[1, 1]),
        ])
    subtowers.append(subtower)
    subtower = InceptionTower(net, '{}/conv'.format(tower_name), subtower_name, [
        dict(name='conv_1', num_output=num_output, kernel_size=[3, 1], pad=[1, 0], stride=[1, 1]),
        ])
    subtowers.append(subtower)
    net[subtower_name] = L.Concat(*subtowers, axis=1)
    towers.append(net[subtower_name])

    tower_name = '{}/tower_1'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=num_output2, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=num_output, kernel_size=3, pad=1, stride=1),
        ])
    subtowers = []
    subtower_name = '{}/mixed'.format(tower_name)
    subtower = InceptionTower(net, '{}/conv_1'.format(tower_name), subtower_name, [
        dict(name='conv', num_output=num_output, kernel_size=[1, 3], pad=[0, 1], stride=[1, 1]),
        ])
    subtowers.append(subtower)
    subtower = InceptionTower(net, '{}/conv_1'.format(tower_name), subtower_name, [
        dict(name='conv_1', num_output=num_output, kernel_size=[3, 1], pad=[1, 0], stride=[1, 1]),
        ])
    subtowers.append(subtower)
    net[subtower_name] = L.Concat(*subtowers, axis=1)
    towers.append(net[subtower_name])

    tower_name = '{}/tower_2'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='pool', pool=pool, kernel_size=3, pad=1, stride=1),
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ])
    towers.append(tower)
    out_layer = '{}/join'.format(out_layer)
    net[out_layer] = L.Concat(*towers, axis=1)
    from_layer = out_layer

  if output_pred:
    net.pool_3 = L.Pooling(net[from_layer], pool=P.Pooling.AVE, kernel_size=8, pad=0, stride=1)
    net.softmax = L.InnerProduct(net.pool_3, num_output=1008)
    net.softmax_prob = L.Softmax(net.softmax)

  return net

def CreateMultiBoxHead(net, data_layer="data", num_classes=[], from_layers=[],
        use_objectness=False, normalizations=[], use_batchnorm=True,
        min_sizes=[], max_sizes=[], prior_variance = [0.1],
        aspect_ratios=[], share_location=True, flip=True, clip=True,
        inter_layer_depth=0, kernel_size=1, pad=0, conf_postfix='', loc_postfix=''):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    if normalizations:
        assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"
    assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
    if max_sizes:
        assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers"

    num = len(from_layers)
    priorbox_layers = []
    loc_layers = []
    conf_layers = []
    objectness_layers = []
    for i in range(0, num):
        from_layer = from_layers[i]

        # Get the normalize value.
        if normalizations:
            if normalizations[i] != -1:
                norm_name = "{}_norm".format(from_layer)
                net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalizations[i]),
                    across_spatial=False, channel_shared=False)
                from_layer = norm_name

        # Add intermediate layers.
        if inter_layer_depth > 0:
            inter_name = "{}_inter".format(from_layer)
            ConvBNLayer(net, from_layer, inter_name, use_bn=use_batchnorm, use_relu=True,
                num_output=inter_layer_depth, kernel_size=3, pad=1, stride=1)
            from_layer = inter_name

        # Estimate number of priors per location given provided parameters.
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        if max_sizes and max_sizes[i]:
            num_priors_per_location = 2 + len(aspect_ratio)
        else:
            num_priors_per_location = 1 + len(aspect_ratio)
        if flip:
            num_priors_per_location += len(aspect_ratio)

        # Create location prediction layer.
        name = "{}_mbox_loc{}".format(from_layer, loc_postfix)
        num_loc_output = num_priors_per_location * 4;
        if not share_location:
            num_loc_output *= num_classes
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False,
            num_output=num_loc_output, kernel_size=kernel_size, pad=pad, stride=1)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layers.append(net[flatten_name])

        # Create confidence prediction layer.
        name = "{}_mbox_conf{}".format(from_layer, conf_postfix)
        num_conf_output = num_priors_per_location * num_classes;
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False,
            num_output=num_conf_output, kernel_size=kernel_size, pad=pad, stride=1)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layers.append(net[flatten_name])

        # Create prior generation layer.
        name = "{}_mbox_priorbox".format(from_layer)
        if max_sizes and max_sizes[i]:
            if aspect_ratio:
                net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes[i], max_size=max_sizes[i],
                    aspect_ratio=aspect_ratio, flip=flip, clip=clip, variance=prior_variance)
            else:
                net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes[i], max_size=max_sizes[i],
                    clip=clip, variance=prior_variance)
        else:
            if aspect_ratio:
                net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes[i],
                    aspect_ratio=aspect_ratio, flip=flip, clip=clip, variance=prior_variance)
            else:
                net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes[i],
                    clip=clip, variance=prior_variance)
        priorbox_layers.append(net[name])

        # Create objectness prediction layer.
        if use_objectness:
            name = "{}_mbox_objectness".format(from_layer)
            num_obj_output = num_priors_per_location * 2;
            ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False,
                num_output=num_obj_output, kernel_size=kernel_size, pad=pad, stride=1)
            permute_name = "{}_perm".format(name)
            net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
            flatten_name = "{}_flat".format(name)
            net[flatten_name] = L.Flatten(net[permute_name], axis=1)
            objectness_layers.append(net[flatten_name])

    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    name = "mbox_loc"
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_conf"
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_priorbox"
    net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])
    if use_objectness:
        name = "mbox_objectness"
        net[name] = L.Concat(*objectness_layers, axis=1)
        mbox_layers.append(net[name])

    return mbox_layers


def SqueezeNetBody(net, from_layer, for_HG_module=False):
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}

    assert from_layer in net.keys()
    ConvBNLayer(net, from_layer, 'conv1', use_bn=True, use_relu=True,
        num_output=64, kernel_size=3, pad=0, stride=1,for_HG_module=for_HG_module)

    net.pool1s = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    #fire2
    ConvBNLayer(net, 'pool1s', 'fire2/squeeze1x1', use_bn=True, use_relu=True,
        num_output=16, kernel_size=1, pad=0, stride=1,for_HG_module=for_HG_module)

    expand_fire2 = []
    ConvBNLayer(net, 'fire2/squeeze1x1', 'fire2/expand1x1', use_bn=True, use_relu=True,
        num_output=64, kernel_size=1, pad=0, stride=1,for_HG_module=for_HG_module)

    ConvBNLayer(net, 'fire2/squeeze1x1', 'fire2/expand3x3', use_bn=True, use_relu=True,
        num_output=64, kernel_size=3, pad=1, stride=1,for_HG_module=for_HG_module)

    expand_fire2.append(net['fire2/expand1x1'])
    expand_fire2.append(net['fire2/expand3x3'])
    net['fire2/concat'] = L.Concat(*expand_fire2, axis=1)
    #fire3
    ConvBNLayer(net, 'fire2/concat', 'fire3/squeeze1x1', use_bn=True, use_relu=True,
        num_output=16, kernel_size=1, pad=0, stride=1,for_HG_module=for_HG_module)

    expand_fire3 = []
    ConvBNLayer(net, 'fire3/squeeze1x1', 'fire3/expand1x1', use_bn=True, use_relu=True,
        num_output=64, kernel_size=1, pad=0, stride=1,for_HG_module=for_HG_module)

    ConvBNLayer(net, 'fire3/squeeze1x1', 'fire3/expand3x3', use_bn=True, use_relu=True,
        num_output=64, kernel_size=3, pad=1, stride=1,for_HG_module=for_HG_module)

    expand_fire3.append(net['fire3/expand1x1'])
    expand_fire3.append(net['fire3/expand3x3'])
    net['fire3/concat'] = L.Concat(*expand_fire3, axis=1)
    #pool3
    net.pool3 = L.Pooling(net['fire3/concat'], pool=P.Pooling.MAX, kernel_size=3, stride=2)
    #fire4
    ConvBNLayer(net, 'pool3', 'fire4/squeeze1x1', use_bn=True, use_relu=True,
        num_output=32, kernel_size=1, pad=0, stride=1,for_HG_module=for_HG_module)

    expand_fire4 = []
    ConvBNLayer(net, 'fire4/squeeze1x1', 'fire4/expand1x1', use_bn=True, use_relu=True,
        num_output=128, kernel_size=1, pad=0, stride=1,for_HG_module=for_HG_module)

    ConvBNLayer(net, 'fire4/squeeze1x1', 'fire4/expand3x3', use_bn=True, use_relu=True,
        num_output=128, kernel_size=3, pad=1, stride=1,for_HG_module=for_HG_module)

    expand_fire4.append(net['fire4/expand1x1'])
    expand_fire4.append(net['fire4/expand3x3'])
    net['fire4/concat'] = L.Concat(*expand_fire4, axis=1)
    #fire5
    ConvBNLayer(net, 'fire4/concat', 'fire5/squeeze1x1', use_bn=True, use_relu=True,
        num_output=32, kernel_size=1, pad=0, stride=1,for_HG_module=for_HG_module)

    expand_fire5 = []
    ConvBNLayer(net, 'fire5/squeeze1x1', 'fire5/expand1x1', use_bn=True, use_relu=True,
        num_output=128, kernel_size=1, pad=0, stride=1,for_HG_module=for_HG_module)

    ConvBNLayer(net, 'fire5/squeeze1x1', 'fire5/expand3x3', use_bn=True, use_relu=True,
        num_output=128, kernel_size=3, pad=1, stride=1,for_HG_module=for_HG_module)
    expand_fire5.append(net['fire5/expand1x1'])
    expand_fire5.append(net['fire5/expand3x3'])
    net['fire5/concat'] = L.Concat(*expand_fire5, axis=1)
    #pool5
    net.pool5 = L.Pooling(net['fire5/concat'], pool=P.Pooling.MAX, kernel_size=3, stride=2)
    #fire6
    ConvBNLayer(net, 'pool5', 'fire6/squeeze1x1', use_bn=True, use_relu=True,
        num_output=48, kernel_size=1, pad=0, stride=1,for_HG_module=for_HG_module)

    expand_fire6 = []
    ConvBNLayer(net, 'fire6/squeeze1x1', 'fire6/expand1x1', use_bn=True, use_relu=True,
        num_output=192, kernel_size=1, pad=0, stride=1,for_HG_module=for_HG_module)

    ConvBNLayer(net, 'fire6/squeeze1x1', 'fire6/expand3x3', use_bn=True, use_relu=True,
        num_output=192, kernel_size=3, pad=1, stride=1,for_HG_module=for_HG_module)

    expand_fire6.append(net['fire6/expand1x1'])
    expand_fire6.append(net['fire6/expand3x3'])
    net['fire6/concat'] = L.Concat(*expand_fire6, axis=1)
    #fire7
    ConvBNLayer(net, 'fire6/concat', 'fire7/squeeze1x1', use_bn=True, use_relu=True,
        num_output=48, kernel_size=1, pad=0, stride=1,for_HG_module=for_HG_module)

    expand_fire7 = []
    ConvBNLayer(net, 'fire7/squeeze1x1', 'fire7/expand1x1', use_bn=True, use_relu=True,
        num_output=192, kernel_size=1, pad=0, stride=1,for_HG_module=for_HG_module)

    ConvBNLayer(net, 'fire7/squeeze1x1', 'fire7/expand3x3', use_bn=True, use_relu=True,
        num_output=192, kernel_size=3, pad=1, stride=1,for_HG_module=for_HG_module)
    expand_fire7.append(net['fire7/expand1x1'])
    expand_fire7.append(net['fire7/expand3x3'])
    net['fire7/concat'] = L.Concat(*expand_fire7, axis=1)

    #fire8
    ConvBNLayer(net, 'fire7/concat', 'fire8/squeeze1x1', use_bn=True, use_relu=True,
        num_output=64, kernel_size=1, pad=0, stride=1,for_HG_module=for_HG_module)

    expand_fire8 = []
    ConvBNLayer(net, 'fire8/squeeze1x1', 'fire8/expand1x1', use_bn=True, use_relu=True,
        num_output=256, kernel_size=1, pad=0, stride=1,for_HG_module=for_HG_module)

    ConvBNLayer(net, 'fire8/squeeze1x1', 'fire8/expand3x3', use_bn=True, use_relu=True,
        num_output=256, kernel_size=3, pad=1, stride=1,for_HG_module=for_HG_module)
    expand_fire8.append(net['fire8/expand1x1'])
    expand_fire8.append(net['fire8/expand3x3'])
    net['fire8/concat'] = L.Concat(*expand_fire8, axis=1)

    #fire9
    ConvBNLayer(net, 'fire8/concat', 'fire9/squeeze1x1', use_bn=True, use_relu=True,
        num_output=64, kernel_size=1, pad=0, stride=1,for_HG_module=for_HG_module)

    expand_fire9 = []
    ConvBNLayer(net, 'fire9/squeeze1x1', 'fire9/expand1x1', use_bn=True, use_relu=True,
        num_output=256, kernel_size=1, pad=0, stride=1,for_HG_module=for_HG_module)

    ConvBNLayer(net, 'fire9/squeeze1x1', 'fire9/expand3x3', use_bn=True, use_relu=True,
        num_output=256, kernel_size=3, pad=1, stride=1,for_HG_module=for_HG_module)
    expand_fire9.append(net['fire9/expand1x1'])
    expand_fire9.append(net['fire9/expand3x3'])
    net['fire9/concat'] = L.Concat(*expand_fire9, axis=1)

    #drop9
    net.drop9 = L.Dropout(net['fire9/concat'], dropout_ratio=0.5, in_place=True)
    #conv_final

    ConvBNLayer(net, 'drop9', 'conv10', use_bn=True, use_relu=True,
        num_output=6, kernel_size=1, pad=0, stride=1,for_HG_module=for_HG_module)

    #get theta
    kwargsfile = {
        'param': dict(lr_mult=1, decay_mult=1),
        'weight_filler': dict(type='constant', value=0),
        'bias_term': True,
        'bias_filler': dict(type='file', file='examples/mppp/util/bias_init.txt')
        }
    net.theta = L.InnerProduct(net.conv10,num_output=6,**kwargsfile)

    return net

def HourGlass(net, from_layer, name, n, In, out, final_relu=False):
    ResBody(net, from_layer, '{}_up1'.format(name), out2a=128, out2b=128, out2c=256, stride=1, use_branch1=(In != 256),  for_HG_module=True)
    ResBody(net, 'res{}_up1'.format(name), '{}_up2'.format(name), out2a=128, out2b=128, out2c=256, stride=1, use_branch1=False, for_HG_module=True)
    ResBody(net, 'res{}_up2'.format(name), '{}_up4'.format(name), out2a=out/2, out2b=out/2, out2c=out, stride=1, use_branch1=(out != 256), for_HG_module=True)

    net['{}_pool1'.format(name)] = L.Pooling(net[from_layer], pool=P.Pooling.MAX, kernel_size=2, stride=2)
    ResBody(net, '{}_pool1'.format(name), '{}_low1'.format(name), out2a=128, out2b=128, out2c=256, stride=1, use_branch1=(In != 256), for_HG_module=True)
    ResBody(net, 'res{}_low1'.format(name), '{}_low2'.format(name), out2a=128, out2b=128, out2c=256, stride=1, use_branch1=False, for_HG_module=True)
    ResBody(net, 'res{}_low2'.format(name), '{}_low5'.format(name), out2a=128, out2b=128, out2c=256, stride=1, use_branch1=False, for_HG_module=True)

    if n > 1:
        HourGlass(net,'res{}_low5'.format(name), '{}_low6'.format(name), n-1, 256, out)
        layer_name = '{}_low6'.format(name)
    else:
        ResBody(net, 'res{}_low5'.format(name), '{}_low6'.format(name), out2a=out/2, out2b=out/2, out2c=out, stride=1, use_branch1=(out != 256), for_HG_module=True)
        layer_name = 'res{}_low6'.format(name)
    ResBody(net, layer_name, '{}_low7'.format(name), out2a=out/2, out2b=out/2, out2c=out, stride=1, use_branch1=False, for_HG_module=True)
    
    factor = 2
    kernel = factor
    stride = factor
    pad = 0
    net['{}_Up5'.format(name)] = L.Deconvolution(net['res{}_low7'.format(name)],
        convolution_param=dict(num_output=out, group=out,kernel_size=kernel, stride=stride, pad=pad,
            bias_term=False, weight_filler=dict(type="nearest")), 
        param=[dict(lr_mult=0, decay_mult=0)])

    net[name] = L.Eltwise(net['res{}_up4'.format(name)], net['{}_Up5'.format(name)])
    if final_relu:
      relu_name = '{}_relu'.format(res_name)
      net[relu_name] = L.ReLU(net[res_name], in_place=True)

def HGStacked(net, from_layer):

    conv_prefix = ''
    conv_postfix = ''
    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''
    ConvBNLayer(net, from_layer, 'conv1_b', use_bn=True, use_relu=True,
        num_output=64, kernel_size=7, pad=3, stride=2,for_HG_module=True,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)

    ResBody(net, 'conv1_b', '1', out2a=64, out2b=64, out2c=128, stride=1, use_branch1=True, for_HG_module=True)

    net.pool1 = L.Pooling(net.res1, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    ResBody(net, 'pool1', '4', out2a=64, out2b=64, out2c=128, stride=1, use_branch1=False, for_HG_module=True)
    ResBody(net, 'res4', '5', out2a=64, out2b=64, out2c=128, stride=1, use_branch1=False, for_HG_module=True)
    ResBody(net, 'res5', '6', out2a=128, out2b=128, out2c=256, stride=1, use_branch1=True, for_HG_module=True)

    HourGlass(net, 'res6', 'hg1', n=4, In=256, out=512)

    ConvBNLayer(net, 'hg1', 'linear1', use_bn=True, use_relu=True,
        num_output=512, kernel_size=1, pad=0, stride=1,for_HG_module=True,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    ConvBNLayer(net, 'linear1', 'linear2', use_bn=True, use_relu=True,
        num_output=256, kernel_size=1, pad=0, stride=1,for_HG_module=True,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    ConvBNLayer(net, 'linear2', 'out1', use_bn=False, use_relu=False, use_scale=False,
        num_output=16, kernel_size=1, pad=0, stride=1,for_HG_module=True,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    ConvBNLayer(net, 'out1', 'out1_', use_bn=False, use_relu=False, use_scale=False,
        num_output=384, kernel_size=1, pad=0, stride=1,for_HG_module=True,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    concat = []
    concat.append(net.linear2)
    concat.append(net.pool1)
    net.concat1 = L.Concat(*concat, axis=1)
    ConvBNLayer(net, 'concat1', 'concat1_CN', use_bn=False, use_relu=False, use_scale=False,
        num_output=256+128, kernel_size=1, pad=0, stride=1,for_HG_module=True,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    net.int1 = L.Eltwise(net.concat1_CN, net['out1_'])

    HourGlass(net, 'int1', 'hg2', n=4, In=384, out=512)
    ConvBNLayer(net, 'hg2', 'linear3', use_bn=True, use_relu=True,
        num_output=512, kernel_size=1, pad=0, stride=1,for_HG_module=True,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    ConvBNLayer(net, 'linear3', 'linear4', use_bn=True, use_relu=True,
        num_output=512, kernel_size=1, pad=0, stride=1,for_HG_module=True,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    ConvBNLayer(net, 'linear4', 'output', use_bn=False, use_relu=False, use_scale=False,
        num_output=16, kernel_size=1, pad=0, stride=1,for_HG_module=True,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)

    return net
