from __future__ import print_function
import sys
import numpy as np
import caffe
import os
import os.path as osp
from xml.dom import minidom
from random import shuffle
from threading import Thread
from PIL import Image
caffe.set_mode_gpu()

# set gpu id
caffe.set_device(0)

# model_name: 10v or pvp
# res_name: linear/mlp/...
def test_one_caffemodel(deploy_file, caffemodel, res_dir, mode, res_name):

    if not os.path.exists(res_dir):
        if mode == 'pvp' or mode == '10v':
            os.makedirs(res_dir)      
        else:
            assert 0,"mode != '10v' or 'pvp'"
            
    net = caffe.Net(deploy_file,      # defines the structure of the model
                    caffemodel,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropoutet_mode_cpu()
    
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    
    def load_annotation(index, pascal_root, im_shape):
        """
        This code is borrowed from Ross Girshick's FAST-RCNN code
        (https://github.com/rbgirshick/fast-rcnn).
        It parses the PASCAL .xml metadata files.
        See publication for further details: (http://arxiv.org/abs/1504.08083).
    
        Thanks Ross!
    
        """
    
        filename = osp.join(pascal_root, 'Annotations', index + '.xml')    
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data
    
        with open(filename) as f:
            data = minidom.parseString(f.read())
    
        objs = data.getElementsByTagName('human_roi')
        num_objs = len(objs)
    
        human_boxes = np.zeros((num_objs, 5), dtype=np.float)
        part_boxes = np.zeros((num_objs*10, 5), dtype=np.float)
        human_classes = np.zeros((num_objs), dtype=np.int32)
    
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            x1 = max(float(get_data_from_tag(obj, 'xmin')),0)*im_shape[0]
            y1 = max(float(get_data_from_tag(obj, 'ymin')),0)*im_shape[1]
            x2 = min(float(get_data_from_tag(obj, 'xmax')),1)*im_shape[0] - 1
            y2 = min(float(get_data_from_tag(obj, 'ymax')),1)*im_shape[1] - 1
            cls = int(float(get_data_from_tag(obj, "name").rstrip()))
            human_boxes[ix, :] = [0,x1, y1, x2, y2]
            human_classes[ix] = cls
            for p_num in range(10):
                parts = data.getElementsByTagName('human'+str(ix+1)+'_part'+str(p_num+1)+'_roi')
                for ix, part in enumerate(parts):
                    # Make pixel indexes 0-based
                    x1 = max(float(get_data_from_tag(part, 'xmin')),0)*im_shape[0]
                    y1 = max(float(get_data_from_tag(part, 'ymin')),0)*im_shape[1]
                    x2 = min(float(get_data_from_tag(part, 'xmax')),1)*im_shape[0] - 1
                    y2 = min(float(get_data_from_tag(part, 'ymax')),1)*im_shape[1] - 1
                    cls = int(float(get_data_from_tag(part, "name").rstrip()))
                    part_boxes[ix*10+p_num, :] = [0,x1, y1, x2, y2]
                    if x1==0.0 and y1==0.0:
                        part_boxes[ix*10+p_num, :] = human_boxes[ix, :]
    
        objs = data.getElementsByTagName('object_roi')
        num_objs = len(objs)
    
        object_boxes = np.zeros((num_objs, 5), dtype=np.float)
        object_classes = np.zeros((num_objs), dtype=np.int32)
    
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            x1 = max(float(get_data_from_tag(obj, 'xmin')),0)*im_shape[0]
            y1 = max(float(get_data_from_tag(obj, 'ymin')),0)*im_shape[1]
            x2 = min(float(get_data_from_tag(obj, 'xmax')),1)*im_shape[0] - 1
            y2 = min(float(get_data_from_tag(obj, 'ymax')),1)*im_shape[1] - 1
            cls = int(float(get_data_from_tag(obj, "name")))
            object_boxes[ix, :] = [0,x1, y1, x2, y2]
            object_classes[ix] = cls
    
        num_human = human_boxes.shape[0]
        num_obj = object_boxes.shape[0]
        rel_boxes = np.zeros((num_human * num_obj, 5), dtype=np.float)
    
        for i in range(num_human):
            for j in range(num_obj):
                x1 = min(human_boxes[i][1],object_boxes[j][1])
                y1 = min(human_boxes[i][2],object_boxes[j][2])
                x2 = max(human_boxes[i][3],object_boxes[j][3])
                y2 = max(human_boxes[i][4],object_boxes[j][4])
                rel_boxes[j+num_obj * i,:] = [0,x1,y1,x2,y2]
    
        scene_boxes = np.zeros((1, 5), dtype=np.float)
        scene_boxes[0, :] = [0, 0.001, 0.001, im_shape[0]-1, im_shape[1]-1]
    
        label = np.zeros(600, dtype=np.int32)
        label_str = data.getElementsByTagName('label')[0].childNodes[0].data
        if ';' in label_str:
            label_str = label_str.split(';')
            for i in label_str[0].split(' '):
                if i != '':
                    label[int(i)] = 1
            for i in label_str[1].split(' '):
                if i != '':
                    label[int(i)] = -1
        else:
            for i in label_str.split(' '):
                if i != '':
                    label[int(i)] = 1 
        f.close()          
        return {'human_boxes': human_boxes,
                'human_classes': human_classes,
                'object_boxes': object_boxes,
                'object_classes': object_classes,
                'rel_boxes': rel_boxes,
                'scene_boxes': scene_boxes,
                'part_boxes': part_boxes,
                'label': label,
                'index': index}
    
    fin = open("./data/hico/test_filelist.txt",'r')
    root = './data/hico'
    
    lines = fin.readlines()
    if mode == 'pvp':
            res1 = open(res_dir + res_name + '.csv', 'w')
    elif mode == '10v':
            res2 = open(res_dir + res_name + '.csv', 'w')
    else:
            assert 0,"mode != '10v' or 'pvp'"
    for name in lines:
            try:
                    image = caffe.io.load_image(osp.join(root, 'JPEGImages', name.rstrip('\n') + '.jpg'))
            except ValueError:
                    print ('error')
                    continue
            net.blobs['data'].reshape(1,3,image.shape[0],image.shape[1])
            transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    
            transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
            transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
            transformer.set_mean('data',np.array([102,115,122]))
            transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    
            net.blobs['data'].data[...] = transformer.preprocess('data', image)
    
            human_roi = np.zeros((3,5)).astype(np.float)
            object_roi = np.zeros((4,5)).astype(np.float)
            rel_roi = np.zeros((12,5)).astype(np.float)
            scene_roi = np.zeros((1,5)).astype(np.float)
            part_roi = np.zeros((30,5)).astype(np.float)
            label = np.zeros((1,600)).astype(np.int32)
            anns = load_annotation(name.rstrip('\n'), root, (image.shape[1],image.shape[0]))
            human_roi = anns['human_boxes']
            object_roi = anns['object_boxes']
            rel_roi = anns['rel_boxes']
            scene_roi = anns['scene_boxes']
            part_roi = anns['part_boxes']
    
            for i in range(3):
                    # Add directly to the caffe data layer
                    net.blobs['human_roi'].data[i, ...] = human_roi[i]
                    net.blobs['obj1_roi'].data[i, ...] = object_roi[0]
                    net.blobs['obj2_roi'].data[i, ...] = object_roi[1]
                    net.blobs['obj3_roi'].data[i, ...] = object_roi[2]
                    net.blobs['obj4_roi'].data[i, ...] = object_roi[3]
                    net.blobs['rel1_roi'].data[i, ...] = rel_roi[0+4*i]
                    net.blobs['rel2_roi'].data[i, ...] = rel_roi[1+4*i]
                    net.blobs['rel3_roi'].data[i, ...] = rel_roi[2+4*i]
                    net.blobs['rel4_roi'].data[i, ...] = rel_roi[3+4*i]
                    net.blobs['scene_roi'].data[i, ...] = scene_roi[0]
                    net.blobs['human_roi'].data[i, ...] = human_roi[i]
                    for p in range(10):
                        net.blobs['part_roi'].data[p*3+i, ...] = part_roi[i*10 + p]
                        net.blobs['human_prior'].data[p*3+i, ...] = human_roi[i]
            net.forward()
    
            # obtain the output probabilities
            if mode == 'pvp':
                    output_prob_pvp_all = net.blobs['cls_score_pvp_all'].data #part pair
            elif mode =='10v':
                    output_prob_single =  net.blobs['cls_score_single'].data

            if mode == 'pvp':
              for roi in range(3):
                    for num in range(len(output_prob_pvp_all[0])):
                                    #print output_prob[num][0][0]
                                    res1.write(str(output_prob_pvp_all[roi][num]))
                                    res1.write(',')
                    res1.write('\n') 
            elif mode == '10v':
                for roi in range(3):
                    for num in range(len(output_prob_single[0])):
                                    #print output_prob[num][0][0]
                                    res2.write(str(output_prob_single[roi][num]))
                                    res2.write(',')
                    res2.write('\n')
    if mode == 'pvp':
        res1.close()
    elif mode == '10v':
        res2.close()
