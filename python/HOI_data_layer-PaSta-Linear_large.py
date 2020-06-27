# imports
import json
import time
import pickle
import scipy.misc
import skimage.io
import caffe

import numpy as np
import os.path as osp
import math

from xml.dom import minidom
from random import shuffle
from threading import Thread
from PIL import Image

from tools import SimpleTransformer


class HOIDataLayer(caffe.Layer):

    """
    This is a simple syncronous datalayer for training a multilabel model on
    PASCAL.
    """

    def setup(self, bottom, top):

        self.top_names = ['data', 'human_roi',  'obj1_roi', 'obj2_roi', 'obj3_roi', 'obj4_roi', 'rel1_roi', 'rel2_roi', 'rel3_roi', 'rel4_roi', 'scene_roi', 'label', 'part_roi', 'label_wts', 'human_prior', 'hp_list', 'hp_list_wts', 'pvp_ankle2', 'pvp_ankle2_wts', 'pvp_knee2', 'pvp_knee2_wts', 'pvp_hip', 'pvp_hip_wts', 'pvp_hand2', 'pvp_hand2_wts', 'pvp_shoulder2', 'pvp_shoulder2_wts', 'pvp_head', 'pvp_head_wts', 'pvp_wts', 'verb_list', 'verb_list_wts', 'object_list', 'object_list_wts']

        # === Read input parameters ===

        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)

        # Check the paramameters for validity.
        check_params(params)

        # store input as class variables
        self.batch_size = params['batch_size']

        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader(params, None)

        # === reshape tops ===
        # 100 is dummy value
        top[0].reshape(
            self.batch_size, 3, 100, 100)
        # Note the 20 channels (because PASCAL has 20 classes.)
        top[1].reshape(3*self.batch_size, 5)
        top[2].reshape(3*self.batch_size, 5)
        top[3].reshape(3*self.batch_size, 5)
        top[4].reshape(3*self.batch_size, 5)
        top[5].reshape(3*self.batch_size, 5)
        top[6].reshape(3*self.batch_size, 5)
        top[7].reshape(3*self.batch_size, 5)
        top[8].reshape(3*self.batch_size, 5)
        top[9].reshape(3*self.batch_size, 5)
        top[10].reshape(3*self.batch_size, 5)

        top[11].reshape(self.batch_size,600)
        top[12].reshape(3*10*self.batch_size, 5)
        top[13].reshape(self.batch_size,600)
        top[14].reshape(3*10*self.batch_size, 5)

        top[15].reshape(self.batch_size, 10)
        top[16].reshape(self.batch_size, 10)
        
        #ankle knee hip hand shoulder head
        top[17].reshape(self.batch_size, 6)
        top[18].reshape(self.batch_size, 6)
        top[19].reshape(self.batch_size, 6)
        top[20].reshape(self.batch_size, 6)
        top[21].reshape(self.batch_size, 3)
        top[22].reshape(self.batch_size, 3)
        top[23].reshape(self.batch_size, 23)
        top[24].reshape(self.batch_size, 23)
        top[25].reshape(self.batch_size, 5)
        top[26].reshape(self.batch_size, 5)
        top[27].reshape(self.batch_size, 12)
        top[28].reshape(self.batch_size, 12)
        #PVP_WTS
        top[29].reshape(self.batch_size, 6)

        #verb_list, object_list
        top[30].reshape(self.batch_size, 117)
        top[31].reshape(self.batch_size, 117)
        top[32].reshape(self.batch_size, 80)
        top[33].reshape(self.batch_size, 80)
        #print_info("PascalMultilabelDataLayerSync", params)

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, human_roi, object_roi, rel_roi, scene_roi, part_roi, label,label_wts,hp_list, hp_list_wts, pvp_ankle2, pvp_ankle2_wts, pvp_knee2, pvp_knee2_wts, pvp_hip, pvp_hip_wts, pvp_hand2, pvp_hand2_wts, pvp_shoulder2, pvp_shoulder2_wts, pvp_head, pvp_head_wts, pvp_wts, verb_list, verb_list_wts, object_list, object_list_wts  = self.batch_loader.load_next_image()
            top[0].reshape(self.batch_size,3,im.shape[1],im.shape[2])
            top[0].data[itt, ...] = im
            human_size = human_roi.shape[0]
            top[1].reshape(human_size*self.batch_size, 5)
            top[2].reshape(human_size*self.batch_size, 5)
            top[3].reshape(human_size*self.batch_size, 5)
            top[4].reshape(human_size*self.batch_size, 5)
            top[5].reshape(human_size*self.batch_size, 5)
            top[6].reshape(human_size*self.batch_size, 5)
            top[7].reshape(human_size*self.batch_size, 5)
            top[8].reshape(human_size*self.batch_size, 5)
            top[9].reshape(human_size*self.batch_size, 5)
            top[10].reshape(3*self.batch_size, 5)
            top[12].reshape(human_size*self.batch_size * 10, 5)
            top[14].reshape(human_size*self.batch_size * 10, 5)
            
            for i in range(3):
                # Add directly to the caffe data layer
                top[1].data[3*itt+i, ...] = human_roi[i]
                top[2].data[3*itt+i, ...] = object_roi[0]
                top[3].data[3*itt+i, ...] = object_roi[1]
                top[4].data[3*itt+i, ...] = object_roi[2]
                top[5].data[3*itt+i, ...] = object_roi[3]
                top[6].data[3*itt+i, ...] = rel_roi[0+4*i]
                top[7].data[3*itt+i, ...] = rel_roi[1+4*i]
                top[8].data[3*itt+i, ...] = rel_roi[2+4*i]
                top[9].data[3*itt+i, ...] = rel_roi[3+4*i]
                top[10].data[3*itt+i, ...] = scene_roi[0]
                for p in range(10):
                    top[12].data[3*itt+p*3+i, ...] = part_roi[i*10 + p]
                    top[12].data[3*itt+p*3+i][0] = itt
                    top[14].data[3*itt+p*3+i, ...] = human_roi[i]
                    top[14].data[3*itt+p*3+i][0] = itt
                # set batch index
                top[1].data[3*itt+i][0] = itt
                top[2].data[3*itt+i][0] = itt
                top[3].data[3*itt+i][0] = itt
                top[4].data[3*itt+i][0] = itt
                top[5].data[3*itt+i][0] = itt
                top[6].data[3*itt+i][0] = itt
                top[7].data[3*itt+i][0] = itt
                top[8].data[3*itt+i][0] = itt
                top[9].data[3*itt+i][0] = itt
                top[10].data[3*itt+i][0] = itt
                top[12].data[3*itt+i][0] = itt
            top[11].data[itt,...] = label
            top[13].data[itt,...] = label_wts
            top[15].data[itt,...] = hp_list
            top[16].data[itt,...] = hp_list_wts
            top[17].data[itt,...] = pvp_ankle2
            top[18].data[itt,...] = pvp_ankle2_wts
            top[19].data[itt,...] = pvp_knee2
            top[20].data[itt,...] = pvp_knee2_wts
            top[21].data[itt,...] = pvp_hip
            top[22].data[itt,...] = pvp_hip_wts
            top[23].data[itt,...] = pvp_hand2
            top[24].data[itt,...] = pvp_hand2_wts
            top[25].data[itt,...] = pvp_shoulder2
            top[26].data[itt,...] = pvp_shoulder2_wts
            top[27].data[itt,...] = pvp_head
            top[28].data[itt,...] = pvp_head_wts
            top[29].data[itt,...] = pvp_wts
            top[30].data[itt,...] = verb_list
            top[31].data[itt,...] = verb_list_wts
            top[32].data[itt,...] = object_list
            top[33].data[itt,...] = object_list_wts
#print top[16].data
            #print human_roi.shape[0],10000
#           print top[0].data
    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


class BatchLoader(object):

    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """

    def __init__(self, params, result):
        self.result = result
        self.batch_size = params['batch_size']
        self.pascal_root = params['root']
        self.im_shape = (100,100) #dummy value
        self.max_size = params['max_size']
        self.list_file = params['list_file']
        # get list of image indexes.
        self.indexlist = [line.rstrip('\n') for line in open(self.list_file)]
        self._cur = 0  # current image
        # this class does some simple data-manipulations
        self.transformer = SimpleTransformer()

        # print "BatchLoader initialized with {} images".format(
        #     len(self.indexlist))

    def load_next_image(self):
        """
        Load the next image in a batch.
        """
        # Did we finish an epoch?
        if self._cur == len(self.indexlist):
            self._cur = 0
        if self._cur == 0:
            shuffle(self.indexlist)

        # Load an image
        index = self.indexlist[self._cur]  # Get the image index
        image_file_name = index + '.jpg'
        #im = np.asarray(Image.open(
            #osp.join(self.pascal_root, 'JPEGImages', image_file_name)))
        im = scipy.misc.imread(osp.join(self.pascal_root, 'JPEGImages', image_file_name),mode='RGB')
        #print im.shape  #height,width,channel
        length = max(im.shape[0],im.shape[1])
        ratio = self.max_size/length
        self.im_shape = (im.shape[1] * ratio, im.shape[0] * ratio)
        #print self.im_shape
        #im = scipy.misc.imresize(im, self.im_shape)  # resize

        # do a simple horizontal flip as data augmentation
        #flip = np.random.choice(2)*2-1
        #im = im[:, ::flip, :]

        # Load and prepare ground truth
        anns = load_annotation(index, self.pascal_root, self.im_shape)
        human_roi = anns['human_boxes']
        object_roi = anns['object_boxes']
        rel_roi = anns['rel_boxes']
        scene_roi = anns['scene_boxes']
        part_roi = anns['part_boxes']
        label = anns['label']
        label_wts = anns['label_wts']
        hp_list = anns['hp_list']
        hp_list_wts = anns['hp_list_wts']
        pvp_ankle2 = anns['pvp_ankle2']
        pvp_ankle2_wts = anns['pvp_ankle2_wts']
        pvp_knee2 = anns['pvp_knee2']
        pvp_knee2_wts = anns['pvp_knee2_wts']
        pvp_hip = anns['pvp_hip']
        pvp_hip_wts = anns['pvp_hip_wts']
        pvp_hand2 = anns['pvp_hand2']
        pvp_hand2_wts = anns['pvp_hand2_wts']
        pvp_shoulder2 = anns['pvp_shoulder2']
        pvp_shoulder2_wts = anns['pvp_shoulder2_wts']
        pvp_head = anns['pvp_head']
        pvp_head_wts = anns['pvp_head_wts']
        pvp_wts = anns['pvp_wts']
        verb_list = anns['verb_list']
        verb_list_wts = anns['verb_list_wts']
        object_list = anns['object_list']
        object_list_wts = anns['object_list_wts']
        self._cur += 1
        # self.transformer.preprocess(im): change RGB to BGR
        if  np.random.choice(2) == 1:
            im,human_roi, object_roi, rel_roi,scene_roi,part_roi = Flip(im,human_roi, object_roi, rel_roi,scene_roi,part_roi, self.im_shape)
        return self.transformer.preprocess(im), human_roi, object_roi,rel_roi, scene_roi, part_roi, label, label_wts, hp_list, hp_list_wts, pvp_ankle2, pvp_ankle2_wts, pvp_knee2, pvp_knee2_wts, pvp_hip, pvp_hip_wts, pvp_hand2, pvp_hand2_wts, pvp_shoulder2, pvp_shoulder2_wts, pvp_head, pvp_head_wts, pvp_wts, verb_list, verb_list_wts, object_list, object_list_wts

def Flip(im,human_roi, object_roi, rel_roi,scene_roi,part_roi, imshape):
    im = im[:, ::-1, :]
    for i in range(human_roi.shape[0]):
        tmp = human_roi[i][1]
        human_roi[i][1] = imshape[0]-human_roi[i][3] - 1
        human_roi[i][3] = imshape[0]-tmp - 1
    for i in range(object_roi.shape[0]):
        tmp = object_roi[i][1]
        object_roi[i][1] = imshape[0]-object_roi[i][3] - 1
        object_roi[i][3] = imshape[0]-tmp - 1
    for i in range(rel_roi.shape[0]):
        tmp = rel_roi[i][1]
        rel_roi[i][1] = imshape[0]-rel_roi[i][3] - 1
        rel_roi[i][3] = imshape[0]-tmp - 1
    for i in range(part_roi.shape[0]):
        tmp = part_roi[i][1]
        part_roi[i][1] = imshape[0]-part_roi[i][3] - 1
        part_roi[i][3] = imshape[0]-tmp - 1
    tmp = scene_roi[0][1]
    scene_roi[0][1] = imshape[0]-scene_roi[0][3] - 1
    scene_roi[0][3] = imshape[0]-tmp - 1
    return im,human_roi, object_roi, rel_roi,scene_roi,part_roi

def addGaussianNoise(x1,y1,x2,y2,imshape):
    width = x2-x1
    ht = y2-y1
    x1 = max(0, x1+np.random.normal(-0.0142,0.05)*width)
    y1 = max(0, y1+np.random.normal(0.0043,0.025)*ht)
    x2 = max(min(imshape[0], x2+np.random.normal(0.0154,0.05)*width), x1+5)
    y2 = max(min(imshape[1], y2+np.random.normal(-0.0013,0.025)*ht), y1+5)

    return x1,y1,x2,y2


def load_annotation(index, pascal_root, im_shape):
    """
    This code is borrowed from Ross Girshick's FAST-RCNN code
    (https://github.com/rbgirshick/fast-rcnn).
    It parses the PASCAL .xml metadata files.
    See publication for further details: (http://arxiv.org/abs/1504.08083).

    Thanks Ross!

    """


    filename = osp.join(pascal_root, 'Annotations', index + '.xml')
    #print 'Loading: {}'.format(filename)

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
        #x1,y1,x2,y2 = addGaussianNoise(x1,y1,x2,y2, im_shape)
        cls = int(float(get_data_from_tag(obj, "name").rstrip()))
        human_boxes[ix, :] = [0,x1, y1, x2, y2]
        human_classes[ix] = cls
        for p_num in range(10):
            parts = data.getElementsByTagName('human'+str(ix+1)+'_part'+str(p_num)+'_roi')
            for _, part in enumerate(parts):
                # Make pixel indexes 0-based
                x1 = max(float(get_data_from_tag(part, 'xmin')),0)*im_shape[0]
                y1 = max(float(get_data_from_tag(part, 'ymin')),0)*im_shape[1]
                x2 = min(float(get_data_from_tag(part, 'xmax')),1)*im_shape[0] - 1
                y2 = min(float(get_data_from_tag(part, 'ymax')),1)*im_shape[1] - 1
                #x1,y1,x2,y2 = addGaussianNoise(x1,y1,x2,y2, im_shape)
                cls = int(float(get_data_from_tag(part, "name").rstrip()))
                part_boxes[ix*10+p_num, :] = [0,x1, y1, x2, y2]
                if x1==0 and y1==0:
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
        #x1,y1,x2,y2 = addGaussianNoise(x1,y1,x2,y2,im_shape)
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

    verb_list = np.zeros(117, dtype=np.int32)
    for i in range(len(label)):
        if label[i] == 1:
            verb_no = get_verb_no(i+1)
            verb_list[verb_no-1] = 1
        if label[i] == -1:
            verb_no = get_verb_no(i+1)
            if verb_list[verb_no-1] != 1:
                verb_list[verb_no-1] = -1

    verb_list_wts = np.zeros(117, dtype=float)
    for i in range(len(verb_list_wts)):
        verb_list_wts[i] = 1.0

    object_list = np.zeros(80, dtype=np.int32)
    for i in range(len(label)):
        if label[i] == 1:
            object_no = get_object_no(i+1)
            object_list[object_no-1] = 1
        if label[i] == -1:
            object_no = get_object_no(i+1)
            if object_list[object_no-1] != 1:
                object_list[object_no-1] = -1
    object_list_wts = np.zeros(80, dtype=float)
    for i in range(len(object_list_wts)):
        object_list_wts[i] = 1.0
            
    #print "label:", label
    f.close()          
    with open(pascal_root+'pos_weights_large.txt', 'r') as f1:
        pos_wts = [float(line.rstrip('\n')) for line in f1]
    with open(pascal_root+'neg_weights.txt', 'r') as f2:
        neg_wts = [float(line.rstrip('\n')) for line in f2]
    label_wts = np.zeros(600, dtype=np.float)
    for i in range(600):
        if label[i] == 1:
          label_wts[i] = math.log10(pos_wts[i])*10
        elif label[i] == 0:
          label_wts[i] = 1
        else:
          label_wts[i] = 0
    
    hp_list = np.zeros(10, dtype=np.int32)
    for i in range(10):
        hp_list[int(i)] = int(data.getElementsByTagName('hp_list')[0].childNodes[i].childNodes[0].data)
 
    hp_list_wts = np.zeros(10, dtype=np.float)
    with open(pascal_root+'hp_list_wts.txt', 'r') as f3:
        hp_wts = [float(line.rstrip('\n')) for line in f3]
    for i in range(10):
        if hp_list[i] == 1:
            hp_list_wts[int(i)] = hp_wts[i]
        else:
            hp_list_wts[i] = 1

    pvp_ankle2 = np.zeros(6, dtype=np.int32)
    for i in range(6):
        pvp_ankle2[int(i)] = int(data.getElementsByTagName('pvp_ankle2')[0].childNodes[i].childNodes[0].data)
    pvp_ankle2_wts = np.zeros(6, dtype=np.float)
    for i in range(6):
        pvp_ankle2_wts[int(i)] = 1.0

    pvp_knee2 = np.zeros(6, dtype=np.int32)
    for i in range(6):
        pvp_knee2[int(i)] = int(data.getElementsByTagName('pvp_knee2')[0].childNodes[i].childNodes[0].data)
    pvp_knee2_wts = np.zeros(6, dtype=np.float)
    for i in range(6):
        pvp_knee2_wts[int(i)] = 1.0

    pvp_hip = np.zeros(3, dtype=np.int32)
    for i in range(3):
        pvp_hip[int(i)] = int(data.getElementsByTagName('pvp_hip')[0].childNodes[i].childNodes[0].data)
    pvp_hip_wts = np.zeros(3, dtype=np.float)
    for i in range(3):
        pvp_hip_wts[int(i)] = 1.0
    
    pvp_hand2 = np.zeros(23, dtype=np.int32)
    for i in range(23):
        pvp_hand2[int(i)] = int(data.getElementsByTagName('pvp_hand2')[0].childNodes[i].childNodes[0].data)
    pvp_hand2_wts = np.zeros(23, dtype=np.float)
    for i in range(23):
        pvp_hand2_wts[int(i)] = 1.0
    
    pvp_shoulder2 = np.zeros(5, dtype=np.int32)
    for i in range(5):
        pvp_shoulder2[int(i)] = int(data.getElementsByTagName('pvp_shoulder2')[0].childNodes[i].childNodes[0].data)
    pvp_shoulder2_wts = np.zeros(5, dtype=np.float)
    for i in range(5):
        pvp_shoulder2_wts[int(i)] = 1.0
    
    pvp_head = np.zeros(12, dtype=np.int32)
    for i in range(12):
        pvp_head[int(i)] = int(data.getElementsByTagName('pvp_head')[0].childNodes[i].childNodes[0].data)
    pvp_head_wts = np.zeros(12, dtype=np.float)
    for i in range(12):
        pvp_head_wts[int(i)] = 1.0

    pvp_wts = np.zeros(6, dtype = np.float)
    for i in range(6):
        pvp_wts[int(i)] = 1.0

    return {'human_boxes': human_boxes,
            'human_classes': human_classes,
            'object_boxes': object_boxes,
            'object_classes': object_classes,
            'rel_boxes': rel_boxes,
            'scene_boxes': scene_boxes,
            'part_boxes': part_boxes,
            'label': label,
            'label_wts': label_wts,
            'index': index,
            'hp_list': hp_list,
            'hp_list_wts': hp_list_wts,
            'pvp_ankle2': pvp_ankle2,
            'pvp_ankle2_wts': pvp_ankle2_wts,
            'pvp_knee2': pvp_knee2,
            'pvp_knee2_wts': pvp_knee2_wts,
            'pvp_hip': pvp_hip,
            'pvp_hip_wts': pvp_hip_wts,
            'pvp_hand2': pvp_hand2,
            'pvp_hand2_wts': pvp_hand2_wts,
            'pvp_shoulder2': pvp_shoulder2,
            'pvp_shoulder2_wts': pvp_shoulder2_wts,
            'pvp_head': pvp_head,
            'pvp_head_wts': pvp_head_wts,
            'pvp_wts': pvp_wts,
            'verb_list': verb_list,
            'verb_list_wts': verb_list_wts,
            'object_list': object_list,
            'object_list_wts': object_list_wts
    }


def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    required = ['batch_size', 'root', 'max_size', 'list_file']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
    """
    Ouput some info regarding the class
    """
    print "{} initialized with bs: {}, max_size: {}.".format(
        name,
        params['batch_size'],
        params['max_size'])



# Param: hoi-no
# Output: verb_id
def get_verb_no(hoi_no):
    verb_name = ''
    with open('./info/hoi.txt', 'r') as f:
        for line in f.readlines():
            if int(line.split()[0]) == hoi_no:
                verb_name = line.split()[2]
                break
    with open('./info/verb.txt', 'r') as f1:
        for line in f1.readlines():
            if line.split()[1] == verb_name:
                return int(line.split()[0])
    return -1


# Param: hoi-no
# Output: object_id
def get_object_no(hoi_no):
    object_name = ''
    with open('./info/hoi.txt', 'r') as f:
        for line in f.readlines():
            if int(line.split()[0]) == hoi_no:
                object_name = line.split()[1]
                break
    with open('./info/object.txt', 'r') as f1:
        for line in f1.readlines():
            if line.split()[1] == object_name:
                return int(line.split()[0])
    return -1



