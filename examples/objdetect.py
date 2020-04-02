import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '/home/xx/git/ssd/'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(1)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL VOC labels
voc_labelmap_file = 'data/coco/labelmap_coco.prototxt'

file = open(voc_labelmap_file, 'r')
voc_labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), voc_labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

#model_def = 'models/VGG_SSD/deploy.prototxt'
#model_def = 'models/VGG_SSD/models/VGGNet/coco/SSD_500x500/deploy.prototxt'
model_def = 'models/VGGNet/coco/SSD_512x512/deploy.prototxt'

#model_weights = 'models/VGG_SSD/VGG_MPII_COCO14_SSD_500x500_iter_60000.caffemodel'
#model_weights = 'models/VGG_SSD/models/VGGNet/coco/SSD_500x500/VGG_coco_SSD_500x500_iter_200000.caffemodel'
model_weights = 'models/VGGNet/coco/SSD_512x512/VGG_coco_SSD_512x512_iter_360000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

#  set net to batch size of 1
import os
import h5py
image_resize = 512
net.blobs['data'].reshape(1,3,image_resize,image_resize)
idx = 0
#root_dir = "/data/xx/MPII_COCO14/test2015/"
root_dir = "/home/xx/secondHDD/xx/data/hico_20150920/images/test2015/"
anno_dir = "data/hico/anno/test2015/"
#if (os.path.exists("/home/xx/data/multiple_ssd/ssd_test") == False):
 #   os.makedirs("/home/xx/data/multiple_ssd/ssd_test")

configThred = 0.2
NMSThred = 0.45

lines = [line.rstrip('\n') .rstrip('\r') for line in open("data/hico/test_list_nopath.txt")]

if not os.path.exists(anno_dir):
    os.makedirs(anno_dir)

arr=[]
FileLength = len(lines)
for i in range(0,FileLength):

    if ((i%1000) == 0):
        print i
    picture = lines[i].split("\t")

    p_name = picture[0]

    filename = os.path.join(root_dir, p_name)
    results = open(anno_dir+p_name[0:-4]+".xml", 'w')

    if (os.path.isfile(filename) == False):
        print "NO"
        continue
    image = caffe.io.load_image(filename)

    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    
    # Forward pass.
    detections = (net.forward()['detection_out'])

    # Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]


    #bound_boxes.append(boxes)
    top_indices = [m for m, conf in enumerate(det_conf) if conf > configThred]
    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
#    top_labels = get_labelname(voc_labelmap, top_label_indices)
    top_labels = det_label[top_indices]
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    human_indices = [m for m, label in enumerate(top_labels) if label == 1]
    human_conf = top_conf[human_indices]
    human_label_indices = det_label[human_indices].tolist()
#    top_labels = get_labelname(voc_labelmap, top_label_indices)
    human_labels = top_labels[human_indices]
    human_xmin = top_xmin[human_indices]
    human_ymin = top_ymin[human_indices]
    human_xmax = top_xmax[human_indices]
    human_ymax = top_ymax[human_indices]

    if len(human_indices) >= 3:
        sort_indices = sorted(range(len(human_conf)), key=lambda k: human_conf[k], reverse=True)
        human_conf = human_conf[sort_indices]
    #    human_label_indices = human_label[sort_indices].tolist()
    #    top_labels = get_labelname(voc_labelmap, top_label_indices)
        human_labels = human_labels[sort_indices]
        human_xmin = human_xmin[sort_indices]
        human_ymin = human_ymin[sort_indices]
        human_xmax = human_xmax[sort_indices]
        human_ymax = human_ymax[sort_indices] 
        xml = '<annotation>\n<folder>\nHICOpascalformat\n</folder>\n<filename>\n'
        xml += p_name + '\n</filename>\n<source>\n<database>\nHICOpascalformat\n</database>\n</source>\n<size>\n'
        xml += '<width>\n' + str(image.shape[1]) + '\n</width>\n' + '<height>\n' + str(image.shape[0]) + '\n</height>\n'
        xml += '<depth>\n3\n</depth>\n</size>\n<segmented>\n0\n</segmented>\n'

        for i in xrange(3):
            xml += '<human_roi>\n<name>\n' + str(human_labels[i]) + '\n</name>\n'
            xml += '<bndbox>\n<xmin>\n' + str(human_xmin[i]) + '\n</xmin>\n'
            xml += '<ymin>\n' + str(human_ymin[i]) + '\n</ymin>\n'
            xml += '<xmax>\n' + str(human_xmax[i]) + '\n</xmax>\n'
            xml += '<ymax>\n' + str(human_ymax[i]) + '\n</ymax>\n</bndbox>\n'
            xml += '<truncated>\n0\n</truncated>\n<difficult>\n0\n</difficult>\n</human_roi>\n'
    elif len(human_indices) == 2:
        sort_indices = [0,0,1]
        human_conf = human_conf[sort_indices]
    #    human_label_indices = human_label[sort_indices].tolist()
    #    top_labels = get_labelname(voc_labelmap, top_label_indices)
        human_labels = human_labels[sort_indices]
        human_xmin = human_xmin[sort_indices]
        human_ymin = human_ymin[sort_indices]
        human_xmax = human_xmax[sort_indices]
        human_ymax = human_ymax[sort_indices] 
        xml = '<annotation>\n<folder>\nHICOpascalformat\n</folder>\n<filename>\n'
        xml += p_name + '\n</filename>\n<source>\n<database>\nHICOpascalformat\n</database>\n</source>\n<size>\n'
        xml += '<width>\n' + str(image.shape[1]) + '\n</width>\n' + '<height>\n' + str(image.shape[0]) + '\n</height>\n'
        xml += '<depth>\n3\n</depth>\n</size>\n<segmented>\n0\n</segmented>\n'

        for i in xrange(3):
            xml += '<human_roi>\n<name>\n' + str(human_labels[i]) + '\n</name>\n'
            xml += '<bndbox>\n<xmin>\n' + str(human_xmin[i]) + '\n</xmin>\n'
            xml += '<ymin>\n' + str(human_ymin[i]) + '\n</ymin>\n'
            xml += '<xmax>\n' + str(human_xmax[i]) + '\n</xmax>\n'
            xml += '<ymax>\n' + str(human_ymax[i]) + '\n</ymax>\n</bndbox>\n'
            xml += '<truncated>\n0\n</truncated>\n<difficult>\n0\n</difficult>\n</human_roi>\n'     
    elif len(human_indices) == 1:
        sort_indices = [0,0,0]
        human_conf = human_conf[sort_indices]
        #human_label_indices = human_label[sort_indices].tolist()
    #    top_labels = get_labelname(voc_labelmap, top_label_indices)
        human_labels = human_labels[sort_indices]
        human_xmin = human_xmin[sort_indices]
        human_ymin = human_ymin[sort_indices]
        human_xmax = human_xmax[sort_indices]
        human_ymax = human_ymax[sort_indices] 
        xml = '<annotation>\n<folder>\nHICOpascalformat\n</folder>\n<filename>\n'
        xml += p_name + '\n</filename>\n<source>\n<database>\nHICOpascalformat\n</database>\n</source>\n<size>\n'
        xml += '<width>\n' + str(image.shape[1]) + '\n</width>\n' + '<height>\n' + str(image.shape[0]) + '\n</height>\n'
        xml += '<depth>\n3\n</depth>\n</size>\n<segmented>\n0\n</segmented>\n'

        for i in xrange(3):
            xml += '<human_roi>\n<name>\n' + str(human_labels[i]) + '\n</name>\n'
            xml += '<bndbox>\n<xmin>\n' + str(human_xmin[i]) + '\n</xmin>\n'
            xml += '<ymin>\n' + str(human_ymin[i]) + '\n</ymin>\n'
            xml += '<xmax>\n' + str(human_xmax[i]) + '\n</xmax>\n'
            xml += '<ymax>\n' + str(human_ymax[i]) + '\n</ymax>\n</bndbox>\n'
            xml += '<truncated>\n0\n</truncated>\n<difficult>\n0\n</difficult>\n</human_roi>\n'      
    else:#no people detected
        xml = '<annotation>\n<folder>\nHICOpascalformat\n</folder>\n<filename>\n'
        xml += p_name + '\n</filename>\n<source>\n<database>\nHICOpascalformat\n</database>\n</source>\n<size>\n'
        xml += '<width>\n' + str(image.shape[1]) + '\n</width>\n' + '<height>\n' + str(image.shape[0]) + '\n</height>\n'
        xml += '<depth>\n3\n</depth>\n</size>\n<segmented>\n0\n</segmented>\n'

        for i in xrange(3):
            xml += '<human_roi>\n<name>\n' + '-1' + '\n</name>\n'
            xml += '<bndbox>\n<xmin>\n' + str(0) + '\n</xmin>\n'
            xml += '<ymin>\n' + str(0) + '\n</ymin>\n'
            xml += '<xmax>\n' + str(1) + '\n</xmax>\n'
            xml += '<ymax>\n' + str(1) + '\n</ymax>\n</bndbox>\n'
            xml += '<truncated>\n0\n</truncated>\n<difficult>\n0\n</difficult>\n</human_roi>\n'       


    top_indices = [m for m, conf in enumerate(det_conf) if conf > 0.1]
    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
#    top_labels = get_labelname(voc_labelmap, top_label_indices)
    top_labels = det_label[top_indices]
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    object_indices = [m for m, label in enumerate(top_labels) if label != 1]

    object_conf = top_conf[object_indices]
   # object_label_indices = det_label[object_indices].tolist()
#    top_labels = get_labelname(voc_labelmap, top_label_indices)
    object_labels = top_labels[object_indices]
    object_xmin = top_xmin[object_indices]
    object_ymin = top_ymin[object_indices]
    object_xmax = top_xmax[object_indices]
    object_ymax = top_ymax[object_indices]

    if len(object_indices) >= 4:
        sort_indices = sorted(range(len(object_conf)), key=lambda k: object_conf[k], reverse=True)
        object_conf = object_conf[sort_indices]
    #    object_label_indices = object_label[sort_indices].tolist()
    #    top_labels = get_labelname(voc_labelmap, top_label_indices)
        object_labels = object_labels[sort_indices]
        object_xmin = object_xmin[sort_indices]
        object_ymin = object_ymin[sort_indices]
        object_xmax = object_xmax[sort_indices]
        object_ymax = object_ymax[sort_indices] 

        for i in xrange(4):
            xml += '<object_roi>\n<name>\n' + str(object_labels[i]) + '\n</name>\n'
            xml += '<bndbox>\n<xmin>\n' + str(object_xmin[i]) + '\n</xmin>\n'
            xml += '<ymin>\n' + str(object_ymin[i]) + '\n</ymin>\n'
            xml += '<xmax>\n' + str(object_xmax[i]) + '\n</xmax>\n'
            xml += '<ymax>\n' + str(object_ymax[i]) + '\n</ymax>\n</bndbox>\n'
            xml += '<truncated>\n0\n</truncated>\n<difficult>\n0\n</difficult>\n</object_roi>\n'
    elif len(object_indices) == 3:
        sort_indices = [0,0,1,2]
        object_conf = object_conf[sort_indices]
    #    object_label_indices = object_label[sort_indices].tolist()
    #    top_labels = get_labelname(voc_labelmap, top_label_indices)
        object_labels = object_labels[sort_indices]
        object_xmin = object_xmin[sort_indices]
        object_ymin = object_ymin[sort_indices]
        object_xmax = object_xmax[sort_indices]
        object_ymax = object_ymax[sort_indices] 

        for i in xrange(4):
            xml += '<object_roi>\n<name>\n' + str(object_labels[i]) + '\n</name>\n'
            xml += '<bndbox>\n<xmin>\n' + str(object_xmin[i]) + '\n</xmin>\n'
            xml += '<ymin>\n' + str(object_ymin[i]) + '\n</ymin>\n'
            xml += '<xmax>\n' + str(object_xmax[i]) + '\n</xmax>\n'
            xml += '<ymax>\n' + str(object_ymax[i]) + '\n</ymax>\n</bndbox>\n'
            xml += '<truncated>\n0\n</truncated>\n<difficult>\n0\n</difficult>\n</object_roi>\n'    
    elif len(object_indices) == 2:
        sort_indices = [0,0,1,1]
        object_conf = object_conf[sort_indices]
    #    object_label_indices = object_label[sort_indices].tolist()
    #    top_labels = get_labelname(voc_labelmap, top_label_indices)
        object_labels = object_labels[sort_indices]
        object_xmin = object_xmin[sort_indices]
        object_ymin = object_ymin[sort_indices]
        object_xmax = object_xmax[sort_indices]
        object_ymax = object_ymax[sort_indices] 

        for i in xrange(4):
            xml += '<object_roi>\n<name>\n' + str(object_labels[i]) + '\n</name>\n'
            xml += '<bndbox>\n<xmin>\n' + str(object_xmin[i]) + '\n</xmin>\n'
            xml += '<ymin>\n' + str(object_ymin[i]) + '\n</ymin>\n'
            xml += '<xmax>\n' + str(object_xmax[i]) + '\n</xmax>\n'
            xml += '<ymax>\n' + str(object_ymax[i]) + '\n</ymax>\n</bndbox>\n'
            xml += '<truncated>\n0\n</truncated>\n<difficult>\n0\n</difficult>\n</object_roi>\n'   
    elif len(object_indices) == 1:
        sort_indices = [0,0,0,0]
        object_conf = object_conf[sort_indices]
    #    object_label_indices = object_label[sort_indices].tolist()
    #    top_labels = get_labelname(voc_labelmap, top_label_indices)
        object_labels = object_labels[sort_indices]
        object_xmin = object_xmin[sort_indices]
        object_ymin = object_ymin[sort_indices]
        object_xmax = object_xmax[sort_indices]
        object_ymax = object_ymax[sort_indices] 

        for i in xrange(4):
            xml += '<object_roi>\n<name>\n' + str(object_labels[i]) + '\n</name>\n'
            xml += '<bndbox>\n<xmin>\n' + str(object_xmin[i]) + '\n</xmin>\n'
            xml += '<ymin>\n' + str(object_ymin[i]) + '\n</ymin>\n'
            xml += '<xmax>\n' + str(object_xmax[i]) + '\n</xmax>\n'
            xml += '<ymax>\n' + str(object_ymax[i]) + '\n</ymax>\n</bndbox>\n'
            xml += '<truncated>\n0\n</truncated>\n<difficult>\n0\n</difficult>\n</object_roi>\n'   
    else:#no object detected

        for i in xrange(4):
            xml += '<object_roi>\n<name>\n' + '-1' + '\n</name>\n'
            xml += '<bndbox>\n<xmin>\n' + str(0) + '\n</xmin>\n'
            xml += '<ymin>\n' + str(0) + '\n</ymin>\n'
            xml += '<xmax>\n' + str(1) + '\n</xmax>\n'
            xml += '<ymax>\n' + str(1) + '\n</ymax>\n</bndbox>\n'
            xml += '<truncated>\n0\n</truncated>\n<difficult>\n0\n</difficult>\n</object_roi>\n'    

    xml += '</annotation>'
    results.write(xml)
    results.close()

print "Done"
print num_boxes

