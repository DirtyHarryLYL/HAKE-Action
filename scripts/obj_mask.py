import os
from xml.dom import minidom
import numpy as np


detection2hico = [0,50,10,19,45,1,17,74,75,12,73,32,64,49,9,11,21,29,38,57,26,31,7,80,35,3,77,37,69,65,34,60,61,63,41,5,6,59,66,68,14,79,27,33,42,62,15,4,2,55,47,16,20,39,51,30,18,23,25,52,8,28,71,76,43,46,54,40,22,44,48,70,58,53,13,24,78,56,67,36,72]

get_hoi_list_by_ob_id = [(-1, -1), (160, 169), (10, 23), (65, 75), (146, 159), (0, 9), (54, 64), (186, 193), (567, 575), (31, 45), (562, 566), (325, 329), (502, 505), (414, 417), (243, 246), (24, 30), (76, 85), (111, 128), (129, 145), (174, 185), (96, 106), (313, 324), (235, 238), (595, 599), (342, 347), (208, 213), (576, 583), (352, 355), (538, 545), (506, 515), (336, 341), (463, 473), (474, 482), (488, 501), (368, 375), (224, 231), (232, 234), (453, 462), (516, 527), (533, 537), (46, 53), (588, 594), (295, 304), (330, 335), (376, 382), (483, 487), (252, 256), (214, 223), (198, 207), (438, 444), (397, 406), (257, 263), (273, 282), (356, 362), (418, 428), (305, 312), (264, 272), (86, 91), (92, 95), (170, 173), (239, 242), (107, 110), (550, 557), (194, 197), (383, 388), (393, 396), (434, 437), (363, 367), (283, 289), (389, 392), (407, 413), (546, 549), (449, 452), (429, 433), (247, 251), (290, 294), (584, 587), (445, 448), (528, 532), (348, 351), (558, 561)]

def get_ican_obj_id(hico_id):
    for i in range(len(detection2hico)):
        if detection2hico[i] == hico_id:
            return i
    return -1

def get_object_hoi(hico_obj_list):
    obj_list = []
    for element in hico_obj_list:
        obj_list.append(get_ican_obj_id(int(element)))
    res = []

    for element in obj_list:
        for i in range(get_hoi_list_by_ob_id[element][0]+1, get_hoi_list_by_ob_id[element][1]+2):
            res.append(i)
    return res


# each line : object_id det_score, y1 x1 y2 x2  ;  x1 y1 x2 y2
def read_xml_file(json_file):
    cls_list = []
    score_list = []
    rois_list = []

    with open(json_file, 'r') as ff:
        ff.readline()
        while True:
            line1 = ff.readline()
            if not line1:
                break
            line2 = ff.readline()
            line3 = ff.readline()
            tmp_bbox = [eval(line1.split(',')[0]), eval(line1.split(',')[1]), eval(line1.split(',')[2]), eval(line1.split(',')[3])]
            score = eval(line2.rstrip('\n'))
            cls = get_hico_object_id(int(line3.split(',')[0]))
            rois_list.append(tmp_bbox)
            if cls not in cls_list:
                cls_list.append(cls)
            score_list.append(score)
    return cls_list, score_list, rois_list


#change the mask-rcnn result object-id to hico_det object-id
def get_hico_object_id(mrcnn_object_no):
    if mrcnn_object_no == 0:
        exit(1)
    return detection2hico[mrcnn_object_no]

# mask rcnn detection result folder
detection_result_folder = './data/hico/boxes'

fin = open("./data/hico/test_filelist.txt",'r')
test_filelist = [name.strip('\n').split('/')[1] for name in fin.readlines()]

def obj_mask_func(filename):
    output_lists = np.loadtxt(filename, delimiter = ',', dtype=str, usecols=range(600))
    output_lists = output_lists.astype(np.float)
    min_value = min([min(lis) for lis in output_lists])
    ff = open(os.path.join(filename), 'w')
    for i in range(len(test_filelist)):
        obj_classes = read_xml_file(os.path.join(detection_result_folder, test_filelist[i]+'.txt'))[0]
        hoi_related_list = get_object_hoi(obj_classes)
        for j in range(3):
            for pi in range(600):
                if pi+1 not in hoi_related_list:
                    output_lists[i*3+j][pi] = min_value
        for pi in range(600):
            ff.write(str(max(output_lists[i*3][pi], output_lists[i*3+1][pi], output_lists[i*3+2][pi])))
            ff.write(',')
        ff.write('\n')
    ff.close()
    

def prior_info(filename, object_file, verb_file, scale1, scale2, result_file):
    a1 = np.loadtxt(filename, delimiter = ',', dtype = str,usecols=range(600))
    arr1 = a1.astype(np.float)
    a2 = np.loadtxt(object_file, delimiter = ',', dtype = str,usecols=range(600))
    arr2 = a2.astype(np.float)
    a3 = np.loadtxt(verb_file, delimiter = ',', dtype = str,usecols=range(600))
    arr3 = a3.astype(np.float)

    a4 = np.zeros(arr1.shape,dtype=float)
    for i in range(arr1.shape[0]):
        for j in range(arr2.shape[1]):
            a4[i][j] = arr1[i][j] + arr2[i][j] * 0.01 * scale1
    for i in range(arr1.shape[0]):
        for j in range(arr3.shape[1]):
            a4[i][j] = a4[i][j] + arr3[i][j] * 0.01 * scale2
    res = open(result_file, "w")
    for i in range(arr1.shape[0]):
        for j in range(arr2.shape[1]):
            res.write(str(a4[i][j]))
            res.write(',')
        res.write('\n')
    res.close()
