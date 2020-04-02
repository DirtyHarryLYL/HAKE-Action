#### Pipeline for generating part box, including how to load alphapose results, sub-pose generation, how to run alphapose.
#For specific task, the data format should be modified.
"""
- How to load alphapose results: Use **load_pose_from_alphapose** to load pose results. **map_17_to_16** means mapping 17 pose points to 16 pose points; **output_part_box** includes the implementation of rule-based part box generation.
- sub-pose generation: Use **load_write_pkl** to generate. For details, see the comments.
- how to run alphapose: Download AlphaPose, run orders like **python3 demo.py --indir image_folder --outdir xxx/pose_results/ --conf 0.05 --fast_inference False**
- Sub image coordinate to image coordinate: Convert the sub image coordinate to original image\'s coordinate
"""

import math
import json as js
import os
import pickle
import scipy.io as sio
from PIL import Image
import numpy as np

'''
keypoint order

there are two set of keypoint order: coco results (17 joints) & mpii results (16 joints)

coco results (17 joints)
{0,  "Nose"}, {1,  "LEye"}, {2,  "REye"}, {3,  "LEar"}, {4,  "REar"}, {5,  "LShoulder"}, {6,  "RShoulder"},
{7,  "LElbow"}, {8,  "RElbow"}, {9,  "LWrist"}, {10, "RWrist"}, {11, "LHip"}, 
{12, "RHip"}, {13, "LKnee"}, {14, "Rknee"}, {15, "LAnkle"}, {16, "RAnkle"}

MPII results (16 body parts)
{0,  "RAnkle"}, {1,  "Rknee"}, {2,  "RHip"}, {3,  "LHip"}, {4,  "LKnee"}, {5,  "LAnkle"},
{6,  "Pelv"}, {7,  "Thrx"}, {8,  "Neck"}, {9,  "Head"}, {10, "RWrist"}, {11, "RElbow"},
{12, "RShoulder"}, {13, "LShoulder"}, {14, "LElbow"}, {15, "LWrist"}

'''
def map_17_to_16(joint_17, flag_score = 1):

    joint_16 = []
    dict_map = {0:16, 1:14, 2:12, 3:11, 4:13, 5:15, 6:[11, 12], 7:[5,6], 9:[1,2], 10:10, 11:8, 12:6, 13:5, 14:7, 15:9}
    for idx in range(16):
        joint_16.append({})
        if idx == 8:
            continue # deal thrx joint later
        elif idx in [6, 7, 9]:
            #calc Pelv joint from two hip joint
            #calc neck joint from two shoulder joint
            #calc head joint from two eye joint
            joint_16[idx]['x'] =(joint_17[dict_map[idx][0]]['x'] + joint_17[dict_map[idx][1]]['x']) * 0.5
            joint_16[idx]['y'] =(joint_17[dict_map[idx][0]]['y'] + joint_17[dict_map[idx][1]]['y']) * 0.5
            if flag_score == 1:
                joint_16[idx]['score'] =(joint_17[dict_map[idx][0]]['score'] + joint_17[dict_map[idx][1]]['score']) * 0.5
        else:
            joint_16[idx] = joint_17[dict_map[idx]]
    #calc thrx joint from head joint and neck, assume the distance is 1:3
    joint_16[8]['x'] = joint_16[7]['x'] * 0.75  + joint_16[9]['x'] * 0.25
    joint_16[8]['y'] = joint_16[7]['y'] * 0.75  + joint_16[9]['y'] * 0.25
    if flag_score == 1:
        joint_16[8]['score'] = joint_16[7]['score'] * 0.75  + joint_16[9]['score'] * 0.25

    return joint_16

'''
1. image_id: It is like "0.jpg"
2. image_bbox: [x1, y1, x2, y2], for 0 to width/height
3. part_bbox: generated from output_part_box
4. joint_17: 17 joints, {x, y, score}
5. joint_16: 16 joints
Output format: {file_name: [{'image_id':, 'image_bbox': , 'part_bbox': , 'joint_17': , 'joint_16': }]}
'''
def load_pose_from_alphapose(alphapose_file):

    print('Loading json from alphapose result')

    with open(alphapose_file,'r') as jsonfile:
        jsfile = js.load(jsonfile)
        total_length = len(jsfile)
        print('The total length is ' + str(total_length))
        part_box_dict = {}

        for id in range(total_length):
            file_name = jsfile[id]['folder'] + '/' + jsfile[id]['image_id'] # each file name
            bbox = jsfile[id]['bboxes']
            bbox_score = jsfile[id]['bbox_score']
            if float(bbox_score) < 0.9:
                continue
            pose_point = jsfile[id]['keypoints']
            joint = []
            for i in range(17):
                joint.append({})
                joint[i] = {'x': pose_point[3 * i], 'y': pose_point[3 * i + 1], 'score': pose_point[3 * i + 2]}

            joint_16 = map_17_to_16(joint)
            image_bbox = {'x1':bbox[0], 'x2': bbox[2], 'y1': bbox[1], 'y2': bbox[3]}

            part_bbox = output_part_box(joint_16,image_bbox)
            tmp = {}
            tmp['image_id'] = file_name
            tmp['image_bbox'] = image_bbox
            tmp['part_bbox'] = part_bbox
            tmp['joint_17'] = joint
            tmp['joint_16'] = joint_16 
            if file_name not in part_box_dict:
                part_box_dict[file_name] = []
            part_box_dict[file_name].append(tmp)
    return part_box_dict

# Rule based part box
def output_part_box(joint, img_bbox): 
    flag_bad_joint = 0
    # 16 part names correspond to the center of 16 input joint
    part_size = [1.2, 1, 1, 1.2, 1.2, 1.2, 0.9, 1, 1, 0.9]
    part = [0, 1, 4, 5, 6, 9, 10, 12, 13, 15]

    height = get_distance(joint, 6, 8)

    if (joint[8]['score'] < 0.2) or (joint[6]['score'] < 0.2):
        flag_bad_joint = 1

    group_score_head = (joint[7]['score'] + joint[8]['score'] + joint[9]['score']) / 3
    group_score_left_arm = (joint[13]['score'] + joint[14]['score'] + joint[15]['score']) / 3
    group_score_right_arm = (joint[10]['score'] + joint[11]['score'] + joint[12]['score']) / 3
    group_score_left_leg = (joint[3]['score'] + joint[4]['score'] + joint[5]['score']) / 3
    group_score_right_leg = (joint[0]['score'] + joint[1]['score'] + joint[2]['score']) / 3


    # 'Pelv'&'Neck' scaling by the distance of Pelv and Neck
    bbox = [None] * 10
    for i in range(10):
        bbox[i] = {'x1': 0, 'x2': 0, 'y1': 0, 'y2': 0}
    for i in range(10):
        score_joint = joint[part[i]]['score']
        # the keypoint is not reliable/ cannot be seen / do not exist
        if (score_joint < 0.2):
            bbox[i]['x1'] = img_bbox['x1']
            bbox[i]['y1'] = img_bbox['y1']
            bbox[i]['x2'] = img_bbox['x2']
            bbox[i]['y2'] = img_bbox['y2']

        # the keypoint is reliable, but the distance cannot be measured by distance between pelv and neck
        elif (score_joint >= 0.2) and (flag_bad_joint == 1):
            if i == 5: # head group
                if group_score_head > 0.2:
                    half_box_width = get_distance(joint, 7, 9)
                    bbox[i]['x1'], bbox[i]['y1'], bbox[i]['x2'], bbox[i]['y2'] = get_part_box(i, joint, half_box_width)
                else:
                    bbox[i]['x1'], bbox[i]['y1'], bbox[i]['x2'], bbox[i]['y2'] = return_image_box(img_bbox)
            elif i in [6,7]: # right arm group
                if group_score_right_arm > 0.2: 
                    half_box_width = get_distance(joint, 10, 12)
                    bbox[i]['x1'], bbox[i]['y1'], bbox[i]['x2'], bbox[i]['y2'] = get_part_box(i, joint, half_box_width)
                else:
                    bbox[i]['x1'], bbox[i]['y1'], bbox[i]['x2'], bbox[i]['y2'] = return_image_box(img_bbox)
            elif i in [9,10]: # left arm group
                if group_score_left_arm > 0.2: 
                    half_box_width = get_distance(joint, 13, 15)
                    bbox[i]['x1'], bbox[i]['y1'], bbox[i]['x2'], bbox[i]['y2'] = get_part_box(i, joint, half_box_width)
                else:
                    bbox[i]['x1'], bbox[i]['y1'], bbox[i]['x2'], bbox[i]['y2'] = return_image_box(img_bbox)
            elif i in [0,1]: # right leg group
                if group_score_right_leg > 0.2: 
                    half_box_width = get_distance(joint, 0, 2)
                    bbox[i]['x1'], bbox[i]['y1'], bbox[i]['x2'], bbox[i]['y2'] = get_part_box(i, joint, half_box_width)
                else:
                    bbox[i]['x1'], bbox[i]['y1'], bbox[i]['x2'], bbox[i]['y2'] = return_image_box(img_bbox)
            elif i in [2,3]: # right arm group
                if group_score_left_leg > 0.2: 
                    half_box_width = get_distance(joint, 3, 5)
                    bbox[i]['x1'], bbox[i]['y1'], bbox[i]['x2'], bbox[i]['y2'] = get_part_box(i, joint, half_box_width)
                else:
                    bbox[i]['x1'], bbox[i]['y1'], bbox[i]['x2'], bbox[i]['y2'] = return_image_box(img_bbox)
            else: # pelv keypoint
                bbox[i]['x1'], bbox[i]['y1'], bbox[i]['x2'], bbox[i]['y2'] = return_image_box(img_bbox)
    
        else: # the keypoint is reliable and the distance can be measured by distance between pelv and neck
            half_box_width = height * part_size[i] / 2
            bbox[i]['x1'], bbox[i]['y1'], bbox[i]['x2'], bbox[i]['y2'] = get_part_box(i, joint, half_box_width)
    return bbox


def get_part_box(i, joint, half_box_width):
    part = [0, 1, 4, 5, 6, 9, 10, 12, 13, 15]
    center_x = joint[part[i]]['x']
    center_y = joint[part[i]]['y']
    return center_x - half_box_width, center_y - half_box_width, center_x + half_box_width, center_y + half_box_width

def get_distance(joint, keypoint1, keypoint2):
    height_y = joint[keypoint1]['y'] - joint[keypoint2]['y'] 
    height_x = joint[keypoint1]['x'] - joint[keypoint2]['x']
    return math.sqrt(height_x ** 2 + height_y ** 2)

def return_image_box(img_bbox):
    return img_bbox['x1'], img_bbox['y1'], img_bbox['x2'], img_bbox['y2']


# pkl_file includes object bounding boxes
# part_bbox_det is the pose results of alphapose
# match_iou_thres is the threshold of box matching, normally 0.7
# dest_dir is the folder to save results
# save_sub_flag is the flag of whether save sub images. Normally set as True on the first time and False on the second time. 
# When set as True, it will call function cut_sub_image_save_pos to save sub images. When set as False, it will match the alphapose results of sub images.
# If still not matched, the box of whole person will be assigned as the part box.
def load_write_pkl(pkl_file, part_bbox_det, match_iou_thres, dest_dir, save_sub_flag, sub_pose_dir):
    all_image_not_paired = {}
    count = 0
    matched_num = 0
    not_matched_num = 0
    # for each box to be matched
    sub_img_dict = {} 
    for pkl_img in pkl_file:
        record = pkl_file[pkl_img]
        if count % 100 == 0:
            print(count)
        count += 1
        img_name = pkl_img # the key of pkl_file is the image_name
        candidates = []
        try:
            candidates = part_bbox_det[img_name] # alphapose boxes in this image
        except:
            pass
        
        for hbox in record:
            # If the part box is assigned, continue
            if len(pkl_file[pkl_img][hbox]) == 5: # ==5 means the part box is assigned
                continue
            # select max_iou based matching
            maxiou = 0
            max_index = -1
            for i in range(len(candidates)):
                if 'matched' not in candidates:
                    candi_hbox = candidates[i]['image_bbox']
                    iou = check_iou(hbox, candi_hbox)
                    if iou >= match_iou_thres and iou > maxiou:
                        maxiou = iou
                        max_index = i
            # if has matched
            if max_index != -1:
                matched_num += 1
                candidates[max_index]['matched'] = True # a flag to indicate that this candicate is matched
                pkl_file[pkl_img][hbox].append(candidates[max_index]['part_bbox'])
                pkl_file[pkl_img][hbox].append(candidates[max_index]['joint_17'])
                pkl_file[pkl_img][hbox].append(candidates[max_index]['joint_16'])
            # not matched
            else:
                if save_sub_flag == True:
                    cut_sub_image_save_pos(img_name, hbox, all_image_not_paired, sub_pose_dir, sub_img_dict)
                else: # assign whole person box
                    tmp_part_bbox = [{'x1': hbox[0], 'y1': hbox[1], 'x2': hbox[2], 'y2': hbox[3]}] * 10
                    tmp_joint_17 = [{'x': -1, 'y': -1, 'score': 0}] * 17
                    tmp_joint_16 = [{'x': -1, 'y': -1, 'score': 0}] * 16
                    pkl_file[pkl_img][hbox].append(tmp_part_bbox)
                    pkl_file[pkl_img][hbox].append(tmp_joint_17)
                    pkl_file[pkl_img][hbox].append(tmp_joint_16)
                not_matched_num += 1
    with open(os.path.join(dest_dir),'wb') as f:
        pickle.dump(pkl_file,f)
    f.close()
    # save the [x1, y1, x2, y2] of the cropped sub box
    with open('./result.json', 'w') as f:
        jStr = js.dumps(sub_img_dict)
        f.write(jStr)
        f.close()
    print('Matched: ', matched_num)
    print('Not Matched: ', not_matched_num)


def cut_sub_image_save_pos(img_filename, human_bbox, all_image_not_paired, image_dir_save, sub_img_dict):
    image_dir = 'xxx'
    if not os.path.exists(image_dir_save):
        os.mkdir(image_dir_save)

    img_to_be_cut =  Image.open(os.path.join(image_dir,img_filename))
    bbox_width = human_bbox[2] - human_bbox[0]
    bbox_height = human_bbox[3] - human_bbox[1]
    crop_coordinate = None
    try:
        x1 = min(max(0, (human_bbox[0] - round(0.1 * bbox_width))), width-1)
        y1 = min(max(0, (human_bbox[1] - round(0.1 * bbox_height))), height-1)
        x2 = min(max(0, (human_bbox[2] + round(0.1 * bbox_width))), width-1)
        y2 = min(max(0, (human_bbox[3] + round(0.1 * bbox_height))), height-1)
        img_cutted = img_to_be_cut.crop((x1, y1, x2, y2))
        crop_coordinate = [x1, y1, x2, y2]
    except:
        x1 = min(max(0, human_bbox[0]), width-1)
        y1 = min(max(0, human_bbox[1]), height-1)
        x2 = min(max(0, human_bbox[2]), width-1)
        y2 = min(max(0, human_bbox[3]), height-1)
        img_cutted = img_to_be_cut.crop((x1, y1, x2, y2))
        crop_coordinate = [x1, y1, x2, y2]
    else:
        pass
    if img_filename not in all_image_not_paired:
        all_image_not_paired[img_filename] = 0
    img_cutted.save(os.path.join(image_dir_save,img_filename.replace('/', '.') + '_human_' + str(all_image_not_paired[img_filename]).replace('/', '.') +'.jpg'))
    sub_img_dict[img_filename.replace('/', '.') + '_human_' + str(all_image_not_paired[img_filename]).replace('/', '.')+'.jpg'] = crop_coordinate
    all_image_not_paired[img_filename] += 1

# checking iou between two boxes
def check_iou(human_bbox_pkl, human_bbox_pose):
    x1, x2, y1, y2 = int(human_bbox_pose['x1']), int(human_bbox_pose['x2']), int(human_bbox_pose['y1']), int(human_bbox_pose['y2'])
    x1d, y1d, x2d, y2d = human_bbox_pkl

    xa = max(x1, x1d)
    ya = max(y1, y1d)
    xb = min(x2, x2d)
    yb = min(y2, y2d)

    iw1 = xb - xa + 1
    iw2 = yb - ya + 1

    if iw1 > 0 and iw2 > 0:
        inter_area = iw1 * iw2
        a_area = (x2 - x1) * (y2 - y1)
        b_area = (x2d - x1d) * (y2d - y1d)
        union_area = a_area + b_area - inter_area
        return inter_area / float(union_area)
    else:
        return 0

data_pkl = 'xxx.pkl'
pkl_file = pickle.load(open(data_pkl,"rb"))
part_bbox_from_gt = [] # no GT part bbox
part_bbox_det = load_pose_from_alphapose('./xxx_alphapose_all.json')
iou_thres = 0.7 # IoU thresold
dest_dir = './part_boxes.pkl'
save_sub_flag = True # save sub picture or not
sub_pose_dir = './xxx'

load_write_pkl(pkl_file , part_bbox_det, iou_thres, dest_dir, save_sub_flag, sub_pose_dir)