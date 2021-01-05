#-*-coding:utf-8-*-
# date:2020-05-21
# Author: x.l.eric
# function:

import os
import json
import cv2
import sys
sys.path.append('./')

from utils.common_utils import *

import glob
import math
import os
import random
import shutil
from pathlib import Path
from PIL import Image
# import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data_iter_au.data_agu import *
import copy
# 图像白化
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y

# 图像亮度、对比度增强
def contrast_img(img, c, b):  # 亮度就是每个像素所有通道都加上b
    rows, cols, channels = img.shape
    # 新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=uint8)
    blank = np.zeros([rows, cols, channels], img.dtype)
    dst = cv2.addWeighted(img, c, blank, 1-c, b)
    return dst

class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, ops, img_size=(256,256), flag_agu = False):
        print('img_size (height,width) : ',img_size[0],img_size[1])
        print('train_path : ',ops.train_path)

        images_list = []
        msgs_list = []
        annos_list = []

        au_f_ct_dict = {}
        use_au_f_ct_dict = {}

        idx =  0
        zero_idx = 0
        use_zero_idx = 0
        diff_f = [1.,2.,3.]
        all_idx = 0

        gg= False
        for doc in os.listdir(ops.train_path):
            path_doc = ops.train_path+doc + '/'
            if gg :
                break
            for person_ in os.listdir(path_doc):
                path_person = path_doc + person_ + '/'
                # print(os.listdir(path_person))
                feature_dict = {}
                t_dict = {}
                p_idx = 0
                flag_break = False
                if gg:
                    break
                for file in os.listdir(path_person):
                    # print(path_person)
                    # print(file)
                    if '.jpg' in file:
                        if not os.path.exists(path_person + file.replace('.jpg','.json')):
                            continue
                        if not os.path.exists(path_person + file.replace('.jpg','.auw')):
                            continue

                        r_ = open(path_person +file.replace('.jpg','.auw'),'r')
                        lines = r_.readlines()
                        print('lines ---------------------------------->>>',lines)
                        gt_label = lines[0].strip().split(' ')
                        print(len(gt_label),'gt_label ---------------------------------->>>',gt_label)

                        if len(gt_label)!=24:
                            continue
                        p_idx += 1
                        au_feature = []
                        au_feature_output = []
                        for i in range(len(gt_label)):
                            au_feature.append(float(gt_label[i]))
                            au_feature_output.append(float(gt_label[i]))
                            if i not in au_f_ct_dict.keys():
                                au_f_ct_dict[i] = 0
                            else:
                                if float(gt_label[i]) != 0.:
                                    au_f_ct_dict[i] += 1

                        fs = open(path_person + file.replace('.jpg','.json'), encoding='utf-8')
                        json_s = json.load(fs)
                        fs.close()

                        all_idx += 1

                        if all_idx>1600:
                            gg =True
                            break
                        #-----------------------------------------------------------------------------

                        # if all_idx >1610:
                        #     gg = True
                        #     break
                        # # for key in json_s.keys():
                        # #     print(key)
                        # img_raw = cv2.imread(path_person +file)
                        # draw_contour(img_raw,json_s['dict_landmarks'],json_s['face_pts'],json_s['r_bboxes'],flag_agu= True)
                        # cv2.waitKey(1)
                        #
                        # images_list.append(path_person +file)
                        # msgs_list.append(json_s)
                        # annos_list.append(au_feature)
                        #
                        # continue
                        #-----------------------------------------------------------------------------
                        #
                        if np.sum(au_feature[0:2]) !=0 and np.sum(au_feature[0:2])==np.sum(au_feature) and (all_idx%60 != 0):# 眼睛滤波
                            continue
                        if np.sum(au_feature) !=0 :
                            diff_f[2] = diff_f[1]
                            diff_f[1] = diff_f[0]
                            diff_f[0] = np.sum(au_feature)
                            print('diff_ f : ',diff_f)
                            if (diff_f[0] == diff_f[1] == diff_f[2]):
                                if all_idx%3 ==0:
                                    continue
                        #
                        if np.sum(au_feature) == 0.:
                            zero_idx += 1
                        if np.sum(au_feature) == 0.:
                            if zero_idx%15 != 0:
                                continue
                            else:# 放一部分自然姿态的人脸
                                use_zero_idx += 1
                                pass
                        #
                        idx += 1
                        print(' {}) {} :  {} '.format(idx,path_person + file,len(au_feature)))
                        print('au_feature sum',np.sum(au_feature),'  use_zero_idx : ',use_zero_idx)
                        print('au_f_ct_dict : ',au_f_ct_dict)

                        for i in range(len(gt_label)):
                            if i not in use_au_f_ct_dict.keys():
                                use_au_f_ct_dict[i] = 0
                            else:
                                if float(gt_label[i]) != 0.:
                                    use_au_f_ct_dict[i] += 1
                        print('------------------------------------------------------------------------------')
                        print('use_au_f_ct_dict : ',use_au_f_ct_dict)

                        images_list.append(path_person +file)
                        msgs_list.append(json_s)
                        annos_list.append(au_feature_output)

                        # 以下都为重采样
                        if (np.sum(au_feature)> 0.) and (np.sum(au_feature[0:2]) ==0.) and (np.sum(au_feature[11:13]) ==0.):# 重采样
                            images_list.append(path_person +file)
                            msgs_list.append(json_s)
                            annos_list.append(au_feature_output)

                            for i in range(len(gt_label)):
                                if i not in use_au_f_ct_dict.keys():
                                    use_au_f_ct_dict[i] = 0
                                else:
                                    if float(gt_label[i]) != 0.:
                                        use_au_f_ct_dict[i] += 1
                        #特定区域重采样
                        if (np.sum(au_feature)> 0.) and (np.sum(au_feature[9:11]) == np.sum(au_feature)):# 下嘴唇向左或右重采样
                            images_list.append(path_person +file)
                            msgs_list.append(json_s)
                            annos_list.append(au_feature_output)

                            for i in range(len(gt_label)):
                                if i not in use_au_f_ct_dict.keys():
                                    use_au_f_ct_dict[i] = 0
                                else:
                                    if float(gt_label[i]) != 0.:
                                        use_au_f_ct_dict[i] += 1
                        if (np.sum(au_feature)> 0.) and ((au_feature[9]+au_feature[10]+au_feature[17]+au_feature[18]+au_feature[20]+au_feature[22]+au_feature[23])\
                        == np.sum(au_feature)):#
                            for kk in range(3):
                                images_list.append(path_person +file)
                                msgs_list.append(json_s)
                                annos_list.append(au_feature_output)

                                for i in range(len(gt_label)):
                                    if i not in use_au_f_ct_dict.keys():
                                        use_au_f_ct_dict[i] = 0
                                    else:
                                        if float(gt_label[i]) != 0.:
                                            use_au_f_ct_dict[i] += 1
                        list_repeat = [9,10,17,18,20,22,23]
                        for tt in list_repeat:

                            if (np.sum(au_feature)> 0.) and (au_feature[tt]== np.sum(au_feature)):#
                                for kk in range(3):
                                    images_list.append(path_person +file)
                                    msgs_list.append(json_s)
                                    annos_list.append(au_feature_output)

                                    for i in range(len(gt_label)):
                                        if i not in use_au_f_ct_dict.keys():
                                            use_au_f_ct_dict[i] = 0
                                        else:
                                            if float(gt_label[i]) != 0.:
                                                use_au_f_ct_dict[i] += 1




        self.images_list = images_list
        self.msgs_list = msgs_list
        self.annos_list = annos_list


        self.img_size = img_size
        self.flag_agu = flag_agu

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):


        img_path = self.images_list[index]
        json_s = self.msgs_list[index]
        au_feature = self.annos_list[index]

        img = cv2.imread(img_path)  # BGR

        # cv_resize_model = [cv2.INTER_LINEAR,cv2.INTER_CUBIC,cv2.INTER_NEAREST,cv2.INTER_AREA]

        img_,_ = draw_contour(img,json_s['dict_landmarks'],json_s['face_pts'],json_s['r_bboxes'],flag_agu= True,img_size = self.img_size[0],draw_flag = False)
        # print(img_.shape,au_feature)
        # cv2.imshow('crop',img_)
        # cv2.waitKey(0)

        # 随机镜像
        if random.randint(0,1) == 0 and self.flag_agu == True:

            img_ = cv2.flip(img_,1)
            # print('a -->> :',au_feature)
            au_feature_reg = copy.deepcopy(au_feature)
            # 左/右眼睛闭合
            au_feature_reg[0] = au_feature[1]
            au_feature_reg[1] = au_feature[0]
            # 左/右眼睑提升
            au_feature_reg[2] = au_feature[3]
            au_feature_reg[3] = au_feature[2]
            # 左/右眉毛下压
            au_feature_reg[4] = au_feature[5]
            au_feature_reg[5] = au_feature[4]
            # 左/右眉毛提升
            au_feature_reg[6] = au_feature[7]
            au_feature_reg[7] = au_feature[6]
            # 下嘴唇向左/向右
            au_feature_reg[9] = au_feature[10]
            au_feature_reg[10] = au_feature[9]
            # 左/右嘴角上扬
            au_feature_reg[11] = au_feature[12]
            au_feature_reg[12] = au_feature[11]
            # 左/右嘴角外展
            au_feature_reg[13] = au_feature[14]
            au_feature_reg[14] = au_feature[13]

            au_feature = au_feature_reg

            # print('b -->> :', au_feature)


        if self.flag_agu == True:
            if random.randint(0,5)==0:
                c = float(random.randint(80,120))/100.
                b = random.randint(-10,10)
                img_ = contrast_img(img_, c, b)

        if self.flag_agu == True:
            if random.randint(0,6)==1:
                # print('agu hue ')
                img_hsv=cv2.cvtColor(img_,cv2.COLOR_BGR2HSV)
                hue_x = random.randint(-10,10)
                # print(cc)
                img_hsv[:,:,0]=(img_hsv[:,:,0]+hue_x)
                img_hsv[:,:,0] =np.maximum(img_hsv[:,:,0],0)
                img_hsv[:,:,0] =np.minimum(img_hsv[:,:,0],180)#范围 0 ~180
                img_=cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)

        # if self.flag_agu == True:
        #     if random.randint(0,9)==1:
        #         img_ = img_agu_channel_same(img_)

        # cv2.namedWindow('crop',0)
        # cv2.imshow('crop',img_)
        # cv2.waitKey(0)
        # img_ = prewhiten(img_)
        img_ = img_.astype(np.float32)
        img_ = (img_-128.)/256.

        # img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        # print('1) ',img_[0:5,0:5,0])
        # img_ = img_ - [123.67, 116.28, 103.53]
        # print('2) ',img_[0:5,0:5,0])

        # print(img_)
        # input_img = input_img.astype(np.float32) / 256.0
        # input_img = np.expand_dims(input_img, 0)

        img_ = img_.transpose(2, 0, 1)
        au_feature = np.array(au_feature)
        # print('au_feature: ',au_feature)
        # landmarks_ = np.expand_dims(landmarks_,0)

        # print(img_.shape,landmarks_.shape)

        return img_,au_feature

def define_sequence_datasets(path):
    person_list = []
    person_idx = 0
    for doc in os.listdir(path):

        for person in os.listdir(path+doc):
            person_idx += 1
            image_list = []
            for file in os.listdir(path+doc+'/'+person):
                if '.jpg' in file:
                    image_list .append(path+doc + '/' + person +'/' + file)

            image_list.sort(key=lambda x: int(x.split('/')[-1].strip('.jpg')), reverse=False)
            # print(image_list)
            person_list.append(image_list)
            print('  define_sequence_datasets - >> idx {} person have {} images '.format(person_idx,len(image_list)))
    print('define_sequence_datasets  -- >> person num : ',len(person_list))\

    return person_list

def sample_sequence_datasets(p_list,person_num = 4,debug = False):
    idx = list(range(0,len(p_list)-1))
    random.shuffle(idx)
    if debug:
        print('----------------------------------------->>>')
    p_idx = 0

    train_images = []
    train_features = []
    for i in range(person_num):
        if debug:
            print('------------->>>')
        # 选定人
        p_list[idx[i]]

        strat_idx = random.randint(0,len(p_list[idx[i]])-6*6-1)

        for j in range(strat_idx,strat_idx + 6*6,4):
            p_idx += 1
            if debug:
                print('{}) personIdx - {} image_idx {}    : {}'.format(p_idx,idx[i],j,p_list[idx[i]][j]))

            #-------------------------------
            if not os.path.exists(p_list[idx[i]][j].replace('.jpg','.json')):
                continue
            if not os.path.exists(p_list[idx[i]][j].replace('.jpg','.auw')):
                continue

            r_ = open(p_list[idx[i]][j].replace('.jpg','.auw'),'r')
            lines = r_.readlines()
            if debug:
                print('lines ---------------------------------->>>',lines)
            gt_label = lines[0].strip().split(' ')
            # print(len(gt_label),'gt_label ---------------------------------->>>',gt_label)

            au_feature_output = []
            for ii in range(len(gt_label)):
                au_feature_output.append(float(gt_label[ii]))

            fs = open(p_list[idx[i]][j].replace('.jpg','.json'), encoding='utf-8')
            json_s = json.load(fs)
            fs.close()

            #-------------------------------------------------------------
            img = cv2.imread(p_list[idx[i]][j])  # BGR

            # cv_resize_model = [cv2.INTER_LINEAR,cv2.INTER_CUBIC,cv2.INTER_NEAREST,cv2.INTER_AREA]

            img_,_ = draw_contour(img,json_s['dict_landmarks'],json_s['face_pts'],json_s['r_bboxes'],flag_agu= True,img_size = 256,draw_flag = False)

            if random.randint(0,5)==0:
                c = float(random.randint(80,120))/100.
                b = random.randint(-10,10)
                img_ = contrast_img(img_, c, b)


            if random.randint(0,6)==1:
                # print('agu hue ')
                img_hsv=cv2.cvtColor(img_,cv2.COLOR_BGR2HSV)
                hue_x = random.randint(-10,10)
                # print(cc)
                img_hsv[:,:,0]=(img_hsv[:,:,0]+hue_x)
                img_hsv[:,:,0] =np.maximum(img_hsv[:,:,0],0)
                img_hsv[:,:,0] =np.minimum(img_hsv[:,:,0],180)#范围 0 ~180
                img_=cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)




            #cv2.namedWindow('imgg',0)
            #cv2.imshow('imgg',img_)
            #cv2.waitKey(1)

            # img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
            # img_ = img_ - [123.67, 116.28, 103.53]

            img_ = img_.astype(np.float32)
            img_ = (img_-128.)/256.

            train_images.append(img_.transpose(2, 0, 1))
            train_features.append(au_feature_output)
    train_images = np.array(train_images)
    train_features = np.array(train_features)
    if debug:
        print('train_images - train_features shape : ',train_images.shape,train_features.shape)

    train_images = torch.from_numpy(train_images).float().cuda()
    train_features = torch.from_numpy(train_features).float().cuda()
    if debug:
        print('train_images - train_features size : ',train_images.size(),train_features.size())

    return train_images,train_features


if __name__ == '__main__':
    path_ = './FEAFA/datasets/'
    pp_idx = 0
    for doc in os.listdir(path_):
        path_doc = path_+doc + '/'

        for person_ in os.listdir(path_doc):
            path_person = path_doc + person_ + '/'
            # print(os.listdir(path_person))
            feature_dict = {}
            t_dict = {}
            p_idx = 0
            flag_break = False
            for file in os.listdir(path_person):
                # print(path_person)
                # print(file)
                if '.jpg' in file:
                    if not os.path.exists(path_person + file.replace('.jpg','.json')):
                        continue
                    if not os.path.exists(path_person + file.replace('.jpg','.auw')):
                        continue
                    img_raw = cv2.imread(path_person +file)
                    pp_idx += 1

                    if pp_idx %120 !=0:
                        continue

                    fs = open(path_person + file.replace('.jpg','.json'), encoding='utf-8')
                    json_s = json.load(fs)
                    fs.close()

                    # for key in json_s.keys():
                    #     print(key)
                    face_crop,dict_landmarks_align = draw_contour(img_raw,json_s['dict_landmarks'],json_s['face_pts'],json_s['r_bboxes'],flag_agu= True,img_size = 112,draw_flag= False)
                    cv2.namedWindow('face_cropxx',0)
                    cv2.imshow('face_cropxx',face_crop)
                    cv2.waitKey(1)

                    cv2.imwrite('./quan_face/image_{}.jpg'.format(pp_idx),cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

                    print(pp_idx,path_person +file)
                    if pp_idx%100 == 0:
                        cv2.namedWindow('image',0)
                        cv2.imshow('image',img_raw)
                        cv2.waitKey(1)
    cv2.destroyAllWindows()
