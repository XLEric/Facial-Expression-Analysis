#-*-coding:utf-8-*-
# date:2020-04-11
# Author: x.l.eric

import os
import shutil
import cv2
import numpy as np
import json
from headpose.pose import *
from imgaug import augmenters as iaa
import json
import random
import math

def mkdir_(path, flag_rm=False):
    if os.path.exists(path):
        if flag_rm == True:
            shutil.rmtree(path)
            os.mkdir(path)
            print('remove {} done ~ '.format(path))
    else:
        os.mkdir(path)

def plot_box(bbox, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)# 目标的bbox
    if label:
        tf = max(tl - 2, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0] # label size
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3 # 字体的bbox
        cv2.rectangle(img, c1, c2, color, -1)  # label 矩形填充
        # 文本绘制
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 4, [225, 255, 255],thickness=tf, lineType=cv2.LINE_AA)

class JSON_Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JSON_Encoder, self).default(obj)


#-------------------------------------------------------------------------------
# eye_left_n,eye_right_n:可能为扰动后的双眼原图坐标
# eye_left_gt_n,eye_right_gt_n:没有扰动后的双眼原图坐标
def face_alignment_aug_fun(imgn,eye_left_n,eye_right_n,eye_left_gt_n = None,eye_right_gt_n=None,nose_gt_n = None,\
facial_landmarks_n = None,\
angle = None,desiredLeftEye=(0.34, 0.42),desiredFaceWidth=256, desiredFaceHeight=None,flag_agu = False,draw_flag = False,\
gt_offset_x = 0,gt_offset_y = 0):

    if desiredFaceHeight is None:
        desiredFaceHeight = desiredFaceWidth
    if eye_left_gt_n is None:
        eye_left_gt_n = eye_left_n
    if eye_right_gt_n is None:
        eye_right_gt_n = eye_right_n

    leftEyeCenter = eye_left_n
    rightEyeCenter = eye_right_n
    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    if angle == None:
        angle = np.degrees(np.arctan2(dY, dX))
    else:
        angle += np.degrees(np.arctan2(dY, dX))#基于正对角度的扰动

    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    desiredRightEyeX = 1.0 - desiredLeftEye[0]
	# determine the scale of the new resulting image by taking
	# the ratio of the distance between eyes in the *current*
	# image to the ratio of distance between eyes in the
	# *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist
    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) / 2,(leftEyeCenter[1] + rightEyeCenter[1]) / 2)
    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
    # update the translation component of the matrix
    tX = desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    M_reg = np.zeros((3,3),dtype = np.float32)
    M_reg[0,:] = M[0,:]
    M_reg[1,:] = M[1,:]
    M_reg[2,:] = (0,0,1.)
    # print(M_reg)
    M_I = np.linalg.inv(M_reg)#矩阵求逆，从而获得，目标图到原图的关系
    # print(M_I)
    # apply the affine transformation
    (w, h) = (desiredFaceWidth, desiredFaceHeight)
    cv_resize_model = [cv2.INTER_LINEAR,cv2.INTER_CUBIC,cv2.INTER_NEAREST,cv2.INTER_AREA]
    if flag_agu:
        output = cv2.warpAffine(imgn, M, (w, h),flags=cv_resize_model[random.randint(0,3)],borderMode=cv2.BORDER_CONSTANT)# INTER_LINEAR INTER_CUBIC INTER_NEAREST
    else:
        output = cv2.warpAffine(imgn, M, (w, h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)#

    #---------------------------------------------------------------------------------------

    ptx1 = int(eye_left_gt_n[0]*M[0][0] + eye_left_gt_n[1]*M[0][1] + M[0][2])
    pty1 = int(eye_left_gt_n[0]*M[1][0] + eye_left_gt_n[1]*M[1][1] + M[1][2])

    ptx2 = int(eye_right_gt_n[0]*M[0][0] + eye_right_gt_n[1]*M[0][1] + M[0][2])
    pty2 = int(eye_right_gt_n[0]*M[1][0] + eye_right_gt_n[1]*M[1][1] + M[1][2])

    # ptx3 = int(nose_gt_n[0]*M[0][0] + nose_gt_n[1]*M[0][1] + M[0][2])
    # pty3 = int(nose_gt_n[0]*M[1][0] + nose_gt_n[1]*M[1][1] + M[1][2])
    if draw_flag == True:
        cv2.circle(output, (ptx1,pty1), np.int(1),(0,0,255), 1)
        cv2.circle(output, (ptx2,pty2), np.int(1),(0,0,255), 1)
    # cv2.circle(output, (ptx3,pty3), np.int(1),(0,255,0), 1)
    # print('facial_landmarks_n',len(facial_landmarks_n))
    dict_landmarks = {}
    if facial_landmarks_n is not None:
        for i in range(len(facial_landmarks_n)):
            x = facial_landmarks_n[i][0]+gt_offset_x
            y = facial_landmarks_n[i][1]+gt_offset_y

            ptx = int(x*M[0][0] + y*M[0][1] + M[0][2])
            pty = int(x*M[1][0] + y*M[1][1] + M[1][2])

            if 67>= i >=60:
                if 'left_eye' not in dict_landmarks.keys():
                    dict_landmarks['left_eye'] = []
                dict_landmarks['left_eye'].append([int(ptx),int(pty),(0,0,255)])


            elif 75>= i >=68:
                if 'right_eye' not in dict_landmarks.keys():
                    dict_landmarks['right_eye'] = []
                dict_landmarks['right_eye'].append([int(ptx),int(pty),(0,255,0)])

            if draw_flag == True:
                cv2.circle(output, (ptx,pty), np.int(1),(255,155,155), 1)

    return output,dict_landmarks


def image_aug_fun(imgn):

    if random.randint(-3,3)==0:#增强原图
        img_aug_list=[]
        img_aug_list.append(imgn.copy())

        if random.randint(-5,5)!=0:#单一方式增强
            # print(' --- 单一增强方式')
            stepx = random.randint(0,7)
            if stepx == 0:
                seq = iaa.Sequential([iaa.Sharpen(alpha=(0.0, 0.05), lightness=(0.85, 1.15))])
                # print('-------------------->>> imgaug 0 : sharp')
            elif stepx == 1:
                seq = iaa.Sequential([iaa.AverageBlur(k=(2))])# blur image using local means with kernel sizes between 2 and 4
                # print('-------------------->>> imgaug 1 : AverageBlur')
            elif stepx == 2:
                seq = iaa.Sequential([iaa.MedianBlur(k=(3))])# blur image using local means with kernel sizes between 3 and 5
                # print('-------------------->>> imgaug 2 : MedianBlur')
            elif stepx == 3:
                seq = iaa.Sequential([iaa.GaussianBlur((0, 0.8))])
                # print('-------------------->>> imgaug 3 : GaussianBlur')
            elif stepx == 4:
                seq = iaa.Sequential([iaa.ContrastNormalization((0.90, 1.10))])
                # print('-------------------->>> imgaug 4 : ContrastNormalization')
            elif stepx == 5:
                seq = iaa.Sequential([iaa.Add((-5, 5))])
                # print('-------------------->>> imgaug 5 : Add')
            elif stepx == 6:
                seq = iaa.Sequential([iaa.AddToHueAndSaturation((-10, 10),per_channel=True)])
                # print('-------------------->>> imgaug 6 : AddToHueAndSaturation')
            elif stepx == 7:
                seq = iaa.Sequential([iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.02*255), per_channel=False, name=None, deterministic=False, random_state=None)])
                # print('-------------------->>> imgaug 7 : AdditiveGaussianNoise')
        else:#复合增强方式
            # print(' *** 复合增强方式')
            # print('-------------------->>> 符合增强')
            seq = iaa.Sequential([
                iaa.Sharpen(alpha=(0.0, 0.05), lightness=(0.9, 1.1)),
                iaa.GaussianBlur((0, 0.8)),
                iaa.ContrastNormalization((0.9, 1.1)),
                iaa.Add((-5, 5)),
                iaa.AddToHueAndSaturation((-5, 5)),
            ])
        images_aug = seq.augment_images(img_aug_list)
    else:
        # print('-------------------->>> origin image')
        return imgn.copy()

    return images_aug[0].copy()
#-------------------------------------------------------------------------------
# img_path_n : 图片路径
# Eye_Left,Eye_Right： 原图中的双眼坐标
# desiredFaceWidth: 输出人脸图片边长
# desiredLeftEye ： 左眼在输出图片的比例
# flag_aug ：是否进行图片增强
def get_face_aug_fun(img_origin,facial_landmarks,Eye_Left,Eye_Right,Nose,desiredFaceWidth,desiredLeftEye,flag_agu=True,\
draw_flag = False,gt_offset_x = 0,gt_offset_y = 0):

    x1,y1 = Eye_Left[0],Eye_Left[1]
    x2,y2 = Eye_Right[0],Eye_Right[1]

    m_height = img_origin.shape[0]
    m_width = img_origin.shape[1]
    # imgn =img_origin.copy()

    if flag_agu == True:
        #--------------------------------------------------------------- 原图增强
        img_origin_copy = image_aug_fun(img_origin)
        #--------------------------------------------------------------- 对齐
        #扰动原图 landmarks pixel
        pixel_offset = 5
        pixel_offset_x = 8
        x1_random = x1 + random.randint(-pixel_offset_x,pixel_offset_x)
        y1_random = y1 + random.randint(-pixel_offset,pixel_offset)
        x2_random = x2 + random.randint(-pixel_offset_x,pixel_offset_x)
        y2_random = y2 + random.randint(-pixel_offset,pixel_offset)
        face_crop,dict_landmarks_align = face_alignment_aug_fun(img_origin_copy,eye_left_n = (x1_random,y1_random),eye_right_n = (x2_random,y2_random),
            eye_left_gt_n = (x1,y1),eye_right_gt_n = (x2,y2),
            facial_landmarks_n = facial_landmarks,
            nose_gt_n = Nose,
            # angle = m_angle,
            desiredFaceWidth=desiredFaceWidth, desiredFaceHeight=None,desiredLeftEye=desiredLeftEye,flag_agu = flag_agu,draw_flag = draw_flag,\
            gt_offset_x = gt_offset_x,gt_offset_y = gt_offset_y)
        # print(':--->>.fg_heat,bg_heat',fg_heat,bg_heat)
    elif flag_agu == False:
        # print('uuuuuuuuuuuuuuuuuuuuuukkkkkkkkkkkkkkkkkkkkkkkkppppppppppppppppppppmmmmmmmmmmmmmwwwwwwwww')
        face_crop,dict_landmarks_align = face_alignment_aug_fun(img_origin,eye_left_n = (x1,y1),eye_right_n = (x2,y2),
            facial_landmarks_n = facial_landmarks,
            nose_gt_n = Nose,
            angle = None,
            desiredFaceWidth=desiredFaceWidth, desiredFaceHeight=None,desiredLeftEye=desiredLeftEye,flag_agu=flag_agu,draw_flag = draw_flag,\
            gt_offset_x = gt_offset_x,gt_offset_y = gt_offset_y)

    return img_origin,face_crop,dict_landmarks_align

def draw_landmarks(img,output,draw_circle):
    img_width = img.shape[1]
    img_height = img.shape[0]
    dict_landmarks = {}
    face_pts = []
    for i in range(int(output.shape[0]/2)):
        x = output[i*2+0]*float(img_width)
        y = output[i*2+1]*float(img_height)
        face_pts .append([x,y])

        if 41>= i >=33:
            if 'left_eyebrow' not in dict_landmarks.keys():
                dict_landmarks['left_eyebrow'] = []
            dict_landmarks['left_eyebrow'].append([int(x),int(y),(0,255,0)])
            if draw_circle:
                cv2.circle(img, (int(x),int(y)), 2, (0,255,0),-1)
        elif 50>= i >=42:
            if 'right_eyebrow' not in dict_landmarks.keys():
                dict_landmarks['right_eyebrow'] = []
            dict_landmarks['right_eyebrow'].append([int(x),int(y),(0,255,0)])
            if draw_circle:
                cv2.circle(img, (int(x),int(y)), 2, (0,255,0),-1)
        elif 67>= i >=60:
            if 'left_eye' not in dict_landmarks.keys():
                dict_landmarks['left_eye'] = []
            dict_landmarks['left_eye'].append([int(x),int(y),(255,55,255)])
            if draw_circle:
                cv2.circle(img, (int(x),int(y)), 2, (255,0,255),-1)
        elif 75>= i >=68:
            if 'right_eye' not in dict_landmarks.keys():
                dict_landmarks['right_eye'] = []
            dict_landmarks['right_eye'].append([int(x),int(y),(255,55,255)])
            if draw_circle:
                cv2.circle(img, (int(x),int(y)), 2, (255,0,255),-1)
        elif 97>= i >=96:
            if draw_circle:
                cv2.circle(img, (int(x),int(y)), 2, (0,0,255),-1)
        elif 54>= i >=51:
            if 'bridge_nose' not in dict_landmarks.keys():
                dict_landmarks['bridge_nose'] = []
            dict_landmarks['bridge_nose'].append([int(x),int(y),(0,170,255)])
            if draw_circle:
                cv2.circle(img, (int(x),int(y)), 2, (0,170,255),-1)
        elif 32>= i >=0:
            if 'basin' not in dict_landmarks.keys():
                dict_landmarks['basin'] = []
            dict_landmarks['basin'].append([int(x),int(y),(255,30,30)])
            if draw_circle:
                cv2.circle(img, (int(x),int(y)), 2, (255,30,30),-1)
        elif 59>= i >=55:
            if 'wing_nose' not in dict_landmarks.keys():
                dict_landmarks['wing_nose'] = []
            dict_landmarks['wing_nose'].append([int(x),int(y),(0,255,255)])
            if draw_circle:
                cv2.circle(img, (int(x),int(y)), 2, (0,255,255),-1)
        elif 87>= i >=76:
            if 'out_lip' not in dict_landmarks.keys():
                dict_landmarks['out_lip'] = []
            dict_landmarks['out_lip'].append([int(x),int(y),(255,255,0)])
            if draw_circle:
                cv2.circle(img, (int(x),int(y)), 2, (255,255,0),-1)
        elif 95>= i >=88:
            if 'in_lip' not in dict_landmarks.keys():
                dict_landmarks['in_lip'] = []
            dict_landmarks['in_lip'].append([int(x),int(y),(50,220,255)])
            if draw_circle:
                cv2.circle(img, (int(x),int(y)), 2, (50,220,255),-1)
        else:
            if draw_circle:
                cv2.circle(img, (int(x),int(y)), 2, (255,0,255),-1)

    return dict_landmarks,face_pts

def draw_contour(image,dict,face_pts,r_bbox,flag_agu = False,img_size = 256,draw_flag = False):
    # if draw_flag:
    #     cv2.namedWindow('image_origin',0)
    #     cv2.imshow('image_origin',image)

    x0 = r_bbox[0]# 全图偏置
    y0 = r_bbox[1]

    # face alignment
    eye_lx = np.mean([dict['left_eye'][i][0]+x0 for i in range(len(dict['left_eye']))])
    eye_ly = np.mean([dict['left_eye'][i][1]+y0 for i in range(len(dict['left_eye']))])
    eye_rx = np.mean([dict['right_eye'][i][0]+x0 for i in range(len(dict['right_eye']))])
    eye_ry = np.mean([dict['right_eye'][i][1]+y0 for i in range(len(dict['right_eye']))])

    img,face_crop,dict_landmarks_align = get_face_aug_fun(image,face_pts,(eye_lx,eye_ly),(eye_rx,eye_ry),None,desiredFaceWidth=img_size,desiredLeftEye=(0.354, 0.395),flag_agu=False,draw_flag = draw_flag,\
    gt_offset_x = x0,gt_offset_y = y0)
    # if draw_flag:
    #     cv2.namedWindow('align',0)
    #     cv2.imshow('align',face_crop)

    face_ola_pts = []
    face_ola_pts.append(face_pts[33])
    face_ola_pts.append(face_pts[38])
    face_ola_pts.append(face_pts[50])
    face_ola_pts.append(face_pts[46])

    face_ola_pts.append(face_pts[60])
    face_ola_pts.append(face_pts[64])
    face_ola_pts.append(face_pts[68])
    face_ola_pts.append(face_pts[72])

    face_ola_pts.append(face_pts[51])
    face_ola_pts.append(face_pts[55])
    face_ola_pts.append(face_pts[59])

    face_ola_pts.append(face_pts[53])
    face_ola_pts.append(face_pts[57])


    #---------------------------------------------------------------------------
    for i in range(len(face_pts)):
        face_pts[i][0] = face_pts[i][0]+x0
        face_pts[i][1] = face_pts[i][1]+y0
        # cv2.circle(image, (int(face_pts[i][0]),int(face_pts[i][1])),6, (55,0,225),-1)

    for key in dict.keys():
        # print(key)
        _,_,color = dict[key][0]

        # if 'left_eye' == key:
        #     eye_x = np.mean([dict[key][i][0]+x0 for i in range(len(dict[key]))])
        #     eye_y = np.mean([dict[key][i][1]+y0 for i in range(len(dict[key]))])
        #     cv2.circle(image, (int(eye_x),int(eye_y)), 3, (255,255,55),-1)
        # if 'right_eye' == key:
        #     eye_x = np.mean([dict[key][i][0]+x0 for i in range(len(dict[key]))])
        #     eye_y = np.mean([dict[key][i][1]+y0 for i in range(len(dict[key]))])
        #     cv2.circle(image, (int(eye_x),int(eye_y)), 3, (255,215,25),-1)

        if 'basin' == key or 'wing_nose' == key:
            pts = np.array([[dict[key][i][0]+x0,dict[key][i][1]+y0] for i in range(len(dict[key]))],np.int32)
            if False:
                cv2.polylines(image,[pts],False,color,thickness = 1)

        else:
            points_array = np.zeros((1,len(dict[key]),2),dtype = np.int32)
            for i in range(len(dict[key])):
                x,y,_ = dict[key][i]
                points_array[0,i,0] = x+x0
                points_array[0,i,1] = y+y0

            # cv2.fillPoly(image, points_array, color)
            if False:
                cv2.drawContours(image,points_array,-1,color,thickness=1)
    # face_landmarks = np.array(face_landmarks)
    #                         face_landmarks = face_landmarks.reshape((66,2))

    pts_num = len(face_ola_pts)
    reprojectdst, euler_angle = get_head_pose(np.array(face_ola_pts).reshape((pts_num,2)),image,vis = False)
    pitch, yaw, roll = euler_angle
    # print('euler_angle : ',euler_angle)
    if draw_flag:
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(image, 'p '+str(int(pitch)), (10+x0,y0-15),font, 1.6, (255, 55,220), 4)
        cv2.putText(image, ' ,y '+str(int(yaw)), (90+x0,y0-15),font, 1.6, (255, 55, 220), 4)
        cv2.putText(image, ' ,r '+str(int(roll)), (170+x0,y0-15),font, 1.6, (255, 55, 220), 4)

        cv2.putText(image, 'p '+str(int(pitch)), (10+x0,y0-15),font, 1.6, (55, 255, 220), 2)
        cv2.putText(image, ' ,y '+str(int(yaw)), (90+x0,y0-15),font, 1.6, (55, 255, 220), 2)
        cv2.putText(image, ' ,r '+str(int(roll)), (170+x0,y0-15),font, 1.6, (75, 255, 220), 2)

    return face_crop,dict_landmarks_align

def refine_face_bbox(bbox,img_shape):
    height,width,_ = img_shape

    x1,y1,x2,y2 = bbox

    expand_w = (x2-x1)
    expand_h = (y2-y1)

    x1 -= expand_w*0.06
    y1 += expand_h*0.15
    x2 += expand_w*0.06
    y2 += expand_h*0.03

    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

    x1 = int(max(0,x1))
    y1 = int(max(0,y1))
    x2 = int(min(x2,width-1))
    y2 = int(min(y2,height-1))

    return (x1,y1,x2,y2)

def draw_eye_msg(eye_feature,str_show):
    imgg = np.zeros([600,600,3], dtype = np.uint8)
    imgg[:,:,:]=255
    pitch = (eye_feature[0][0])*180.
    yaw = (eye_feature[0][1])*180.

    cv2.putText(imgg, ' {} Pitch : {:.2f} '.format(str_show,pitch), (5,40),cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 0, 255),5)
    cv2.putText(imgg, ' {} Pitch : {:.2f} '.format(str_show,pitch), (5,40),cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 125, 255),2)

    cv2.putText(imgg, ' {} Yaw : {:.2f}'.format(str_show,yaw), (5,80),cv2.FONT_HERSHEY_DUPLEX, 1.5, (155, 50, 255),5)
    cv2.putText(imgg, ' {} Yaw : {:.2f}'.format(str_show,yaw), (5,80),cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 125, 0),2)

    print('  {}  -   pitch = {} , yaw = {}'.format(str_show,pitch,yaw))

    #-------------------------------------------------------------
    cv2.circle(imgg, (300,300), 3, (0,0,255),-1)
    cv2.circle(imgg, (300,300), 30, (255,0,255),1)

    cv2.ellipse(imgg,(300,300),(200,160),0,0,360,(255,0,0),2)
    #----------------------
    # pitch = 180

    # au_feature = np.array([float(pitch)/180.-0.5,float(yaw)/180.-0.5])
    # Pitch_move = ((180. - pitch)/180.-0.5)*160*2*2 +300
    # Yaw_move = ((1.-(180. - yaw)/180.)-0.5)*200*2*2 + 300

    Pitch_move = pitch/180.*160*2 +300
    Yaw_move = yaw/180.*200*2 + 300


    cv2.line(imgg, (300,300), (int(Yaw_move), int(Pitch_move)), (100,10,10), 3)



    cv2.circle(imgg, (int(Yaw_move), int(Pitch_move)), 90, (0,255,255),-1)
    cv2.circle(imgg, (int(Yaw_move), int(Pitch_move)), 68, (0,255,25),3)

    cv2.line(imgg, (300,300-160), (300,300+160), (255,155,155), 3)
    cv2.line(imgg, (300-200,300), (300+200,300), (55,155,255), 3)

    cv2.line(imgg, (int(Yaw_move),300-160), (int(Yaw_move),300+160), (0,255,0), 3)
    cv2.line(imgg, (300-200,int(Pitch_move)), (300+200,int(Pitch_move)), (0,55,255), 3)

    # cv2.namedWindow(str_show,0)
    # cv2.imshow(str_show,imgg)

    return imgg


def get_z_coord(Euler_Angle_r):
    z_coor= [0.,0.,1.]
    if Euler_Angle_r[0]>180.:
        Euler_Angle_r[0] = 360. - Euler_Angle_r[0]
    else:
        Euler_Angle_r[0] = -Euler_Angle_r[0]
    angles1 = [Euler_Angle_r[0],Euler_Angle_r[1],0.]
    theta = np.zeros((3, 1), dtype=np.float64)
    theta[0] = angles1[0]*np.pi/180.0
    theta[1] = angles1[1]*np.pi/180.0
    theta[2] = angles1[2]*np.pi/180.0
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))

    z_coor_ = np.dot(z_coor, R)

    return z_coor_
