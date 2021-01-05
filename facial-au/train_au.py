#-*-coding:utf-8-*-
# date:2020-04-24
# Author: x.l.eric

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import  sys
import yaml
yaml.warnings({'YAMLLoadWarning': False})

# from tensorboardX import SummaryWriter
from utils.model_utils import *
from utils.common_utils import *
from data_iter_au.datasets_au import *

from models.resnet_s import resnet18, resnet34, resnet50, resnet101, resnet152
from models.mobilenetv2 import MobileNetV2
from loss.loss import *
import cv2
import time
import json
from datetime import datetime
import random

def trainer(ops,f_log):
    if 1:
        person_list = define_sequence_datasets(ops.train_path)


        os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS

        if ops.log_flag:
            sys.stdout = f_log

        set_seed(ops.seed)
        #---------------------------------------------------------------- 构建模型
        print('use model : %s'%(ops.model))

        if ops.model == 'resnet_18':
            model_=resnet18(pretrained = ops.pretrained, num_classes=ops.num_classes, img_size=ops.img_size[0],dropout_factor=ops.dropout)
        elif ops.model == 'resnet_34':
            model_=resnet34(pretrained = ops.pretrained, num_classes=ops.num_classes, img_size=ops.img_size[0],dropout_factor=ops.dropout)
        elif ops.model == 'resnet_50':
            model_=resnet50(pretrained = ops.pretrained, num_classes=ops.num_classes, img_size=ops.img_size[0],dropout_factor=ops.dropout)
        elif ops.model == 'resnet_101':
            model_=resnet101(pretrained = ops.pretrained, num_classes=ops.num_classes, img_size=ops.img_size[0],dropout_factor=ops.dropout)
        elif ops.model == 'resnet_152':
            model_=resnet152(pretrained = ops.pretrained, num_classes=ops.num_classes, img_size=ops.img_size[0],dropout_factor=ops.dropout)
        elif ops.model == 'mobilenetv2':
            model_=MobileNetV2(n_class =ops.num_classes, input_size=ops.img_size[0],dropout_factor=ops.dropout)
        else:
            print('error no the struct model : {}'.format(ops.model))


        use_cuda = torch.cuda.is_available()

        device = torch.device("cuda:0" if use_cuda else "cpu")
        model_ = model_.to(device)

        # print(model_)# 打印模型结构
        # Dataset
        val_split = []
        dataset = LoadImagesAndLabels(ops= ops,img_size=ops.img_size,flag_agu=ops.flag_agu)
        print('len train datasets : %s'%(dataset.__len__()))
        # Dataloader
        dataloader = DataLoader(dataset,
                                batch_size=ops.batch_size,
                                num_workers=ops.num_workers,
                                shuffle=True,
                                pin_memory=False,
                                drop_last = True)
        # 优化器设计
        # optimizer_Adam = torch.optim.Adam(model_.parameters(), lr=ops.init_lr, betas=(0.9, 0.99),weight_decay=1e-6)
        optimizer_SGD = optim.SGD(model_.parameters(), lr=ops.init_lr, momentum=ops.momentum, weight_decay=ops.weight_decay)# 优化器初始化
        optimizer = optimizer_SGD
        # 加载 finetune 模型
        if os.access(ops.fintune_model,os.F_OK):# checkpoint
            chkpt = torch.load(ops.fintune_model, map_location=device)
            model_.load_state_dict(chkpt)
            print('load fintune model : {}'.format(ops.fintune_model))

        print('/**********************************************/')
        # 损失函数
        if ops.loss_define != 'wing_loss':
            criterion = nn.MSELoss(reduce=True, reduction='mean')

        step = 0
        idx = 0

        # 变量初始化
        best_loss = np.inf
        loss_mean = 0. # 损失均值
        loss_idx = 0. # 损失计算计数器
        flag_change_lr_cnt = 0 # 学习率更新计数器
        init_lr = ops.init_lr # 学习率

        epochs_loss_dict = {}

        for epoch in range(0, ops.epochs):
            if ops.log_flag:
                sys.stdout = f_log
            print('\nepoch %d ------>>>'%epoch)
            model_.train()
            # 学习率更新策略
            if loss_mean!=0.:
                if best_loss > (loss_mean/loss_idx):
                    flag_change_lr_cnt = 0
                    best_loss = (loss_mean/loss_idx)
                else:
                    flag_change_lr_cnt += 1

                    if flag_change_lr_cnt > 10:
                        init_lr = init_lr*ops.lr_decay
                        set_learning_rate(optimizer, init_lr)
                        flag_change_lr_cnt = 0

            loss_mean = 0. # 损失均值
            loss_idx = 0. # 损失计算计数器

            for i, (imgs_, pts_) in enumerate(dataloader):

                seq_imgs,seq_fe = sample_sequence_datasets(person_list)
                # print('imgs_, pts_',imgs_.size(), pts_.size())
                if use_cuda:
                    imgs_ = imgs_.cuda().float()  # pytorch 的 数据输入格式 ： (batch, channel, height, width)
                    pts_ = pts_.cuda().float()


                imgs_ = torch.cat([imgs_,seq_imgs],dim=0)
                pts_ = torch.cat([pts_,seq_fe],dim=0)

                # print('imgs_ , pts_ size :',imgs_.size(),pts_.size())

                output = model_(imgs_.float())
                if ops.loss_define == 'wing_loss':
                    loss = got_total_wing_loss(output, pts_)
                else:
                    loss_all = criterion(output, pts_)
                    loss_none_eye = criterion(output[:,2:24].float(),pts_[:,2:24].float())
                    loss_mouth_corner = criterion(output[:,9:13].float(),pts_[:,9:13].float())
                    loss_nose = criterion(output[:,21:24].float(),pts_[:,21:24].float())
                    loss = loss_all*0.4+loss_none_eye*0.6 + loss_mouth_corner*0.5 + loss_nose*0.5
                    # loss_eyebrow = torch.abs(output[:,17:27]-crop_landmarks[:,17:27].float())
                    # loss_nose = torch.abs(output[:,27:36]- crop_landmarks[:,27:36].float())
                    # loss_eye = torch.abs(output[:,36:48]- crop_landmarks[:,36:48].float())
                    # loss_mouse = torch.abs(output[:,48:66]- crop_landmarks[:,48:66].float())

                loss_mean += loss.item()
                loss_idx += 1.
                if i%10 == 0:
                    loc_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print('  %s - %s - epoch [%s/%s] (%s/%s):'%(loc_time,ops.model,epoch,ops.epochs,i,int(dataset.__len__()/ops.batch_size)),\
                    'loss : %.6f - %.6f'%(loss_mean/loss_idx,loss.item()),\
                    ' lr : %.5f'%init_lr,' bs :',ops.batch_size,\
                    ' img_size: %s x %s'%(ops.img_size[0],ops.img_size[1]),' best_loss: %.6f'%best_loss)
                    # time.sleep(10)
                if i%500 == 0:
                    torch.save(model_.state_dict(), ops.model_exp + 'model_epoch-{}.pth'.format(epoch))
                # 计算梯度
                loss.backward()
                # 优化器对模型参数更新
                optimizer.step()
                # 优化器梯度清零
                optimizer.zero_grad()
                step += 1

                # 一个 epoch 保存连词最新的 模型
                # if i%(int(dataset.__len__()/ops.batch_size/2-1)) == 0 and i > 0:
                #     torch.save(model_.state_dict(), ops.model_exp + 'latest.pth')
            # 每一个 epoch 进行模型保存
            torch.save(model_.state_dict(), ops.model_exp + 'model_epoch-{}.pth'.format(epoch))

    # except Exception as e:
    #     print('Exception : ',e) # 打印异常
    #     print('Exception  file : ', e.__traceback__.tb_frame.f_globals['__file__'])# 发生异常所在的文件
    #     print('Exception  line : ', e.__traceback__.tb_lineno)# 发生异常所在的行数

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=' Project Classification Train')
    parser.add_argument('--seed', type=int, default = 123,
        help = 'seed') # 设置随机种子
    parser.add_argument('--model_exp', type=str, default = './model_exp_au',
        help = 'model_exp') # 模型输出文件夹
    parser.add_argument('--model', type=str, default = 'resnet_34',
        help = 'model : resnet_18,resnet_34,resnet_50,resnet_101,resnet_152,mobilenetv2') # 模型类型
    parser.add_argument('--num_classes', type=int , default = 24,
        help = 'num_classes') #  au
    parser.add_argument('--GPUS', type=str, default = '0',
        help = 'GPUS') # GPU选择

    parser.add_argument('--train_path', type=str,
        default = 'H:/face/FEAFA/datasets/',
        help = 'train_path')# 训练集标注信息
    parser.add_argument('--test_path', type=str,
        default = './FEAFA/datasets/',
        help = 'test_path')# 测试集标注信息

    parser.add_argument('--val_factor', type=float, default = 0.0,
        help = 'val_factor') # 从训练集中分离验证集对应的比例
    parser.add_argument('--test_interval', type=int, default = 1,
        help = 'test_interval') # 训练集和测试集 计算 loss 间隔
    parser.add_argument('--pretrained', type=bool, default = True,
        help = 'imageNet_Pretrain') # 初始化学习率
    parser.add_argument('--fintune_model', type=str, default = './weights_au/model_au.pth',
        help = 'fintune_model') # fintune model
    parser.add_argument('--loss_define', type=str, default = 'wing_loss',#wing_loss
        help = 'define_loss： wing_loss, mse_loss') # 损失函数定义
    parser.add_argument('--init_lr', type=float, default = 1e-3,
        help = 'init_learningRate') # 初始化学习率
    parser.add_argument('--lr_decay', type=float, default = 0.9,
        help = 'learningRate_decay') # 学习率权重衰减率
    parser.add_argument('--weight_decay', type=float, default = 1e-5,
        help = 'weight_decay - default : 5e-4') # 优化器正则损失权重
    parser.add_argument('--momentum', type=float, default = 0.9,
        help = 'momentum') # 优化器动量
    parser.add_argument('--batch_size', type=int, default = 32,
        help = 'batch_size') # 训练每批次图像数量
    parser.add_argument('--dropout', type=float, default = 0.5,
        help = 'dropout') # dropout
    parser.add_argument('--epochs', type=int, default = 2000,
        help = 'epochs') # 训练周期
    parser.add_argument('--num_workers', type=int, default = 8,
        help = 'num_workers') # 训练数据生成器线程数
    parser.add_argument('--img_size', type=tuple , default = (256,256),
        help = 'img_size') # 输入模型图片尺寸
    parser.add_argument('--flag_agu', type=bool , default = True,
        help = 'data_augmentation') # 训练数据生成器是否进行数据扩增
    parser.add_argument('--fix_res', type=bool , default = False,
        help = 'fix_resolution') # 输入模型样本图片是否保证图像分辨率的长宽比
    parser.add_argument('--clear_model_exp', type=bool, default = False,
        help = 'clear_model_exp') # 模型输出文件夹是否进行清除
    parser.add_argument('--log_flag', type=bool, default = False,
        help = 'log flag') # 是否保存训练 log

    #--------------------------------------------------------------------------
    args = parser.parse_args()# 解析添加参数
    #--------------------------------------------------------------------------
    mkdir_(args.model_exp, flag_rm=args.clear_model_exp)
    loc_time = time.localtime()
    args.model_exp = args.model_exp + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", loc_time)+'/'
    mkdir_(args.model_exp, flag_rm=args.clear_model_exp)

    f_log = None
    if args.log_flag:
        f_log = open(args.model_exp+'/train_{}.log'.format(time.strftime("%Y-%m-%d_%H-%M-%S",loc_time)), 'a+')
        sys.stdout = f_log

    print('---------------------------------- log : {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", loc_time)))
    print('\n/******************* {} ******************/\n'.format(parser.description))

    unparsed = vars(args) # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print('{} : {}'.format(key,unparsed[key]))

    unparsed['time'] = time.strftime("%Y-%m-%d %H:%M:%S", loc_time)

    fs = open(args.model_exp+'train_ops.json',"w",encoding='utf-8')
    json.dump(unparsed,fs,ensure_ascii=False,indent = 1)
    fs.close()

    trainer(ops = args,f_log = f_log)# 模型训练

    if args.log_flag:
        sys.stdout = f_log
    print('well done : {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
