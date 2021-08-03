#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
import cv2
import time
import math
import random
import json
import _thread
import tensorflow as tf
import traceback
from log import ALL_LOG_OBJ
from skimage.measure import label, regionprops
from utils import list_images, resize_image, is_overlap, need_merge,save_crop_image
from concurrent.futures import ThreadPoolExecutor
# from inference import SemanticSegment
# semantic_segment_instance = SemanticSegment()



"""参数配置"""
config_path = './config.json'
config_data = json.loads(open(config_path).read(), encoding='utf-8')

# 分类数量
NUM_CLASSES = config_data['NUM_CLASSES']
# batch数量
BATCH_SIZE = config_data['BATCH_SIZE']

MIN_QR_WH_RATIO = config_data["MIN_QR_WH_RATIO"]
MAX_QR_WH_RATIO = config_data["MAX_QR_WH_RATIO"]
QR_AREA = config_data["QR_AREA"]
QR_RECT_H = config_data["QR_RECT_H"]



""""""
global product_type, product_id, resize_ratio, dilate_ratio


def create_batch_list(crop_image_list):
    batch_list = []
    temp = []
    for crop_id, crop_img in enumerate(crop_image_list):
        temp.append(crop_img)
        if len(temp) == BATCH_SIZE:
            batch_list.append(temp)
            temp = []
    if len(temp) != 0:
        batch_list.append(temp)
    return batch_list


def prediction(crop_image_list, label_name_list):
    # 把所有crop小图分组成batch进行存储
    batch_list = create_batch_list(crop_image_list)
    # 定义列表：存储每个类别的所有crop预测结果
    total_crop_prediction_list = []
    for label_id in range(len(label_name_list)):
        total_crop_prediction_list.append([])

    for batch_id, batch in enumerate(batch_list):  # 一张大图中所有的crop小图打包成batch，循环每个batch
        batch_result = semantic_segment_instance.predict_multi_label(batch, label_name_list)
        # 获取每张预测小图的不同类别的mask，存入各自cls的列表中
        for img_id in range(len(batch_result)):   # 循环一个batch里面的图片数量，比如batch=2，就循环2次，每次循环抓取所有的pred结果
            for label_id, label_name in enumerate(label_name_list):
                total_crop_prediction_list[label_id].append(batch_result[img_id][label_name])
    return total_crop_prediction_list


def get_defect_result(input_image, image_resize_ratio, crop_rect_list, prediction_defect_list, label_name_list):
    # 把每类缺陷的crop拼接成整图，然后存入list
    img_h, img_w = input_image.shape[0], input_image.shape[1]
    stitiching_img_list = []
    for label_id in range(len(prediction_defect_list)):
        temp_img = image_stitiching(prediction_defect_list[label_id], crop_rect_list, img_h, img_w)
        stitiching_img_list.append(temp_img)

    # 为了提速，先将缺陷膨胀，然后resize进行联通区域查找
    label_list = []
    defect_resize_ratio = 0.5
    recovery_ratio = 1 / defect_resize_ratio
    # se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for label_id in range(len(prediction_defect_list)):
        # temp_img = cv2.dilate(stitiching_img_list[label_id], se)
        # temp_img = cv2.resize(temp_img, (0, 0), fx=defect_resize_ratio, fy=defect_resize_ratio, interpolation=cv2.INTER_LINEAR)
        # label_list.append(temp_img)
        temp_img = cv2.resize(stitiching_img_list[label_id], (0, 0), fx=defect_resize_ratio, fy=defect_resize_ratio, interpolation=cv2.INTER_LINEAR)
        _, temp_img = cv2.threshold(temp_img, 5, 255, cv2.THRESH_BINARY)
        label_list.append(temp_img)

    # 计算连通区域
    cc_info = []
    for n, cc_label in enumerate(label_list):
        label_img = label(cc_label, neighbors=8, connectivity=2)
        all_connect_info = regionprops(label_img)

        for element in all_connect_info:
            if element.filled_area > 10:
                info = {}
                ly, lx, ry, rx = element.bbox
                info['box_left'] = int(lx * recovery_ratio)  # *recovery_ratio的目的是 在前面缩放了50%为了寻找联通区域提速
                info['box_top'] = int(ly * recovery_ratio)
                info['box_right'] = int(rx * recovery_ratio)
                info['box_bottom'] = int(ry * recovery_ratio)
                info['box_width'] = info['box_right'] - info['box_left']
                info['box_height'] = info['box_bottom'] - info['box_top']
                info['box_area'] = info['box_width'] * info['box_height']
                info['type'] = n + 3 + 1  # 缺陷类别  +3代表前面有frontier, corner, mark.  再+1 broken是第4个，chipping是第5个，与配置文件对应了
                info['merge_flag'] = False  # 是否被合并的标记，初始化为False
                cc_info.append(info)
    ALL_LOG_OBJ.logger.info('Defect rect before merge: %d' % len(cc_info))

    # 合并同类型并且有交叠的框
    for n in range(len(cc_info)):
        for k in range(len(cc_info)):
            if n != k and cc_info[n]['type'] == cc_info[k]['type'] \
                    and (cc_info[n]['merge_flag'] is False) \
                    and (cc_info[k]['merge_flag'] is False):
                bbox1 = (cc_info[n]['box_left'], cc_info[n]['box_top'],
                         cc_info[n]['box_right'] - cc_info[n]['box_left'],
                         cc_info[n]['box_bottom'] - cc_info[n]['box_top'])
                bbox2 = (cc_info[k]['box_left'], cc_info[k]['box_top'],
                         cc_info[k]['box_right'] - cc_info[k]['box_left'],
                         cc_info[k]['box_bottom'] - cc_info[k]['box_top'])

                if need_merge(bbox1, bbox2):
                    new_x = min(bbox1[0], bbox2[0])
                    new_y = min(bbox1[1], bbox2[1])
                    new_r = max(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
                    new_b = max(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
                    cc_info[n]['box_left'] = new_x
                    cc_info[n]['box_top'] = new_y
                    cc_info[n]['box_right'] = new_r
                    cc_info[n]['box_bottom'] = new_b
                    cc_info[k]['merge_flag'] = True

    if len(cc_info) > 0:
        cc_info = [cell for cell in cc_info if cell['merge_flag'] is False]
        ALL_LOG_OBJ.logger.info('Defect num after merge: %d' % len(cc_info))
    else:
        ALL_LOG_OBJ.logger.info('!!!! Attention, defect rect list is 0 !!!!')
    return cc_info


def get_qr_by_tradition_algo(input_image):
    qr_rect = [{'x': 0, 'y': 0, 'w': 100, 'h': 100}]

    # convert to gray image ...
    if input_image.ndim == 3:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # binary image ...
    _, binary_image = cv2.threshold(input_image, 200, 255, cv2.THRESH_BINARY)

    # find blob ...
    label_img = label(binary_image, neighbors=8, connectivity=2)
    all_connect_info = regionprops(label_img)
    for element in all_connect_info:
        ly, lx, ry, rx = element.bbox
        rect_x, rect_y = lx, ly
        rect_w, rect_h = rx - lx, ry - ly
        w_h_ratio = rect_w / rect_h
        rect_area = rect_w * rect_h

        if (w_h_ratio > MIN_QR_WH_RATIO) and (w_h_ratio < MAX_QR_WH_RATIO) and (rect_area > QR_AREA) and (rect_h > QR_RECT_H):
            qr_rect[0]['x'] = rect_x
            qr_rect[0]['y'] = rect_y
            qr_rect[0]['w'] = rect_w
            qr_rect[0]['h'] = rect_h
            break
    return qr_rect


def get_qr_by_deeplearning_algo(input_image):
    qr_rect = [{'x': 0, 'y': 0, 'w': 0, 'h': 0}]

    return 0


def qr_detection(input_image):
    ALL_LOG_OBJ.logger.info('**************************************************')
    ALL_LOG_OBJ.logger.info('********** QR code detect begining! ************')
    ALL_LOG_OBJ.logger.info('**************************************************')

    # 用传统算法获取QR的矩形坐标
    qr_rect = get_qr_by_tradition_algo(input_image)
    # qr_x, qr_y = qr_rect[0]['x'], qr_rect[0]['y']
    # qr_w, qr_h = qr_rect[0]['w'], qr_rect[0]['h']
    # cv2.imwrite("./test.png", input_image[qr_y:qr_y+qr_h, qr_x:qr_x+qr_w])

    # 用深度学习算法获取QR的矩形坐标




    ALL_LOG_OBJ.logger.info('**************************************************')
    ALL_LOG_OBJ.logger.info('********** QR code defect finished! ************')
    ALL_LOG_OBJ.logger.info('**************************************************')
    return qr_rect


if __name__ == '__main__':
    product_type = 'big'  # 'big' 'middle', or 'small'
    product_id = 0  # 0, 1 or 2
    resize_ratio = 0.5
    dilate_ratio = 1 / resize_ratio
    image_file_path = "D:/4_data/B10_cell_data/test_http/temp"
    file_list = os.listdir(image_file_path)
    for id1, image_name in enumerate(file_list):
        #  加载原始图像数据
        image_path = os.path.join(image_file_path, image_name)
        gray_resized_image = cv2.imread(image_path, -1)
        if gray_resized_image.ndim == 3:
            gray_resized_image = cv2.cvtColor(gray_resized_image, cv2.COLOR_BGR2GRAY)

        start = time.time()
        print(image_name)

        image_info_dict = {'image_name': image_name,  # str: xxxxx.bmp
                           'zm_or_fm': 'ZM',  # str: 'ZM' or 'FM'
                           'long_or_short': 'L',  # str: 'L' or 'S'
                           'left_or_right': 'L',  # str: 'L' or 'R'
                           'resize_ratio': 0.5,  # float: 0.5
                           'image_type': 'big',  # str: 'big' or 'middle' or 'small'
                           'image_id': 0}  # int: 0 or 1 or 2
        _ = qr_detection(gray_resized_image)
        end = time.time()
        print("the time is: ", ((end - start) * 1000))
        print("bingo...")
