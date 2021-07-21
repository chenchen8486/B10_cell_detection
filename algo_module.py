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
from utils import save_crop_image
from log import ALL_LOG_OBJ
from traditional_algo import *
from skimage.measure import label, regionprops
from utils import list_images, resize_image, is_overlap, need_merge
from concurrent.futures import ThreadPoolExecutor
from inference import SemanticSegment
semantic_segment_instance = SemanticSegment()
pool = ThreadPoolExecutor(max_workers=8)


"""参数配置"""
config_path = './config.json'
config_data = json.loads(open(config_path).read(), encoding='utf-8')
# 接收到的图像，在纵向裁切的尺寸（节约图像运算时间）
IMAGE_CUT_SIZE = config_data['IMAGE_CUT_SIZE']
# 分类数量
NUM_CLASSES = config_data['NUM_CLASSES']
# 水平和垂直crop的时候重叠像素数量
CROP_OVERLAP_X, CROP_OVERLAP_Y = config_data['CROP_OVERLAP_X'], config_data['CROP_OVERLAP_Y']
# batch数量
BATCH_SIZE = config_data['BATCH_SIZE']
# crop小图的尺寸
SUB_IMAGE_SIZE = config_data['SUB_IMAGE_SIZE']
# 类别标签名称的列表
label_name_list = config_data['LABEL_NAME_LIST']
# ["frontier", "mark", "corner",  "corner_dirty", "dirty_pt", "crack", "bubble"],
# 为了给真实缺陷尺寸扩大一点（便于最后存图或显示）
crop_corner_offset = config_data['CROP_CORNER_OFFSET']
DEFECT_NAME_LIST = config_data['DEFECT_NAME_LIST']
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


def get_defect_result(input_image, crop_rect_list, prediction_defect_list):
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
                info['type'] = n + 1  # 缺陷类别
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


def get_detection_result(input_image, direction_type, product_type, product_id, resize_ratio, rotate_angle,
                         inlier_pt_list, fit_vertical_line, prediction_result_list, crop_rect_list):
    dilate_ratio = 1 / resize_ratio

    # 1) frontier: 无论id是什么，磨边是共同需要检测的 ok
    # 返回结果：(1)在存储图像上绘制，绘制宽高单位是原始分辨率， 已经定义好了crop_offset了，并存按照offset制作了小图。
    #         (2)磨边宽度，现有图像宽度(非原始图像尺寸)，单位是像素
    try:
        frontier_crop_list = prediction_result_list[0]
        merge_frontier = image_stitiching(frontier_crop_list, crop_rect_list, input_image.shape[0], input_image.shape[1])
        frontier_measure_result_dict_ = get_frontier_result2(merge_frontier, input_image, resize_ratio,
                                                             inlier_pt_list, fit_vertical_line, rotate_angle)
    except:
        ALL_LOG_OBJ.logger.info('!!!! Frontier measure failed !!!!')
        frontier_measure_result_dict_ = {'frontier_width1': 0, 'crop_img1': None,
                                         'frontier_width2': 0, 'crop_img2': None,
                                         'frontier_width3': 0, 'crop_img3': None,
                                         'frontier_average': 0, 'crop_rect1': [], 'crop_rect2': [], 'crop_rect3': []}

    # 2) mark: 根据不同的图像左右方向，图像id，进行mark测距 ok
    # 返回结果：(1)在存储图像上绘制，绘制宽高单位是原始分辨率， 已经定义好了crop_offset了，并存按照offset制作了小图。
    #         (2)mark宽高，现有图像宽高(非原始图像尺寸)，单位是像素
    try:
        mark_crop_list = prediction_result_list[1]
        mark_measure_result_dict_ = get_mark_result(input_image, mark_crop_list, crop_rect_list, direction_type,
                                                    product_type, product_id, resize_ratio, dilate_ratio, rotate_angle,
                                                    fit_vertical_line)
    except:
        ALL_LOG_OBJ.logger.info('!!!! Mark measure failed !!!!')
        mark_measure_result_dict_ = {'upper_mark': [0, 0], 'below_mark': [0, 0], 'upper_crop_img': None,
                                     'below_crop_img': None, 'upper_rect': [], 'below_rect': []}

    # # 3) corner: 根据不同的图像左右方向，图像id，进行corner测距
    # 返回结果：(1)在存储图像上绘制，绘制宽高单位是原始分辨率， 已经定义好了crop_offset了，并存按照offset制作了小图。
    #         (2)corner宽度，现有图像宽度(非原始图像尺寸)，单位是像素
    try:
        corner_measure_result_dict_ = get_corner_tradition(input_image, direction_type, product_id, product_type, dilate_ratio)
    except:
        ALL_LOG_OBJ.logger.info('!!!! Corner measure failed !!!!')
        corner_measure_result_dict_ = {'upper_corner': [0, 0], 'below_corner': [0, 0], 'upper_crop_img': None,
                                       'below_crop_img': None, 'upper_rect': [], 'below_rect': []}

    # 4) defect: 无论id是什么，缺陷是共同需要检测的
    # 返回结果：(1)defect所有信息，现有图像尺寸(非原始图像尺寸)，单位是像素。 图像绘制与包存在http中进行。
    try:
        prediction_defect_list = []
        for label_id in range(3, len(prediction_result_list)):
            prediction_defect_list.append(prediction_result_list[label_id])
        defect_result_list_ = get_defect_result(input_image, crop_rect_list, prediction_defect_list)
    except:
        ALL_LOG_OBJ.logger.info('!!!! Defect detection failed !!!!')
        defect_result_list_ = []
    return mark_measure_result_dict_, corner_measure_result_dict_, frontier_measure_result_dict_, defect_result_list_


def cut_image(input_image, direction_type):
    if direction_type == 'left':
        cut_input_image = input_image[:, IMAGE_CUT_SIZE:]
    else:
        cut_input_image = input_image[:, :input_image.shape[1]-IMAGE_CUT_SIZE]
    return cut_input_image


def data_arrange(dilate_ratio, direction_type, mark_mesure_result_dict, corner_mesure_result_dict,
                 frontier_mesure_result_dict, defect_result_list):

    if direction_type == 'left':
        # 针对mark 测距的x坐标进行偏移补偿
        if mark_mesure_result_dict['upper_mark'][0] != 0 and mark_mesure_result_dict['upper_mark'][1] != 0:
            mark_mesure_result_dict['upper_mark'][0] = mark_mesure_result_dict['upper_mark'][0] + IMAGE_CUT_SIZE
        if mark_mesure_result_dict['below_mark'][0] != 0 and mark_mesure_result_dict['below_mark'][1] != 0:
            mark_mesure_result_dict['below_mark'][0] = mark_mesure_result_dict['below_mark'][0] + IMAGE_CUT_SIZE
        if len(mark_mesure_result_dict['upper_rect']) != 0:
            mark_mesure_result_dict['upper_rect'][0] = mark_mesure_result_dict['upper_rect'][0] + IMAGE_CUT_SIZE
        if len(mark_mesure_result_dict['below_rect']) != 0:
            mark_mesure_result_dict['below_rect'][0] = mark_mesure_result_dict['below_rect'][0] + IMAGE_CUT_SIZE

        # 对corner水平和垂直测量进行x坐标偏移补偿
        if corner_mesure_result_dict['upper_corner'][0] != 0 and corner_mesure_result_dict['upper_corner'][1] != 0:
            corner_mesure_result_dict['upper_corner'][0] = corner_mesure_result_dict['upper_corner'][0] + IMAGE_CUT_SIZE
        if corner_mesure_result_dict['below_corner'][0] != 0 and corner_mesure_result_dict['below_corner'][1] != 0:
            corner_mesure_result_dict['below_corner'][0] = corner_mesure_result_dict['below_corner'][0] + IMAGE_CUT_SIZE
        if len(corner_mesure_result_dict['upper_rect']) != 0:
            corner_mesure_result_dict['upper_rect'][0] = corner_mesure_result_dict['upper_rect'][0] + IMAGE_CUT_SIZE
        if len(corner_mesure_result_dict['below_rect']) != 0:
            corner_mesure_result_dict['below_rect'][0] = corner_mesure_result_dict['below_rect'][0] + IMAGE_CUT_SIZE

        # 对fonrtier截取到的rect进行x坐标偏移补偿
        if len(frontier_mesure_result_dict['crop_rect1']) != 0:
            frontier_mesure_result_dict['crop_rect1'][0] = frontier_mesure_result_dict['crop_rect1'][0] + IMAGE_CUT_SIZE
        if len(frontier_mesure_result_dict['crop_rect2']) != 0:
            frontier_mesure_result_dict['crop_rect2'][0] = frontier_mesure_result_dict['crop_rect2'][0] + IMAGE_CUT_SIZE
        if len(frontier_mesure_result_dict['crop_rect3']) != 0:
            frontier_mesure_result_dict['crop_rect3'][0] = frontier_mesure_result_dict['crop_rect3'][0] + IMAGE_CUT_SIZE

    # 对缺陷x坐标进行偏移补偿
    if len(defect_result_list) != 0:
        for id, defect in enumerate(defect_result_list):
            if direction_type == 'left':
                # x偏移补偿
                defect['box_left'] = defect['box_left'] + IMAGE_CUT_SIZE
                defect['box_right'] = defect['box_right'] + IMAGE_CUT_SIZE
    return mark_mesure_result_dict, corner_mesure_result_dict, defect_result_list


def test_draw_result(cut_input_image, mark_mesure_result_dict, corner_mesure_result_dict, frontier_mesure_result_dict,
                     defect_result_list):

    temp = cv2.cvtColor(cut_input_image, cv2.COLOR_GRAY2BGR)
    # 存储mark #####
    if mark_mesure_result_dict['upper_crop_img'] is not None:
        cv2.imwrite('./result/upper_mark.png', mark_mesure_result_dict['upper_crop_img'])
    if mark_mesure_result_dict['below_crop_img'] is not None:
        cv2.imwrite('./result/below_mark.png', mark_mesure_result_dict['below_crop_img'])

    # 绘制corner #####
    if corner_mesure_result_dict['upper_crop_img'] is not None:
        cv2.imwrite('./result/upper_corner.png', corner_mesure_result_dict['upper_crop_img'])
    if corner_mesure_result_dict['below_crop_img'] is not None:
        cv2.imwrite('./result/below_corner.png', corner_mesure_result_dict['below_crop_img'])

    # 绘制frontier #####
    if frontier_mesure_result_dict['crop_img1'] is not None:
        cv2.imwrite('./result/frontier1.png', frontier_mesure_result_dict['crop_img1'])
    if frontier_mesure_result_dict['crop_img2'] is not None:
        cv2.imwrite('./result/frontier2.png', frontier_mesure_result_dict['crop_img2'])
    if frontier_mesure_result_dict['crop_img3'] is not None:
        cv2.imwrite('./result/frontier3.png', frontier_mesure_result_dict['crop_img3'])

    # 绘制缺陷 #####
    if len(defect_result_list) != 0:
        for cc in defect_result_list:
            cv2.rectangle(temp, (cc['box_left'], cc['box_top']), (cc['box_right'], cc['box_bottom']), (0, 0, 255), 1)
        cv2.imwrite('./result/defect.png', temp)


def draw_all_defect(input_image, defect_result_list, product_type, product_id, direction_type):
    # 缩放图像便于后续的计算与显示
    ui_resize = 0.25
    input_resize_gray = cv2.resize(input_image, (0, 0), fx=ui_resize, fy=ui_resize, interpolation=cv2.INTER_LINEAR)
    input_resize_bgr = cv2.cvtColor(input_resize_gray, cv2.COLOR_GRAY2BGR)

    # 绘制缺陷 #####
    ui_dilate = 1 / ui_resize
    color_list = [(0, 0, 255),  (255, 0, 0)]
    if len(defect_result_list) != 0:
        for cc in defect_result_list:
            read_defect_rect = input_image[cc['box_top']:cc['box_bottom'], cc['box_left']:cc['box_right']]
            defect_std_val = np.std(read_defect_rect, ddof=1)
            if defect_std_val < 5:  # 如果缺陷区域标准差很小， 证明是误报
                continue

            rect_left, rect_top = int(cc['box_left'] * ui_resize), int(cc['box_top'] * ui_resize)
            rect_right, rect_bottom = int(cc['box_right'] * ui_resize), int(cc['box_bottom'] * ui_resize)
            cv2.rectangle(input_resize_bgr, (rect_left, rect_top), (rect_right, rect_bottom), color_list[cc['type']-1], 1)
            text_pos_x = rect_left
            text_pos_y = rect_top - 4
            cv2.putText(input_resize_bgr, DEFECT_NAME_LIST[cc['type']-1], (text_pos_x, text_pos_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_list[cc['type']-1], 1)

    # 对图像进行裁剪
    canny_bw_image = contour_detect(input_resize_bgr)
    cut_upper, cut_below, cut_left, cut_right = 0, 0, 0, 0
    cut_offset = 50
    # 1） 判断上下裁切位置
    if product_id == 0 and (product_type == 'big' or product_type == 'middle'):  # 上图
        cut_upper = calcul_hori_pos(canny_bw_image, 3, 0)
    elif (product_id == 1 and product_type == 'middle') or product_id == 2:  # 下图
        cut_below = calcul_hori_pos(canny_bw_image, 3, product_id)
    elif product_id == 0 and product_type == 'small':  # 上图 & 下图
        # 垂直投影 裁切行的位置
        cut_upper = calcul_hori_pos(canny_bw_image, 3, 0)
        cut_below = calcul_hori_pos(canny_bw_image, 3, 2)
    # 2) 开始上下裁切
    if cut_upper != 0:
        cut_upper = cut_upper - cut_offset
        if cut_upper < 0:
            cut_upper = cut_upper + cut_offset  # 如果有负数，就恢复回来
        input_resize_bgr = input_resize_bgr[cut_upper:, :, :]
    if cut_below != 0:
        cut_below = cut_below + cut_offset
        if cut_below > input_resize_bgr.shape[0]:
            cut_below = cut_below - cut_offset
        input_resize_bgr = input_resize_bgr[:cut_below, :]

    # 3）判断左右裁切位置
    if direction_type == 'left':
        cut_left = calcul_vert_pos(canny_bw_image, 3, direction_type)
        cut_left = cut_left - cut_offset
        if cut_left < 0:
            cut_left = cut_left + cut_offset
        input_resize_bgr = input_resize_bgr[:, cut_left:, :]
    else:
        cut_right = calcul_vert_pos(canny_bw_image, 3, direction_type)
        cut_right = cut_right + cut_offset
        if cut_right > input_resize_bgr.shape[1]:
            cut_right = cut_right - cut_offset
        input_resize_bgr = input_resize_bgr[:, :cut_right, :]

    return input_resize_bgr


def defect_detection(input_image, _product_type, _product_id, _resize_ratio):
    product_type = _product_type
    product_id = int(_product_id)
    resize_ratio = _resize_ratio
    dilate_ratio = 1 / _resize_ratio

    ALL_LOG_OBJ.logger.info('**************************************************')
    ALL_LOG_OBJ.logger.info('********** Semantic segment begining! ************')
    ALL_LOG_OBJ.logger.info('**************************************************')
    ALL_LOG_OBJ.logger.info('Receive Info' + 'Product typ: ' + str(product_type) + '   Product_id: ' + str(product_id))

    # 1) 图像左右相机分类
    direction_type = direction_classify(input_image)
    ALL_LOG_OBJ.logger.info('Camera classify finished and result:  ' + str(direction_type))

    # 2) 原始图像进行纵向裁切
    cut_input_image = cut_image(input_image, direction_type)
    ALL_LOG_OBJ.logger.info('Cut input image finished and width is: ' + str(cut_input_image.shape[1]))

    # 3) 对原始图像进行角度估计，如果存在旋转，估计出旋转矩阵，便于后续在旋转后的图像上测量编剧
    fit_vertical_line, rotate_angle, vertical_inlier_pt_list = calcul_rotate_angle(cut_input_image, direction_type,
                                                                                     "fit_vertical_line")
    ALL_LOG_OBJ.logger.info('Calculate rotation finished and angle is: ' + str(rotate_angle))

    # 4) 对原始图像进行切分，切分成768*768的图片
    crop_rect_list, crop_image_list = crop_overlap_image(cut_input_image, direction_type, SUB_IMAGE_SIZE,
                                                         SUB_IMAGE_SIZE, CROP_OVERLAP_X, CROP_OVERLAP_Y)
    ALL_LOG_OBJ.logger.info('Crop rect finished and rect number is: ' + str(len(crop_rect_list)))

    # 5) Inference给出预测结果并存储为缺陷坐标信息作为输出，返回每个类别的所有大图crop的list
    total_crop_prediction_list = prediction(crop_image_list, label_name_list)
    ALL_LOG_OBJ.logger.info('Prediction finished...')

    # 6) 根据mask获取最终预测结果
    mark_mesure_result_dict, corner_mesure_result_dict, frontier_mesure_result_dict, defect_result_list = \
        get_detection_result(cut_input_image, direction_type, product_type, product_id, resize_ratio, rotate_angle,
                             vertical_inlier_pt_list, fit_vertical_line, total_crop_prediction_list, crop_rect_list)

    # 7) 数据整理（将CUT_SIZE进行补偿）
    data_arrange(dilate_ratio, direction_type, mark_mesure_result_dict, corner_mesure_result_dict,
                 frontier_mesure_result_dict, defect_result_list)

    # 8) draw all defects in whole sheet
    draw_defect_panel = draw_all_defect(input_image, defect_result_list, product_type, product_id, direction_type)

    ALL_LOG_OBJ.logger.info('**************************************************')
    ALL_LOG_OBJ.logger.info('********** Semantic segment finished! ************')
    ALL_LOG_OBJ.logger.info('**************************************************')
    return direction_type, mark_mesure_result_dict, corner_mesure_result_dict, frontier_mesure_result_dict, defect_result_list, draw_defect_panel


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
        _, mark_mesure_result_dict, corner_mesure_result_dict, frontier_mesure_result_dict, defect_result_list = \
            defect_detection(gray_resized_image, product_type, product_id, resize_ratio)
        end = time.time()
        print("the time is: ", ((end - start) * 1000))
        print("bingo...")
