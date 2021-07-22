#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
import cv2
import time
import math
import random
import json
import traceback
from utils import *
from skimage.measure import label, regionprops
# global product_type, product_id, resize_ratio, dilate_ratio
from log import ALL_LOG_OBJ

"""参数配置"""
config_path = './config.json'
config_data = json.loads(open(config_path).read(), encoding='utf-8')
COL_STEP, ROW_STEP = config_data['COL_STEP'], config_data['ROW_STEP']  # 水平垂直投影时的步长
FIT_PT_NUM = config_data['FIT_PT_NUM']  # 设定一个最大收集投影点的数量，只要收集到这个数量，就可以停止遍历
L1_THRESH = config_data['L1_THRESH']  # 遍历的时候必须通过这个阈值来判断当前行和上一行是否存在超差，从而判断当前行是否是投影点

# 磨边算法参数
offset_x, offset_y = config_data['OFFSET_X'], config_data['OFFSET_Y']
high_variance = config_data['HIGH_VARIANCE']  # 从边缘开始到磨边的差值如果超过这个阈值就开始记录坐标位置了
low_variance = config_data['LOW_VARIANCE']  # 从边磨的另一测到均匀边缘的阈值
high_bright_th = config_data['HIGH_BRIGHT_THRESH']  # 有些磨边左右两侧会出现很白的像素点，如果超过这个阈值，就需要把这些点填充
fill_high_bright_val = config_data['FILL_HIGH_BRIGHT_VAL']  # 对找到的高亮值进行填充
frontier_text_region_height = config_data['FRONTIER_TEXT_REGION_HEIGHT']

# mark算法参数
mark_size_th = config_data['MARK_SIZE_THRESH']
crop_mark_offset = config_data['CROP_MARK_OFFSET']
mark_text_region_height = config_data['MARK_TEXT_REGION_HEIGHT']

# corner算法参数
corner_size_th = config_data['CORNER_SIZE_THRESH']
crop_corner_offset = config_data['CROP_CORNER_OFFSET']
corner_text_region_height = config_data['CORNER_TEXT_REGION_HEIGHT']

# 相机分辨率
camera_resolution = config_data['CAMERA_RESOLUTION']

def get_rotation_matrix(rotate_center_pt, rotate_angle, scale=1.0):
    # 获取旋转矩阵
    rotate_matrix = cv2.getRotationMatrix2D(rotate_center_pt, rotate_angle, scale)
    return rotate_matrix


def get_rotate_pt(rotate_matrix, pt):
    # 获取mark点旋转后的坐标 map_x, map_y
    map_x = float(pt[0]) * rotate_matrix[0][0] + float(pt[1]) * rotate_matrix[0][1] + rotate_matrix[0][2]
    map_y = float(pt[0]) * rotate_matrix[1][0] + float(pt[1]) * rotate_matrix[1][1] + rotate_matrix[1][2]
    mapped_pt = (int(map_x), int(map_y))
    return mapped_pt


def calcul_rotate_matrix(rotate_angle, image_width, image_height):
    # 计算旋转矩阵
    center_pt = (image_width // 2, image_height // 2)
    rotate_matrix = get_rotation_matrix(center_pt, rotate_angle)
    return rotate_matrix


def direction_classify(input_image):
    resize_image = cv2.resize(input_image, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
    first_col = resize_image[:, 0]
    last_col = resize_image[:, -1]
    first_col_std = np.std(first_col, ddof=1)
    last_col_std = np.std(last_col, ddof=1)

    if first_col_std <= last_col_std:
        direction_type = 'left'
    else:
        direction_type = 'right'
    return direction_type


def calcul_ange_between_2lines(line_a, line_b, fit_line_type="fit_horizontal_line"):
    """
    功能描述：根据线段a,b计算出他们的夹角（线段a是拟合直线， 线段b是标准水平直线）。
    注意：如果图片顺时针旋转，返回的角度就是负数
    :param line_a: 线段a，其中包含了线段a的点的列表，列表中每个点用(x, y)描述（注：列表中只存储了起点和端点）。
    :param line_b: 线段b，其中包含了线段a的点的列表，列表中每个点用(x, y)描述（注：列表中只存储了起点和端点）。
    :return: cos_theta是返回的两个线段的夹角。
    """
    if fit_line_type == "fit_horizontal_line":
        line_a_x = [p[0] for p in line_a]  # 提取x
        line_a_y = [p[1] for p in line_a]  # 提取y
        line_b_x = [p[0] for p in line_b]  # 提取x
        line_b_y = [p[1] for p in line_b]  # 提取y
    else:
        line_a_y = [p[0] for p in line_a]  # 提取x
        line_a_x = [p[1] for p in line_a]  # 提取y
        line_b_y = [p[0] for p in line_b]  # 提取x
        line_b_x = [p[1] for p in line_b]  # 提取y

    fit_a = np.polyfit(line_a_x, line_a_y, 1)       # 用一次多项式x=a*y+b拟合这些点，fit是(a,b)
    k1, b1 = fit_a[0], fit_a[1]
    fit_b = np.polyfit(line_b_x, line_b_y, 1)       # 用一次多项式x=a*y+b拟合这些点，fit是(a,b)
    k2, b2 = fit_b[0], fit_b[1]

    # 计算两条直线的夹角
    cos_theta = int(math.fabs(np.arctan((k1 - k2) / (float(1 + k1 * k2))) * 180 / np.pi) + 0.5)

    # 如果拟合线在标准水平线之上，角度为负， 否则角度为正
    cos_theta = -cos_theta if line_a_y[-1] >= line_b_y[-1] else cos_theta
    return cos_theta


def fit_line_by_ransac(point_list, sigma=7, iters=200, P=0.99):
    """
    功能描述：通过给定的原始点列表，通过RANSAC筛选出内点。
    :param point_list: 原始点列表。
    :param sigma: 数据和模型之家你可接受的差值。
    :param iters: 最大迭代次数。
    :param P: 希望得到正确模型的概率。
    :return: inlier_pt_list: 筛选过后的内点列表。
             best_a: 最优秀模型的斜率。
             best_b: 最优秀模型的截距。
    """
    # 最好模型的参数估计
    best_a, best_b = 0, 0  # 直线斜率,截距
    n_total = 0  # 内点数目
    inlier_pt_list = []
    for i in range(iters):
        # 随机选两个点去求解模型
        sample_index = random.sample(range(len(point_list)), 2)
        x_1 = point_list[sample_index[0]][0]
        y_1 = point_list[sample_index[0]][1]
        x_2 = point_list[sample_index[1]][0]
        y_2 = point_list[sample_index[1]][1]

        if x_2 == x_1:
            continue

        # y = ax + b 求解出a，b
        a = (y_2 - y_1) / (x_2 - x_1)
        b = y_1 - a * x_1

        # 算出内点数目
        total_inlier = 0
        inlier_pt_list = []
        for index in range(len(point_list)):
            y_estimate = a * point_list[index][0] + b
            if abs(y_estimate - point_list[index][1]) < sigma:
                total_inlier += 1
                inlier_pt_list.append([point_list[index][0], point_list[index][1]])

        # 判断当前的模型是否比之前估算的模型好
        if total_inlier > n_total:
            iters = math.log(1 - P) / math.log(1 - pow(total_inlier / len(point_list), 2) + 0.00001)
            n_total = total_inlier
            best_a = a
            best_b = b

        # 判断是否当前模型已经符合超过一半的点
        if total_inlier > len(point_list) // 2:
            break
    return inlier_pt_list, best_a, best_b


def vertical_projection(input_gray_image, direction_type, row_step=3,  fit_pt_num=15, l1_threshold=10):
    """
    功能描述：通过对图像垂直扫描，遍历每列像素，最终找到芯片的水平边缘位置的像素点（便于后续芯片的直线拟合）。
    :param input_gray_image: 输入一张灰度图像。
    :param direction_type: 获取camera type('left' or 'right')
    :param row_step: 遍历行的时候按照这个步长进行遍历，节约遍历时间。
    :param col_step: 遍历列的时候按照这个步长进行遍历，节约遍历时间（本质上不需要过的边缘点）。
    :param fit_pt_num: 设定一个最大收集投影点的数量，只要收集到这个数量，就可以停止遍历，节约效率。
    :param l1_threshold: 在当前列，行遍历的时候必须通过这个阈值来判断当前行和上一行是否存在超差，从而判断当前行是否是投影点。
    :return: inlier_pt_list: 返回了所有投影点的内点（内点通过RANSAC选择） inlier_pt_list = [x, y]。
             vertical_projection_pt_list: 返回了最原始的所有投影点。
    """
    height, width = input_gray_image.shape[0], input_gray_image.shape[1]
    col_step = width // fit_pt_num
    # 判断是左相机还是右相机
    if direction_type == 'left':
        col_start = width - 1
        col_end = 0 + col_step      # 防止越界
        col_step = -col_step
    else:
        col_start = 0
        col_end = width - col_step  # 防止越界

    # 垂直投影，获取投影到产品水平边缘的x,y坐标以及相对位置
    vertical_projection_pt_list = []
    for col_id in range(col_start, col_end, col_step):
        if len(vertical_projection_pt_list) >= fit_pt_num:
            break
        for row_id in range(row_step, height-row_step, row_step):
            previous_val = input_gray_image[row_id-row_step, col_id]
            current_val = input_gray_image[row_id, col_id]
            abs_l1 = abs(np.int(current_val) - np.int(previous_val))
            if abs_l1 > l1_threshold:
                vertical_projection_pt_list.append([col_id, row_id])
                break
    inlier_pt_list, best_k, best_b = fit_line_by_ransac(vertical_projection_pt_list, 3)
    return inlier_pt_list, vertical_projection_pt_list


def horizontal_projection(input_gray_image, direction_type, col_step=3, fit_pt_num=15, l1_threshold=10):
    '''
    功能：根据输入的灰度图像进行水平投影，返回水平投影的原始点和滤波后的准确内点
    参数：
    input_gray_image: 输入的灰度图像
    direction_type: 相机方向，是左侧相机拍摄还是右侧相机拍摄('left', 'right')，左侧相机拍摄背景从左侧开始，反之一样。
    col_step: 为了减少便利次数，优化速度，col_step描述每一行遍历时，间隔多少点计算一次灰度值
    fit_pt_num: 描述了需要水平投影的次数，即：间隔多少行，进行一次水平投影
    l1_threshold: 当前像素与上一个像素的灰度阈值
    返回：
    inlier_pt_list: 精确过滤后的投影点坐标列表[(y1, x1), (y2, x2), ...]
    horizontal_projection_list: 原始投影点坐标列表[(y1, x1), (y2, x2), ...]
    '''
    height, width = input_gray_image.shape[0], input_gray_image.shape[1]
    row_step = height // fit_pt_num
    if direction_type == 'left':
        col_start = 0
        col_end = width - col_step  # 防止越界
    else:
        col_start = width - 1 - col_step
        col_end = 0 + col_step      # 防止越界
        col_step = -col_step
    # 水平投影，获取投影的xy坐标以及相对位置
    horizontal_projection_list = []
    for row_id in range(0, height-row_step, row_step):
        # 为了节约在行上的遍历耗时，只要收集了fit_pt_num个点就可以直线拟合了
        if len(horizontal_projection_list) >= fit_pt_num:
            break
        for col_id in range(col_start, col_end, col_step):
            previous_val = input_gray_image[row_id, col_id+col_step]
            current_val = input_gray_image[row_id, col_id]
            abs_l1 = abs(np.int(current_val) - np.int(previous_val))
            if abs_l1 > l1_threshold:
                horizontal_projection_list.append([row_id, col_id])
                break
    inlier_pt_list, best_k, best_b = fit_line_by_ransac(horizontal_projection_list, 3)
    horizontal_projection_list = [(i[1], i[0]) for i in horizontal_projection_list]
    inlier_pt_list = [(i[1], i[0]) for i in inlier_pt_list]
    return inlier_pt_list, horizontal_projection_list


def calcul_rotate_angle(input_image, direction_type, fit_line_type):
    # 水平投影, 获取垂直边缘线段的拟合结果
    horizontal_inlier_pt_list, aa = horizontal_projection(input_image, direction_type, COL_STEP, FIT_PT_NUM, L1_THRESH)

    # 计算拟合直线和垂直直线端点
    fit_line_start_x, fit_line_start_y = horizontal_inlier_pt_list[0][0], horizontal_inlier_pt_list[0][1]
    fit_line_end_x, fit_line_end_y = horizontal_inlier_pt_list[-1][0], horizontal_inlier_pt_list[-1][1]
    straight_line_start_x, straight_line_start_y = horizontal_inlier_pt_list[0][0], horizontal_inlier_pt_list[0][1]
    straight_line_end_x, straight_line_end_y = horizontal_inlier_pt_list[0][0], horizontal_inlier_pt_list[-1][1]

    # 计算垂直边缘与线的夹角
    fit_line = [(fit_line_start_x, fit_line_start_y), (fit_line_end_x, fit_line_end_y)]
    straight_line = [(straight_line_start_x, straight_line_start_y), (straight_line_end_x, straight_line_end_y)]
    intersection_angle = calcul_ange_between_2lines(fit_line, straight_line, fit_line_type)
    return fit_line, intersection_angle, horizontal_inlier_pt_list


def get_frontier_result2(merge_frontier, input_image, resize_ratio, inlier_pt_list, fit_vertical_line, rotate_angle):
    frontier_measure_result_dict = {'frontier_width1': 0, 'crop_img1': None,
                                    'frontier_width2': 0, 'crop_img2': None,
                                    'frontier_width3': 0, 'crop_img3': None,
                                    'frontier_average': 0,  'crop_rect1': [], 'crop_rect2': [], 'crop_rect3': []}

    img_h, img_w = input_image.shape[0], input_image.shape[1]
    dilate_ratio = 1 / resize_ratio
    result_img_list, crop_rect_list, result_w_list = [], [], []
    for pt_id, pt in enumerate(inlier_pt_list):
        # 定义统一尺寸的crop roi，便于未来存储
        roi_start_x = pt[0] - offset_x
        if roi_start_x < 0:
            roi_start_x = 0

        roi_start_y = pt[1] - offset_y
        if roi_start_y < 0:
            roi_start_y = 0

        roi_end_x = roi_start_x + 2 * offset_x
        if roi_start_x == 0:
            roi_end_x = roi_start_x + 2 * offset_x
        elif roi_end_x > img_w:
            roi_end_x = img_w
            roi_start_x = img_w - 2 * offset_x

        roi_end_y = roi_start_y + 2 * offset_y
        if roi_start_y == 0:
            roi_end_y = roi_start_y + 2 * offset_y
        if roi_end_y > img_h:
            roi_end_y = img_h
            roi_start_y = img_h - 2 * offset_y

        crop_roi = merge_frontier[roi_start_y:roi_end_y, roi_start_x:roi_end_x]
        frontier_pos_list = []
        frontier_white_num_list = []
        # 循环当前roi中的每一行，计算白色区域的宽度
        for row_id in range(crop_roi.shape[0]):
            white_pos = np.vstack(np.where(crop_roi[row_id, :] == 255))  # 计算每行的白像素数量
            if white_pos.size == 0:  # 证明当前行没有找到任何白像素
                frontier_white_num_list.append(0)
                frontier_pos_list.append([0, 0])
                continue
            frontier_white_num_list.append(white_pos[0].shape[0])        # 存储当前行白像素数量
            start_pos_x, end_pos_x = white_pos[0][0], white_pos[0][-1]   # 获取当前行白像素连通的起始和终止点
            frontier_pos_list.append([start_pos_x, end_pos_x])           # 存储当前行白像素连通的起始和终止点

        if len(frontier_pos_list) == 0 or len(frontier_white_num_list) == 0:
            continue

        # 对每一行白像素数量的列表，进行升序排列，和排序后对应的id， sort_num_list = [(id, width_val), (id, width_val),....]
        sort_num_list = sorted(enumerate(frontier_white_num_list), key=lambda x: x[1], reverse=False)
        # 获取中位
        median_pos = len(sort_num_list) // 2
        # 将中位数对应的id获取
        median_row_id = sort_num_list[median_pos][0]
        median_row_id_width = sort_num_list[median_pos][1]
        # 获取绘制的直线起始和终止点坐标位置
        line_start_pos = (frontier_pos_list[median_row_id][0], median_row_id)
        line_end_pos = (frontier_pos_list[median_row_id][1], median_row_id)

        # 绘制
        # 1) 将crop出来的frontier小图转换成rgb
        crop_ori_roi = input_image[roi_start_y:roi_end_y, roi_start_x:roi_end_x]
        crop_ori_roi_bgr = cv2.cvtColor(crop_ori_roi, cv2.COLOR_GRAY2BGR)
        # 2) 创建一个带有横幅的图像buffer， frontier_text_region_height就是这个横幅的高度
        draw_panel = np.zeros((crop_ori_roi.shape[0]+frontier_text_region_height, crop_ori_roi.shape[1], 3), np.uint8)
        # 3) 将图像拷贝到draw_panel里面
        draw_panel[frontier_text_region_height:, :, :] = crop_ori_roi_bgr
        # 4） 绘制frontier小横线，但是要加上横幅高度这个offset
        cv2.line(draw_panel, (line_start_pos[0], line_start_pos[1]+frontier_text_region_height), (line_end_pos[0], line_end_pos[1]+frontier_text_region_height),
                 (0, 0, 255), 2)
        # 5) 写下标题文字
        cv2.putText(draw_panel, "Edge measure " + str(pt_id) +": " + str(median_row_id_width * dilate_ratio * camera_resolution) + " um",
                    (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        cv2.putText(draw_panel, "Pos: " + "(x=" + str(pt[0] * dilate_ratio * camera_resolution) + " um, y=" + str(pt[1] * dilate_ratio * camera_resolution) + " um)",
                    (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        result_img_list.append(draw_panel)
        result_w_list.append(median_row_id_width)
        crop_rect_list.append([roi_start_x, roi_start_y, roi_end_x-roi_start_x, roi_end_y-roi_start_y])
        # cv2.imwrite("D:/SAVE_B10_CELL_RESULT/2021_7_20/frontier/30000B000C0_0_defectClass/" + str(pt_id) +".bmp", draw_panel)
    # 数据存储
    frontier_measure_result_dict = {'frontier_width1': result_w_list[0], 'crop_img1': result_img_list[0],
                                    'frontier_width2': result_w_list[1], 'crop_img2': result_img_list[1],
                                    'frontier_width3': result_w_list[2], 'crop_img3': result_img_list[2],
                                    'frontier_average': np.mean(result_w_list), 'crop_rect1': crop_rect_list[0],
                                    'crop_rect2': crop_rect_list[1], 'crop_rect3': crop_rect_list[2]}

    return frontier_measure_result_dict


def get_frontier_result(input_image, direction_type, resize_ratio, fit_vertical_line, rotate_angle):
    """
    功能：获取磨边的宽度
    描述：根据获取到的内点或内点直线（首尾两个点）大致crop出一个略宽一点的矩形，
         然后根据旋转矩阵把这个矩形旋转，然后再内缩一下，去除旋转后的黑色填充。
         然后根据这个矩形计算每一行的磨边宽度，获取磨边梯度的两个点，这个点col坐标需要记下来，便于绘制存储小图。
    """
    frontier_mesure_result_dict = {'frontier_width': 0, 'crop_img': None}
    # 裁切roi区域，未来对roi区域进行磨边宽度测量
    roi_start_x = fit_vertical_line[0][0] - offset_x
    roi_start_y = fit_vertical_line[0][1] - offset_y
    roi_end_x = roi_start_x + 2 * offset_x
    roi_end_y = fit_vertical_line[0][1] + 2 * offset_y
    if roi_start_x < 0:
        roi_start_x = 0
    if roi_start_y < 0:
        roi_start_y = 0
    if roi_end_x > input_image.shape[1]:
        roi_end_x = input_image.shape[1]
    if roi_end_y > input_image.shape[0]:
        roi_end_y = input_image.shape[0]
    crop_roi = input_image[roi_start_y:roi_end_y, roi_start_x:roi_end_x]
    rotate_matrix = calcul_rotate_matrix(rotate_angle, crop_roi.shape[1], crop_roi.shape[0])
    rotate_crop_roi = cv2.warpAffine(crop_roi, rotate_matrix, (crop_roi.shape[1], crop_roi.shape[0]))
    rotate_crop_roi_copy = rotate_crop_roi.copy()

    # 有些小图的磨边左右两侧有一些浅白色的边缘，这些边缘一旦过宽，会影响测距，所以首先对这些区域像素先进行暗色填充。
    interet_pos = np.vstack(np.where(rotate_crop_roi > high_bright_th))
    interet_pt_num = interet_pos[0].shape[0]
    if interet_pt_num > 0:
        rotate_crop_roi[interet_pos[0, :], interet_pos[1, :]] = fill_high_bright_val
    # rotate_crop_roi = cv2.GaussianBlur(rotate_crop_roi, (3, 3), 1, 0)  # 高斯模糊
    rotate_crop_roi = cv2.blur(rotate_crop_roi, (3, 15))
    struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    rotate_crop_roi = cv2.erode(rotate_crop_roi, struct_element)

    # 根据左右相机进行判断
    step_x = 1  # 遍历步长
    frontier_pos_x = []
    row_id = rotate_crop_roi.shape[0] // 2
    if direction_type == 'left':
        start_x = 1
        end_x = rotate_crop_roi.shape[1] - step_x
    else:
        start_x = rotate_crop_roi.shape[1] - 1 - step_x
        end_x = 0 + step_x
        step_x = -step_x

    # 开始测量： 假设，磨边的左右两侧超差 要比 磨边位置与起始位置超差小
    # 假设左相机，从左侧第一个像素开始遍历，一旦找到超差救济路这个点坐标，
    # 然后，利用后续遍历像素与超差前一个像素进行比较，如果再次发现出现了更加小的超差，证明已经超过了磨边到达了磨边的右侧区域了
    for col_id in range(start_x, end_x, step_x):
        if len(frontier_pos_x) == 2:
            break
        if len(frontier_pos_x) == 0:
            previous_val = rotate_crop_roi[row_id, col_id - step_x]
            current_val = rotate_crop_roi[row_id, col_id]
            abs_diff = abs(np.int(current_val) - np.int(previous_val))
            if abs_diff > high_variance:
                frontier_pos_x.append(col_id)
        else:
            current_val = rotate_crop_roi[row_id, col_id]
            abs_diff = abs(np.int(current_val) - np.int(previous_val))
            if abs_diff < low_variance:
                frontier_pos_x.append(col_id)

    # 如果找到的磨边坐标小于2个就是异常现象了
    if len(frontier_pos_x) != 2:
        frontier_width = 0
        rotate_crop_roi_bgr = cv2.cvtColor(rotate_crop_roi_copy, cv2.COLOR_GRAY2BGR)
        cv2.putText(rotate_crop_roi_bgr, 'Exception !', (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        frontier_mesure_result_dict['crop_img'] = rotate_crop_roi_bgr
    else:
        real_ratio = np.int(1 // resize_ratio)
        frontier_width = abs(frontier_pos_x[0] - frontier_pos_x[1])
        rotate_crop_roi_bgr = cv2.cvtColor(rotate_crop_roi_copy, cv2.COLOR_GRAY2BGR)
        cv2.line(rotate_crop_roi_bgr, (frontier_pos_x[0], row_id), (frontier_pos_x[1], row_id), (0, 0, 255), 2)
        cv2.putText(rotate_crop_roi_bgr, "W: "+str(frontier_width*real_ratio), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 255), 1)
        frontier_mesure_result_dict['frontier_width'] = frontier_width
        frontier_mesure_result_dict['crop_img'] = rotate_crop_roi_bgr
    return frontier_mesure_result_dict


def crop_and_draw_mark(input_img, dilate_ratio, mark_center_x, mark_center_y, hori_frontier_x, hori_frontier_y,
                       vert_frontier_x, vert_frontier_y, dist_x, dist_y, offset_xy):

    # 将输入灰度图转换RGB图像
    input_bgr_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
    # 绘制水平和垂直距离的线段
    cv2.line(input_bgr_img, (mark_center_x, mark_center_y), (hori_frontier_x, hori_frontier_y), (0, 0, 255), 4)
    cv2.line(input_bgr_img, (mark_center_x, mark_center_y), (vert_frontier_x, vert_frontier_y), (0, 0, 255), 4)

    # 定义crop的具体坐标， 但是要保证的是crop出来的图像尺寸都是一致的
    crop_x = mark_center_x - offset_xy
    if crop_x <= 0:
        crop_x = 0

    crop_y = mark_center_y - offset_xy
    if crop_y <= 0:
        crop_y = 0

    crop_right = crop_x + offset_xy * 2
    if crop_x == 0:
        crop_right = crop_x + offset_xy * 2
    elif crop_right > input_img.shape[1]:
        crop_right = input_img.shape[1]
        crop_x = input_img.shape[1] - offset_xy * 2

    crop_bottom = crop_y + offset_xy * 2
    if crop_y == 0:
        crop_bottom = crop_y + offset_xy * 2
    elif crop_bottom > input_img.shape[0]:
        crop_bottom = input_img.shape[0]
        crop_y = input_img.shape[0] - offset_xy * 2

    crop_rect = [crop_x, crop_y, crop_right-crop_x, crop_bottom-crop_y]

    # 1) 将大图中绘制的mark区域进行crop
    crop_bgr_img = input_bgr_img[crop_y:crop_bottom, crop_x:crop_right, :]
    # 2) 创建一个带有横幅的图像buffer， frontier_text_region_height就是这个横幅的高度
    crop_img_h, crop_img_w = crop_bgr_img.shape[0], crop_bgr_img.shape[1]
    draw_panel = np.zeros((crop_img_h + mark_text_region_height, crop_img_w, 3), np.uint8)
    # 3) 将图像拷贝到draw_panel里面
    draw_panel[mark_text_region_height:, :, :] = crop_bgr_img
    # 4) 写下标题文字
    cv2.putText(draw_panel, "Vert distance: " + str(int(dist_y * dilate_ratio)*camera_resolution) + " um",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(draw_panel, "Hori distance: " + str(int(dist_x * dilate_ratio)*camera_resolution) + " um",
                (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return draw_panel, crop_rect


def image_stitiching(crop_img_list, crop_rect_list, image_height, image_width):
    # 设置整图像buffer
    stitching_image = np.zeros((image_height, image_width), np.uint8)

    for id, rect in enumerate(crop_rect_list):
        left = rect[0]
        top = rect[1]
        right = rect[0] + rect[2]
        bottom = rect[1] + rect[3]
        stitching_crop_img = stitching_image[top:bottom, left:right]
        merge_crop_img = np.bitwise_or(stitching_crop_img, crop_img_list[id])
        stitching_image[top:bottom, left:right] = merge_crop_img
    return stitching_image


def get_mark_region(bw_img, mark_area_th):

    resize_ratio = 0.5
    recovery_ratio = 1 / resize_ratio
    bw_img_resize = cv2.resize(bw_img, (0, 0), fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)
    _, bw_img_resize = cv2.threshold(bw_img_resize, 10, 255, cv2.THRESH_BINARY)
    mark_area_th = mark_area_th * resize_ratio * resize_ratio

    label_img = label(bw_img_resize, neighbors=8, connectivity=2)
    all_connect_info = regionprops(label_img)
    mark_centroid_list = []

    # 证明没有找到任何的mark
    if len(all_connect_info) == 0:
        return mark_centroid_list

    # 存储找到的mark的中心点，找到就退出
    for element in all_connect_info:
        if element.filled_area > mark_area_th:
            # 存储成x,y形式，默认是element.centroid=[y,x]
            mark_centroid_list.append([int(element.centroid[1]*recovery_ratio), int(element.centroid[0]*recovery_ratio)])
            break
    return mark_centroid_list


def get_corner_region(bw_img, corner_area_th):

    resize_ratio = 0.5
    recovery_ratio = 1 / resize_ratio
    bw_img_resize = cv2.resize(bw_img, (0, 0), fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)
    _, bw_img_resize = cv2.threshold(bw_img_resize, 10, 255, cv2.THRESH_BINARY)
    corner_area_th = corner_area_th * resize_ratio * resize_ratio

    label_img = label(bw_img_resize, neighbors=8, connectivity=2)
    all_connect_info = regionprops(label_img)
    corner_pos_result_list = []

    # 证明没有找到任何的corner
    if len(all_connect_info) == 0:
        return corner_pos_result_list

    # 存储找到的corner的中心点，找到就退出
    corner_area_list, corner_area_ratio_list, corner_pos_list = [], [], []
    for element in all_connect_info:
        corner_area_list.append(element.filled_area)
        area_ratio = element.bbox_area / element.area
        corner_area_ratio_list.append(area_ratio)
        corner_x, corner_y = element['bbox'][1], element['bbox'][0]
        corner_h = abs(element['bbox'][0] - element['bbox'][2])
        corner_w = abs(element['bbox'][1] - element['bbox'][3])
        corner_pos_list.append([corner_x, corner_y, corner_w, corner_h])

    max_id = corner_area_ratio_list.index(max(corner_area_ratio_list))
    if corner_area_list[max_id] > corner_area_th:
        corner_pos_list[max_id][0] = int(corner_pos_list[max_id][0] * recovery_ratio)
        corner_pos_list[max_id][1] = int(corner_pos_list[max_id][1] * recovery_ratio)
        corner_pos_list[max_id][2] = int(corner_pos_list[max_id][2] * recovery_ratio)
        corner_pos_list[max_id][3] = int(corner_pos_list[max_id][3] * recovery_ratio)

        corner_pos_result_list.append(corner_pos_list[max_id])
    return corner_pos_result_list

def crop_and_draw_corner_tradition(input_img, dilate_ratio, corner_pos_list, offset_xy, img_upper_below_type='upper'):
    # 将输入灰度图转换RGB图像
    input_bgr_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
    # 绘制corner的矩形
    corner_x, corner_y = corner_pos_list[0][0], corner_pos_list[0][1]
    corner_w, corner_h = corner_pos_list[0][2], corner_pos_list[0][3]
    # cv2.rectangle(input_bgr_img, (corner_x, corner_y), (corner_x+corner_w, corner_y+corner_h), (0, 0, 255), 1)

    if img_upper_below_type == 'upper':
        cv2.line(input_bgr_img, (corner_x, corner_y), (corner_x + corner_w, corner_y), (0, 0, 255), 1)  # 绘制水平线
        cv2.line(input_bgr_img, (corner_x + corner_w, corner_y), (corner_x + corner_w, corner_y + corner_h),
                 (0, 0, 255), 1)  # 绘制垂直线
        cv2.line(input_bgr_img, (corner_x, corner_y), (corner_x + corner_w, corner_y + corner_h), (0, 0, 255),
                 1)  # 绘制R角线段
    else:
        cv2.line(input_bgr_img, (corner_x, corner_y + corner_h), (corner_x + corner_w, corner_y + corner_h),
                 (0, 0, 255), 1)  # 绘制水平线
        cv2.line(input_bgr_img, (corner_x + corner_w, corner_y), (corner_x + corner_w, corner_y + corner_h),
                 (0, 0, 255), 1)  # 绘制垂直线
        cv2.line(input_bgr_img, (corner_x, corner_y + corner_h), (corner_x + corner_w, corner_y), (0, 0, 255),
                 1)  # 绘制R角线段

    # 定义crop的具体坐标， 但是要保证的是crop出来的图像尺寸都是一致的（corner_center就是corner矩形的中心坐标）
    corner_center_x, corner_center_y = int(corner_x + (corner_w / 2)), int(corner_y + (corner_h / 2))
    crop_x = corner_center_x - offset_xy
    if crop_x <= 0:
        crop_x = 0

    crop_y = corner_center_y - offset_xy
    if crop_y <= 0:
        crop_y = 0

    crop_right = crop_x + offset_xy * 2
    if crop_x == 0:
        crop_right = crop_x + offset_xy * 2
    elif crop_right > input_img.shape[1]:
        crop_right = input_img.shape[1]
        crop_x = input_img.shape[1] - offset_xy * 2

    crop_bottom = crop_y + offset_xy * 2
    if crop_y == 0:
        crop_bottom = crop_y + offset_xy * 2
    elif crop_bottom > input_img.shape[0]:
        crop_bottom = input_img.shape[0]
        crop_y = input_img.shape[0] - offset_xy * 2

    crop_rect = [crop_x, crop_y, crop_right - crop_x, crop_bottom - crop_y]

    # 1) 将大图中绘制的mark区域进行crop
    crop_bgr_img = input_bgr_img[crop_y:crop_bottom, crop_x:crop_right, :]
    # 2) 创建一个带有横幅的图像buffer，corner_text_region_height就是这个横幅的高度
    crop_img_h, crop_img_w = crop_bgr_img.shape[0], crop_bgr_img.shape[1]
    draw_panel = np.zeros((crop_img_h + corner_text_region_height, crop_img_w, 3), np.uint8)
    # 3) 将图像拷贝到draw_panel里面
    draw_panel[corner_text_region_height:, :, :] = crop_bgr_img
    # 4) 写下标题文字
    cv2.putText(draw_panel, "Corner width: " + str(int(corner_w * dilate_ratio * camera_resolution)) + " um",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(draw_panel, "Corner height: " + str(int(corner_h * dilate_ratio * camera_resolution)) + " um",
                (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return draw_panel, crop_rect


def get_mark_result(input_image, image_location_class, mark_crop_list, crop_rect_list, direction_type,
                    product_type, product_id, resize_ratio, dilate_ratio, rotate_angle, fit_vertical_line):
    """
    功能：获取mark到上（下）和左（右）边距，但是需要判断在固定位置到底是否存在有mark，有才能提供测量结果。
    描述：如果在上述指定条件下，首先判断是否存在mark，如果不存在直接返回无，否则：记录mark中心点坐标，
         根据这个中心点在指定区域下crop出来一个稍大检测区域，计算检测区域第一行和第一列的std用来填充旋转后的黑边。
         然后crop出来的大图旋转，最后根据逐行的std和逐列的std判断出坐标位置，根据这些与mark中心坐标计算边距。
    参数：
     image_location_class是列表中其中一个元素： ['NO_OBJ', 'ALL', 'UPPER', 'BELOW', 'MIDDLE']
    """

    mark_mesure_result_dict = {'upper_mark': [0, 0], 'below_mark': [0, 0],
                               'upper_crop_img': None, 'below_crop_img': None,
                               'upper_rect': [], 'below_rect': []}
    img_w, img_h = input_image.shape[1], input_image.shape[0]
    stitiching_mark_img = image_stitiching(mark_crop_list, crop_rect_list, img_h, img_w)
    mark_area_th = int(mark_size_th * resize_ratio * resize_ratio)
    if direction_type == 'left':
        cut_pt_x, cut_pt_y = fit_vertical_line[0][0] - 200, img_h // 2
    else:
        cut_pt_x, cut_pt_y = fit_vertical_line[0][0] + 200, img_h // 2

    # ***** 上图 *****
    # if product_id == 0 and (product_type == 'big' or product_type == 'middle'):
    if image_location_class == 'UPPER':
        crop_rotate_stitiching_mark_img = stitiching_mark_img[0:cut_pt_y, :]
        mark_centroid_list = get_mark_region(crop_rotate_stitiching_mark_img, mark_area_th)
        upper_mark_centroid_x, upper_mark_centroid_y = mark_centroid_list[0][0], mark_centroid_list[0][1]
        inlier_pt_list, _ = vertical_projection(input_image, direction_type, COL_STEP, FIT_PT_NUM, L1_THRESH)
        upper_hori_frontier_x, upper_hori_frontier_y = upper_mark_centroid_x, inlier_pt_list[-1][1]
        vert_frontier_x, vert_frontier_y = fit_vertical_line[0][0], upper_mark_centroid_y
        dist_x_upper = np.abs(fit_vertical_line[0][0] - upper_mark_centroid_x)
        dist_y_upper = np.abs(inlier_pt_list[0][1] - upper_mark_centroid_y)

        # 绘制
        temp = input_image[0:cut_pt_y, :]
        crop_bgr_img, crop_rect = crop_and_draw_mark(temp, dilate_ratio, upper_mark_centroid_x, upper_mark_centroid_y,
                                          upper_hori_frontier_x, upper_hori_frontier_y, vert_frontier_x,
                                          vert_frontier_y, dist_x_upper, dist_y_upper, crop_mark_offset)
        # 存储
        mark_mesure_result_dict['upper_mark'] = [int(dist_x_upper), int(dist_y_upper)]
        mark_mesure_result_dict['upper_crop_img'] = crop_bgr_img
        mark_mesure_result_dict['upper_rect'] = crop_rect

    # ***** 下图 *****
    # elif (product_id == 1 and product_type == 'middle') or product_id == 2:
    elif image_location_class == 'BELOW':
        crop_rotate_stitiching_mark_img = stitiching_mark_img[cut_pt_y:, :]
        mark_centroid_list = get_mark_region(crop_rotate_stitiching_mark_img, mark_area_th)
        below_mark_centroid_x, below_mark_centroid_y = mark_centroid_list[0][0], mark_centroid_list[0][1]
        below_mark_centroid_y = int(below_mark_centroid_y + cut_pt_y)
        flip_input_img = cv2.flip(input_image, 0)
        inlier_pt_list, _ = vertical_projection(flip_input_img, direction_type, COL_STEP, FIT_PT_NUM, L1_THRESH)
        real_inlier_pt = [[pt[0], img_h - pt[1]] for pt in inlier_pt_list]
        below_hori_frontier_x, below_hori_frontier_y = below_mark_centroid_x, real_inlier_pt[-1][1]
        vert_frontier_x, vert_frontier_y = fit_vertical_line[-1][0], below_mark_centroid_y
        dist_x_below = np.abs(fit_vertical_line[1][0] - below_mark_centroid_x)
        dist_y_below = np.abs(real_inlier_pt[-1][1] - below_mark_centroid_y)

        # 绘制
        temp = input_image[cut_pt_y:, :]
        crop_bgr_img, crop_rect = crop_and_draw_mark(temp, dilate_ratio, below_mark_centroid_x,
                                                     below_mark_centroid_y - cut_pt_y, below_hori_frontier_x,
                                                     below_hori_frontier_y - cut_pt_y, vert_frontier_x,
                                                     vert_frontier_y - cut_pt_y, dist_x_below, dist_y_below,
                                                     crop_mark_offset)
        crop_rect[1] = crop_rect[1] + cut_pt_y

        # 存储
        mark_mesure_result_dict['below_mark'] = [int(dist_x_below), int(dist_y_below)]
        mark_mesure_result_dict['below_crop_img'] = crop_bgr_img
        mark_mesure_result_dict['below_rect'] = crop_rect

    # ***** 全图 *****
    # elif product_id == 0 and product_type == 'small':
    elif image_location_class == 'ALL':
        # 计算上半部分的mark
        crop_rotate_stitiching_mark_img = stitiching_mark_img[0:cut_pt_y, :]
        mark_centroid_list = get_mark_region(crop_rotate_stitiching_mark_img, mark_area_th)
        upper_mark_centroid_x, upper_mark_centroid_y = mark_centroid_list[0][0], mark_centroid_list[0][1]
        inlier_pt_list, _ = vertical_projection(input_image, direction_type, COL_STEP, FIT_PT_NUM, L1_THRESH)
        upper_hori_frontier_x, upper_hori_frontier_y = upper_mark_centroid_x, inlier_pt_list[-1][1]
        upper_vert_frontier_x, upper_vert_frontier_y = fit_vertical_line[0][0], upper_mark_centroid_y
        dist_x_upper = np.abs(fit_vertical_line[0][0] - upper_mark_centroid_x)
        dist_y_upper = np.abs(inlier_pt_list[0][1] - upper_mark_centroid_y)

        # 计算下半部分的mark
        crop_rotate_stitiching_mark_img = stitiching_mark_img[cut_pt_y:, :]
        mark_centroid_list = get_mark_region(crop_rotate_stitiching_mark_img, mark_area_th)
        below_mark_centroid_x, below_mark_centroid_y = mark_centroid_list[0][0], mark_centroid_list[0][1]
        below_mark_centroid_y = int(below_mark_centroid_y + cut_pt_y)
        flip_input_img = cv2.flip(input_image, 0)
        inlier_pt_list, _ = vertical_projection(flip_input_img, direction_type, COL_STEP, FIT_PT_NUM, L1_THRESH)
        real_inlier_pt = [[pt[0], img_h - pt[1]] for pt in inlier_pt_list]
        below_hori_frontier_x, below_hori_frontier_y = below_mark_centroid_x, real_inlier_pt[-1][1]
        below_vert_frontier_x, below_vert_frontier_y = fit_vertical_line[-1][0], below_mark_centroid_y
        dist_x_below = np.abs(fit_vertical_line[1][0] - below_mark_centroid_x)
        dist_y_below = np.abs(real_inlier_pt[-1][1] - below_mark_centroid_y)

        # 绘制
        temp_upper, temp_below = input_image[0:cut_pt_y, :], input_image[cut_pt_y:, :]
        crop_upper_bgr_img, crop_rect_upper = crop_and_draw_mark(temp_upper, dilate_ratio, upper_mark_centroid_x,
                                                                 upper_mark_centroid_y, upper_hori_frontier_x,
                                                                 upper_hori_frontier_y, upper_vert_frontier_x,
                                                                 upper_vert_frontier_y, dist_x_upper, dist_y_upper,
                                                                 crop_mark_offset)
        crop_below_bgr_img, crop_rect_below = crop_and_draw_mark(temp_below, dilate_ratio, below_mark_centroid_x,
                                                                 below_mark_centroid_y - cut_pt_y, below_hori_frontier_x
                                                                 , below_hori_frontier_y - cut_pt_y,
                                                                 below_vert_frontier_x, below_vert_frontier_y - cut_pt_y
                                                                 , dist_x_below, dist_y_below, crop_mark_offset)
        crop_rect_below[1] = crop_rect_below[1] + cut_pt_y
        # 存储
        mark_mesure_result_dict['upper_mark'] = [int(dist_x_upper), int(dist_y_upper)]
        mark_mesure_result_dict['below_mark'] = [int(dist_x_below), int(dist_y_below)]
        mark_mesure_result_dict['upper_crop_img'] = crop_upper_bgr_img
        mark_mesure_result_dict['below_crop_img'] = crop_below_bgr_img
        mark_mesure_result_dict['upper_rect'] = crop_rect_upper
        mark_mesure_result_dict['below_rect'] = crop_rect_below
    else:
        mark_mesure_result_dict['upper_mark'] = [0, 0]
        mark_mesure_result_dict['below_mark'] = [0, 0]
        mark_mesure_result_dict['upper_crop_img'] = None
        mark_mesure_result_dict['below_crop_img'] = None
        mark_mesure_result_dict['upper_rect'] = []
        mark_mesure_result_dict['below_rect'] = []

    return mark_mesure_result_dict


def get_corner_deeplearning(input_image, corner_crop_list, crop_rect_list, product_type, product_id, resize_ratio,
                      dilate_ratio, rotate_angle):
    corner_mesure_result_dict = {'upper_corner': [0, 0], 'below_corner': [0, 0], 'upper_crop_img': None,
                                 'below_crop_img': None, 'upper_rect': [], 'below_rect': []}
    img_w, img_h = input_image.shape[1], input_image.shape[0]
    stitiching_mark_img = image_stitiching(corner_crop_list, crop_rect_list, img_h, img_w)
    corner_area_th = int(corner_size_th * resize_ratio * resize_ratio)
    cut_pt_y = img_h // 2
    if product_id == 0 and (product_type == 'big' or product_type == 'middle'):
        crop_rotate_stitiching_mark_img = stitiching_mark_img[0:cut_pt_y, :]
        corner_pos_list = get_corner_region(crop_rotate_stitiching_mark_img, corner_area_th)
        if len(corner_pos_list) == 0:
            corner_mesure_result_dict['upper_corner'] = [0, 0]
            corner_mesure_result_dict['upper_crop_img'] = None
            corner_mesure_result_dict['upper_rect'] = []
        else:
            # 绘制
            temp = input_image[0:cut_pt_y, :]
            crop_bgr_img, crop_rect = crop_and_draw_corner_tradition(temp, dilate_ratio, corner_pos_list, crop_corner_offset)

            # 存储
            corner_mesure_result_dict['upper_corner'] = [int(corner_pos_list[0][2]), int(corner_pos_list[0][3])]
            corner_mesure_result_dict['upper_crop_img'] = crop_bgr_img
            corner_mesure_result_dict['upper_rect'] = crop_rect

    elif (product_id == 1 and product_type == 'middle') or product_id == 2:
        crop_rotate_stitiching_mark_img = stitiching_mark_img[cut_pt_y:, :]
        corner_pos_list = get_corner_region(crop_rotate_stitiching_mark_img, corner_area_th)
        if len(corner_pos_list) == 0:
            corner_mesure_result_dict['below_corner'] = [0, 0]
            corner_mesure_result_dict['below_crop_img'] = None
            corner_mesure_result_dict['below_rect'] = []
        else:
            # 绘制
            temp = input_image[cut_pt_y:, :]
            crop_bgr_img, crop_rect = crop_and_draw_corner_tradition(temp, dilate_ratio, corner_pos_list, crop_corner_offset)
            crop_rect[1] = crop_rect[1] + cut_pt_y
            # 存储
            corner_mesure_result_dict['below_corner'] = [int(corner_pos_list[0][2]), int(corner_pos_list[0][3])]
            corner_mesure_result_dict['below_crop_img'] = crop_bgr_img
            corner_mesure_result_dict['below_rect'] = crop_rect

    elif product_id == 0 and product_type == 'small':
        # 上半部分图像
        crop_rotate_stitiching_mark_img = stitiching_mark_img[0:cut_pt_y, :]
        corner_pos_list = get_corner_region(crop_rotate_stitiching_mark_img, corner_area_th)
        if len(corner_pos_list) == 0:
            corner_mesure_result_dict['upper_corner'] = [0, 0]
            corner_mesure_result_dict['upper_crop_img'] = None
            corner_mesure_result_dict['upper_rect'] = []
        else:
            # 绘制
            temp = input_image[0:cut_pt_y, :]
            crop_bgr_img, crop_rect = crop_and_draw_corner_tradition(temp, dilate_ratio, corner_pos_list, crop_corner_offset)

            # 存储
            corner_mesure_result_dict['upper_corner'] = [int(corner_pos_list[0][2]), int(corner_pos_list[0][3])]
            corner_mesure_result_dict['upper_crop_img'] = crop_bgr_img
            corner_mesure_result_dict['upper_rect'] = crop_rect
        # 下半部分图像
        crop_rotate_stitiching_mark_img = stitiching_mark_img[cut_pt_y:, :]
        corner_pos_list = get_corner_region(crop_rotate_stitiching_mark_img, corner_area_th)
        if len(corner_pos_list) == 0:
            corner_mesure_result_dict['below_corner'] = [0, 0]
            corner_mesure_result_dict['below_crop_img'] = None
            corner_mesure_result_dict['below_rect'] = []
        else:
            # 绘制
            temp = input_image[cut_pt_y:, :]
            crop_bgr_img, crop_rect = crop_and_draw_corner_tradition(temp, dilate_ratio, corner_pos_list, crop_corner_offset)
            crop_rect[1] = crop_rect[1] + cut_pt_y
            # 存储
            corner_mesure_result_dict['below_corner'] = [int(corner_pos_list[0][2]), int(corner_pos_list[0][3])]
            corner_mesure_result_dict['below_crop_img'] = crop_bgr_img
            corner_mesure_result_dict['below_rect'] = crop_rect

    return corner_mesure_result_dict



def crop_corner_roi(input_image, cross_pt):
    """
    说明：根据水平拟合直线和垂直拟合直线，找到相交点，根据这个相交点扣取一个ROI区域便于找到三角形。
    """
    crop_radius_x, crop_radius_y = 400, 400
    image_height, image_width = input_image.shape[0], input_image.shape[1]
    roi_left = cross_pt[0] - crop_radius_x
    roi_top = cross_pt[1]
    roi_right = cross_pt[0]
    roi_bottom = cross_pt[1] + crop_radius_y
    if roi_left < 0:
        roi_left = 0
    if roi_bottom > image_height:
        roi_bottom = image_height

    roi_w = roi_right - roi_left
    roi_h = roi_bottom - roi_top
    crop_corner_roi_rect = [roi_left, roi_top, roi_w, roi_h]
    crop_corner_roi_img = input_image[roi_top:roi_bottom, roi_left:roi_right]
    return crop_corner_roi_rect, crop_corner_roi_img


def find_triangle_edge(crop_corner_roi_img, contour_mask):
    """
    说明：根据交点crop出来的ROI区域，从交点开始对列进行垂直投影，最终根据条件找到三角形的另外两个角点
    方法：1）因为图像都是在upper的了， 所以交点位置开始做垂直投影，从右向左侧投影。
         2）第一个端点：即竖线端点，是投影的第一列，如果遇到白像素就是端点。
         3）第二个端点：即横线端点，是投影到左侧的时候，如发现把像素的y坐标与交点y坐标间距很闲的时候，证明找到。
         o-----o
          \    |
            \  |
              \|
               o
    """
    roi_h, roi_w = crop_corner_roi_img.shape[0], crop_corner_roi_img.shape[1]
    height_th = 3
    # 1) 寻找垂直线段的倒角端点
    # -4的目的是留有一些冗余， 因为如果图像倾斜很有可能最后一列的白像素不是纵向想找到的端点
    find_vert_flag = False
    for col_id in range(roi_w-4, -1, -1):
        if find_vert_flag == True:  # 如果在纵向找到了白色点，证明已经找到端点，马上退出。
            break
        for row_id in range(0, roi_h):
            cur_val = contour_mask[row_id, col_id]
            # 如果crop出来的图的交叉点的列找到255像素值，就记录当前位置，然后停止。
            if cur_val == 255:
                endpoint_vert = [col_id+3, row_id]  # 把前面冗余补充回来
                find_vert_flag = True
                break
    # 2）寻找水平线段的倒角端点(从最后一列便利，计算每列遇到白像素的高度,如果高度小于阈值，证明找到)
    for col_id in range(roi_w-1, -1, -1):
        white_pos = np.vstack(np.where(contour_mask[:, col_id] == 255))
        if white_pos[0].shape[0] == 0:
            continue
        each_col_first_white_pos = white_pos[0][0]  # 取[1]代表row坐标集合, [0]代表遇到的第一个白像素行id
        if each_col_first_white_pos < height_th:
            endpoint_hori = [col_id, each_col_first_white_pos]
            break
    return endpoint_hori, endpoint_vert


def find_triangle_2_corner_pt(vertical_inlier_pt_list, horizontal_inlier_pt_list, input_image):
    '''说明： 根据真实图像水平和垂直投影的内点列表， 估算出水平和垂直直线的延长线， 找到交点， 再估算出三角形另外两个端点'''
    image_width = input_image.shape[1]
    # 线段起点： 确定水平线和垂直线的起点坐标(起点坐标存储是[x, y]结构的)
    vertical_start_pt = vertical_inlier_pt_list[1]
    horizontal_start_pt = horizontal_inlier_pt_list[-3]
    # 线段终点：确定水平线和垂直线的终点坐标(终点坐标存储是[x, y]结构的)
    vertical_end_pt = [vertical_start_pt[0], 0]
    horizontal_end_pt = [image_width, horizontal_start_pt[1]]
    # 两条直线的交点(cross_pt格式[x, y])
    cross_pt = findIntersection(horizontal_start_pt, horizontal_end_pt, vertical_start_pt, vertical_end_pt)
    # crop出来倒角的检测区域(crop_corner_roi_rect格式[x, y, w, h], crop_corner_roi_img是gray)
    crop_corner_roi_rect, crop_corner_roi_img = crop_corner_roi(input_image, cross_pt)
    # 获取roi的外边缘轮廓(contour_mask是二值化)
    contour_mask = contour_detect(crop_corner_roi_img)
    # 检测水平和垂直端点（三角形的两个角点）
    endpoint_hori, endpoint_vert = find_triangle_edge(crop_corner_roi_img, contour_mask)
    # 加上偏差，获取大图上的绝对坐标位置
    offset_x, offset_y = crop_corner_roi_rect[0], crop_corner_roi_rect[1]
    endpoint_hori = [endpoint_hori[0] + offset_x, endpoint_hori[1] + offset_y]
    endpoint_vert = [endpoint_vert[0] + offset_x, endpoint_vert[1] + offset_y]
    corner_w = abs(endpoint_hori[0] - cross_pt[0])
    corner_h = abs(endpoint_vert[1] - cross_pt[1])

    # # TEST CODE...
    # input_image_bgr = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    # cv2.line(input_image_bgr, (horizontal_start_pt[0], horizontal_start_pt[1]), (horizontal_end_pt[0], horizontal_end_pt[1]),(255,0,0), 2)
    # cv2.line(input_image_bgr, (vertical_start_pt[0], vertical_start_pt[1]), (vertical_end_pt[0], vertical_end_pt[1]),(0,0,255), 2)
    # cv2.circle(input_image_bgr, (cross_pt[0], cross_pt[1]), 2, (0, 0, 255), -1)
    # for pt in vertical_inlier_pt_list:
    #     cv2.circle(input_image_bgr, (pt[0], pt[1]), 3, (0, 0, 255), -1)
    # cv2.imwrite("D:/B10_cell_detection/B10_demo_python/debug_file/inference/2021_7_14/corner/0_cross_pt.bmp", input_image_bgr)
    # cv2.imwrite("D:/B10_cell_detection/B10_demo_python/debug_file/inference/2021_7_14/corner/1_roi.bmp", crop_corner_roi_img)
    # cv2.imwrite("D:/B10_cell_detection/B10_demo_python/debug_file/inference/2021_7_14/corner/2_contour.bmp", contour_mask)
    #
    # input_image_bgr = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    # cv2.line(input_image_bgr, (endpoint_hori[0], endpoint_hori[1]), (cross_pt[0], cross_pt[1]), (255, 0, 0), 2)
    # cv2.line(input_image_bgr, (endpoint_vert[0], endpoint_vert[1]), (cross_pt[0], cross_pt[1]), (0,0,255), 2)
    # cv2.imwrite("D:/B10_cell_detection/B10_demo_python/debug_file/inference/2021_7_14/corner/3_corner.bmp", input_image_bgr)
    return endpoint_hori[0], endpoint_hori[1], corner_w, corner_h


def calcul_hori_pos(canny_bw_img, image_location_class, row_step=3):
    '''说明： 利用canny图像进行水平投影， 根据每行的白像素数量，大致定位到边缘的行坐标信息'''
    height, width = canny_bw_img.shape[0], canny_bw_img.shape[1]
    roi_y = 0
    if image_location_class == 'UPPER':
        # 从上到下循环
        for row_id in range(0, height, row_step):
            white_pos = np.vstack(np.where(canny_bw_img[row_id, :] == 255))
            if white_pos[0].shape[0] == 0:
                continue
            if white_pos[0].shape[0] > (width // 10):
                roi_y = row_id
                break
    else:
        # 从下到上循环
        for last_row_id in range(height-1, -1, -row_step):
            white_pos = np.vstack(np.where(canny_bw_img[last_row_id, :] == 255))
            if white_pos[0].shape[0] == 0:
                continue
            if white_pos[0].shape[0] > (width // 10):
                roi_y = last_row_id
                break
    return roi_y


def calcul_vert_pos(canny_bw_img, col_step=3, direction_type='left'):
    '''说明： 利用canny图像进行垂直投影， 根据每列的白像素数量，大致定位到边缘的列坐标信息'''
    height, width = canny_bw_img.shape[0], canny_bw_img.shape[1]
    roi_x = 0
    if direction_type == 'left':
        # 从做到右
        for col_id in range(0, width, col_step):
            white_pos = np.vstack(np.where(canny_bw_img[:, col_id] == 255))
            if white_pos[0].shape[0] == 0:
                continue
            if white_pos[0].shape[0] > (height // 10):
                roi_x = col_id
                break
    else:
        # 从右到左
        for col_id in range(width-1, -1, -col_step):
            white_pos = np.vstack(np.where(canny_bw_img[:, col_id] == 255))
            if white_pos[0].shape[0] == 0:
                continue
            if white_pos[0].shape[0] > (height // 10):
                roi_x = col_id
                break
    return roi_x


def crop_corner_roi_img(input_image, image_location_class, crop_size):
    ''' 说明：把图缩放到很小， 然后提Canny边缘，水平垂直投影找到边，估算roi的x,y坐标，根据这个crop出区域 '''
    canny_resize_ratio = 0.1

    # 对图像进行缩放 & Canny边缘提取
    resize_image = cv2.resize(input_image, (0, 0), fx=canny_resize_ratio, fy=canny_resize_ratio, interpolation=cv2.INTER_LINEAR)
    canny_bw_image = contour_detect(resize_image)

    # 通过水平 & 垂直投影大致估算出 Roi的角点
    roi_center_y = calcul_hori_pos(canny_bw_image, image_location_class, 3)
    roi_center_x = calcul_vert_pos(canny_bw_image, 3, 'right')

    # 将点换算到原始坐标系
    canny_dilate_ratio = 1 / canny_resize_ratio
    roi_center_x, roi_center_y = int(roi_center_x * canny_dilate_ratio), int(roi_center_y * canny_dilate_ratio)

    # 原图上crop出ROI图像区域，便于后面估计倒角
    crop_left_size, crop_top_size, crop_right_size, crop_bottom_size = crop_size[0], crop_size[1], crop_size[2], crop_size[3]
    height, width = input_image.shape[0], input_image.shape[1]
    crop_left = roi_center_x - crop_left_size
    crop_top = roi_center_y - crop_top_size
    crop_right = roi_center_x + crop_right_size
    crop_bottom = roi_center_y + crop_bottom_size
    if crop_left < 0:
        crop_left = 0
    if crop_top < 0:
        crop_top = 0
    if crop_right > width:
        crop_right = width
    if crop_bottom > height:
        crop_bottom = height

    # 获取 roi_rect & roi_image
    roi_rect = [crop_left, crop_top, crop_right-crop_left, crop_bottom-crop_top]
    roi_image = input_image[crop_top:crop_bottom, crop_left:crop_right]
    return roi_rect, roi_image


def get_corner_tradition(input_image, image_location_class, direction_type, product_id, product_type, dilate_ratio):
    """
    算法说明：
    1） 首先判断如果是左相机拍照， 先将左相机图像，统一转换成右相机，便于统一算法计算。
    2） 在原始尺寸图片上对图像进行超级缩放，Canny边缘，水平垂直投影到边缘，找到倒角区域的ROI: roi_image, roi_rect
    3） 把crop出来的图像的Canny进行水平垂直投影，找到真正的拟合点。
    4） 然后寻找待拟合三角形的 交点，以这个交点找到两条边的端点，根据水平边的起点作为外接矩形的起点，两条边到交点的距离作为宽高
       （这样定义外接矩形可能有问题，最好通过3个点直接绘制三角形, 但是直接绘制三角形显示不是非常美观）
    5） 无论上图或者下图，矩形起点都是左上角，下图的话也是先要翻转到上图，便于统一处理。
    ******* 重要： 函数中crop_left, crop_top, crop_right, crop_bottom是超参数，需要根据现场特定情况设定，但是两个直角边一定要大一些。
    参数：
    image_location_class是列表中其中一个元素： ['NO_OBJ', 'ALL', 'UPPER', 'BELOW', 'MIDDLE']
    """
    # 定义corner数据结构
    corner_measure_result_dict = {'upper_corner': [0, 0], 'below_corner': [0, 0], 'upper_crop_img': None,
                                 'below_crop_img': None, 'upper_rect': [], 'below_rect': []}
    image_height, image_width = input_image.shape[0], input_image.shape[1]
    cut_pt_y = image_height // 2

    # 如果是左相机，图像统一翻转到右侧
    if direction_type == 'left':
        input_image = np.flip(input_image, 1)

    # 上图
    # if product_id == 0 and (product_type == 'big' or product_type == 'middle'):
    if image_location_class == 'UPPER':
        crop_left, crop_top, crop_right, crop_bottom = 600, 80, 80, 600  # 根据定位到corner的粗定位点后，外扩参数
        crop_size = [crop_left, crop_top, crop_right, crop_bottom]
        roi_rect, roi_image = crop_corner_roi_img(input_image, image_location_class, crop_size)
        roi_canny_img = contour_detect(roi_image)
        # 在ROI获取"水平拟合线"段的内点列表
        horizontal_inlier_pt_list, _ = vertical_projection(roi_canny_img, 'right', 1, 30, L1_THRESH)
        # 在ROI获取"垂直拟合线"段的内点列表
        vertical_inlier_pt_list, _ = horizontal_projection(roi_canny_img, 'right', 1, 30, L1_THRESH)
        # 在ROI获取倒角外接矩形坐标
        rect_x, rect_y, rect_w, rect_h = find_triangle_2_corner_pt(vertical_inlier_pt_list, horizontal_inlier_pt_list, roi_image)
        # 在真实图像上计算倒角矩形[x, y, w, h]
        roi_offset_x, roi_offset_y = roi_rect[0], roi_rect[1]
        corner_pos_list = [[rect_x+roi_offset_x, rect_y+roi_offset_y, rect_w, rect_h]]

        # 绘制
        temp = input_image[0:cut_pt_y, :]
        crop_bgr_img, crop_rect = crop_and_draw_corner_tradition(temp, dilate_ratio, corner_pos_list, crop_corner_offset, 'upper')
        # 存储
        corner_measure_result_dict['upper_corner'] = [corner_pos_list[0][2], corner_pos_list[0][3]]
        corner_measure_result_dict['upper_crop_img'] = crop_bgr_img
        # 如果是左侧相机采集图像， 统一按照右侧处理
        if direction_type == 'left':
            corner_measure_result_dict['upper_rect'] = [image_width-crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]]
        else:
            corner_measure_result_dict['upper_rect'] = crop_rect
    # 下图
    # elif (product_id == 1 and product_type == 'middle') or product_id == 2:
    elif image_location_class == 'BELOW':
        crop_left, crop_top, crop_right, crop_bottom = 600, 600, 80, 80
        crop_size = [crop_left, crop_top, crop_right, crop_bottom]
        roi_rect, roi_image = crop_corner_roi_img(input_image, image_location_class, crop_size)
        flip_roi_image = np.flip(roi_image, 0)  # 反转ROI成上图，便于后续操作与上图一致
        flip_roi_canny_img = contour_detect(flip_roi_image)

        # 在ROI获取"水平拟合线"段的内点列表
        horizontal_inlier_pt_list, _ = vertical_projection(flip_roi_canny_img, 'right', 1, 20, L1_THRESH)
        # 在ROI获取"垂直拟合线"段的内点列表
        vertical_inlier_pt_list, _ = horizontal_projection(flip_roi_canny_img, 'right', 1, 20, L1_THRESH)
        # 在ROI获取倒角外接矩形坐标
        rect_x, rect_y, rect_w, rect_h = find_triangle_2_corner_pt(vertical_inlier_pt_list, horizontal_inlier_pt_list, flip_roi_image)
        # 在真实图像上计算倒角矩形[x, y, w, h](无论上图或下图， 倒角外界矩形都是以左上角为起点)
        roi_offset_x, roi_offset_y = roi_rect[0], roi_rect[1]
        corner_pos_list = [[rect_x + roi_offset_x, roi_rect[3]-rect_y-rect_h+roi_offset_y, rect_w, rect_h]]

        # 绘制
        crop_bgr_img, crop_rect = crop_and_draw_corner_tradition(input_image, dilate_ratio, corner_pos_list, crop_corner_offset, 'below')
        # 存储
        corner_measure_result_dict['below_corner'] = [corner_pos_list[0][2], corner_pos_list[0][3]]
        corner_measure_result_dict['below_crop_img'] = crop_bgr_img
        # 如果是左侧相机采集图像， 统一按照右侧处理
        if direction_type == 'left':
            corner_measure_result_dict['below_rect'] = [image_width - crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]]
        else:
            corner_measure_result_dict['below_rect'] = crop_rect

    # 上图 & 下图
    # elif product_id == 0 and product_type == 'small':
    elif image_location_class == 'ALL':
        ''' 上半张图'''
        crop_left, crop_top, crop_right, crop_bottom = 600, 80, 80, 600
        crop_size = [crop_left, crop_top, crop_right, crop_bottom]
        roi_rect, roi_image = crop_corner_roi_img(input_image, 'UPPER', crop_size)
        roi_canny_img = contour_detect(roi_image)
        # 在ROI获取"水平拟合线"段的内点列表
        horizontal_inlier_pt_list, _ = vertical_projection(roi_canny_img, 'right', 1, FIT_PT_NUM, L1_THRESH)
        # 在ROI获取"垂直拟合线"段的内点列表
        vertical_inlier_pt_list, _ = horizontal_projection(roi_canny_img, 'right', 1, FIT_PT_NUM, L1_THRESH)
        # 在ROI获取倒角外接矩形坐标
        rect_x, rect_y, rect_w, rect_h = find_triangle_2_corner_pt(vertical_inlier_pt_list, horizontal_inlier_pt_list,
                                                                   roi_image)
        # 在真实图像上计算倒角矩形[x, y, w, h]
        roi_offset_x, roi_offset_y = roi_rect[0], roi_rect[1]
        corner_pos_list = [[rect_x + roi_offset_x, rect_y + roi_offset_y, rect_w, rect_h]]

        # 绘制
        temp = input_image[0:cut_pt_y, :]
        crop_bgr_img, crop_rect = crop_and_draw_corner_tradition(temp, dilate_ratio, corner_pos_list, crop_corner_offset, 'upper')
        # 存储
        corner_measure_result_dict['upper_corner'] = [corner_pos_list[0][2], corner_pos_list[0][3]]
        corner_measure_result_dict['upper_crop_img'] = crop_bgr_img
        # 如果是左侧相机采集图像， 统一按照右侧处理
        if direction_type == 'left':
            corner_measure_result_dict['upper_rect'] = [image_width - crop_rect[0], crop_rect[1], crop_rect[2],
                                                        crop_rect[3]]
        else:
            corner_measure_result_dict['upper_rect'] = crop_rect

        ''' 下半张图'''
        crop_left, crop_top, crop_right, crop_bottom = 600, 600, 80, 80
        crop_size = [crop_left, crop_top, crop_right, crop_bottom]
        roi_rect, roi_image = crop_corner_roi_img(input_image, 'BELOW', crop_size)
        flip_roi_image = np.flip(roi_image, 0)  # 反转ROI成上图，便于后续操作与上图一致
        flip_roi_canny_img = contour_detect(flip_roi_image)

        # 在ROI获取"水平拟合线"段的内点列表
        horizontal_inlier_pt_list, _ = vertical_projection(flip_roi_canny_img, 'right', 1, FIT_PT_NUM, L1_THRESH)
        # 在ROI获取"垂直拟合线"段的内点列表
        vertical_inlier_pt_list, _ = horizontal_projection(flip_roi_canny_img, 'right', 1, FIT_PT_NUM, L1_THRESH)
        # 在ROI获取倒角外接矩形坐标
        rect_x, rect_y, rect_w, rect_h = find_triangle_2_corner_pt(vertical_inlier_pt_list, horizontal_inlier_pt_list, flip_roi_image)
        # 在真实图像上计算倒角矩形[x, y, w, h](无论上图或下图， 倒角外界矩形都是以左上角为起点)
        roi_offset_x, roi_offset_y = roi_rect[0], roi_rect[1]
        corner_pos_list = [[rect_x + roi_offset_x, roi_rect[3]-rect_y-rect_h+roi_offset_y, rect_w, rect_h]]

        # 绘制
        crop_bgr_img, crop_rect = crop_and_draw_corner_tradition(input_image, dilate_ratio, corner_pos_list, crop_corner_offset, 'below')
        # 存储
        corner_measure_result_dict['below_corner'] = [corner_pos_list[0][2], corner_pos_list[0][3]]
        corner_measure_result_dict['below_crop_img'] = crop_bgr_img
        # 如果是左侧相机采集图像， 统一按照右侧处理
        if direction_type == 'left':
            corner_measure_result_dict['below_rect'] = [image_width - crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]]
        else:
            corner_measure_result_dict['below_rect'] = crop_rect
    else:
        corner_measure_result_dict['upper_corner'] = [0, 0]
        corner_measure_result_dict['below_corner'] = [0, 0]
        corner_measure_result_dict['upper_crop_img'] = None
        corner_measure_result_dict['below_crop_img'] = None
        corner_measure_result_dict['upper_rect'] = []
        corner_measure_result_dict['below_rect'] = []
    return corner_measure_result_dict

