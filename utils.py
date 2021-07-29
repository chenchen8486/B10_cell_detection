# coding: utf-8
import os
import cv2
import numpy as np
import json

"""参数配置"""
config_path = './config.json'
config_data = json.loads(open(config_path).read(), encoding='utf-8')
SUB_IMAGE_SIZE = config_data['SUB_IMAGE_SIZE']
MERGE_DIS_THRE = config_data['MERGE_DIS_THRE']


def list_images(path, file_type='images'):
    """
    列出文件夹中所有的文件，返回
    :param file_type: 'images' or 'any'
    :param path: a directory path, like '../data/pics'
    :return: all the images in the directory
    """
    IMAGE_SUFFIX = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG',
                    '.tiff', '.TIFF', '.bmp', '.BMP', '.tif', '.TIF']
    paths = []
    for file_and_dir in os.listdir(path):
        if os.path.isfile(os.path.join(path, file_and_dir)):
            if file_type == 'images':
                if os.path.splitext(file_and_dir)[1] in IMAGE_SUFFIX:
                    paths.append(os.path.abspath(os.path.join(path,
                                                              file_and_dir)))
            elif file_type == 'any':
                paths.append(os.path.abspath(os.path.join(path, file_and_dir)))
            else:
                if os.path.splitext(file_and_dir)[1] == file_type:
                    paths.append(os.path.abspath(os.path.join(path,
                                                              file_and_dir)))
    return paths


def resize_image(im, side_len=SUB_IMAGE_SIZE):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    im = cv2.resize(im, (side_len, side_len))

    ratio_h = side_len / float(h)
    ratio_w = side_len / float(w)

    return im, (ratio_h, ratio_w)


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def is_overlap(rect1, rect2):
    '''
    计算两个矩形的交并比
    :param rect1:第一个矩形框。表示为x,y,w,h，其中x,y表示矩形右上角的坐标
    :param rect2:第二个矩形框。
    :return:是否有重叠部分
    '''
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    inter_w = (w1 + w2) - (max(x1 + w1, x2 + w2) - min(x1, x2))
    inter_h = (h1 + h2) - (max(y1 + h1, y2 + h2) - min(y1, y2))

    if inter_h <= 0 or inter_w <= 0:  # 代表相交区域面积为0
        return False
    else:
        return True


def need_merge(rect1, rect2, thre=MERGE_DIS_THRE):
    '''
    根据两个框的曼哈顿距离判断是否需要合并
    :param rect1:第一个矩形框。表示为x,y,w,h，其中x,y表示矩形右上角的坐标
    :param rect2:第二个矩形框。
    :return:是否需要合并
    '''
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    inter_w = (w1 + w2) - (max(x1 + w1, x2 + w2) - min(x1, x2))
    inter_h = (h1 + h2) - (max(y1 + h1, y2 + h2) - min(y1, y2))

    v_dis = abs(min((y1 + h1), (y2 + h2)) - max(y1, y2))
    h_dis = abs(min((x1 + w1), (x2 + w2)) - max(x1, x2))
    if v_dis + h_dis < thre or (v_dis < 0.5 * thre and inter_w > 0) or \
            (h_dis < 0.5 * thre and inter_h > 0) or is_overlap(rect1, rect2):
        return True
    elif (x1 > x2) and ((x1+w1) < (x2+w2)) and (y1 > y2) and ((y1+h1) < (y2+h2)):
        # rect1 在 rect2 内部
        return True
    elif (x2 > x1) and ((x2+w2) < (x1+w1)) and (y2 > y1) and ((y2+h2) < (y1+h1)):
        # rect2 在 rect1 内部
        return True
    else:
        return False


def save_crop_image(input_image, batch_list, save_path, image_name):
    for batch_id, rect in enumerate(batch_list):
        start_x = rect[0]
        start_y = rect[1]
        end_x = rect[0] + rect[2]
        end_y = rect[1] + rect[3]
        batch_image = input_image[start_y:end_y, start_x:end_x]
        batch_save_path = os.path.join(save_path, str(start_x) + '_' + str(start_y) + '_' +image_name)
        cv2.imwrite(batch_save_path, batch_image)
    return 0


def crop_right_camera_overlap_image(input_image, batch_w, batch_h, overlap_x=256, overlap_y=50):
    height, width = input_image.shape[0], input_image.shape[1]
    crop_rect_list = []
    row_id, col_id = 0, 0
    cut_img_w_th = int(batch_w / 3)
    while (row_id + batch_h) <= height:
        # 如果是第一次对行进行循环
        if row_id == 0:
            batch_start_y = 0
        else:
            batch_start_y = batch_start_y + batch_h - overlap_y
        while (col_id + batch_w) <= width:
            # 如果是第一次对列进行循环
            if col_id == 0:
                batch_start_x = 0
            else:
                batch_start_x = batch_start_x + batch_w - overlap_x  # 对列的起点坐标进行累加，便于存储
            batch = (batch_start_x, batch_start_y, batch_w, batch_h)
            crop_rect_list.append(batch)
            # 对列的起点ID进行累加，便于下一次循环
            col_id = col_id + batch_w - overlap_x
        # 如果当前列到宽的距离 还比阈值大，如果overlap参数存在的话， 证明还可以在横向crop出一个batch.
        if abs(width - col_id) > cut_img_w_th:
            batch_start_x = width - batch_w
            batch = (batch_start_x, batch_start_y, batch_w, batch_h)
            crop_rect_list.append(batch)
            row_id = row_id + batch_h - overlap_y
            col_id = 0
        # 否则，直接换行进行下一行进行循环
        else:
            row_id = row_id + batch_h - overlap_y
            col_id = 0

    # 为了对最后一行进行crop的补救
    while (col_id + batch_w) <= width:
        if col_id == 0:
            batch_start_x = 0
        else:
            batch_start_x = batch_start_x + batch_w - overlap_x
        batch_start_y = height - batch_h
        batch = (batch_start_x, batch_start_y, batch_w, batch_h)
        crop_rect_list.append(batch)
        col_id = col_id + batch_w - overlap_x
    if abs(width - col_id) > cut_img_w_th:
        batch_start_x = width - batch_w
        batch_start_y = height - batch_h
        batch = (batch_start_x, batch_start_y, batch_w, batch_h)
        crop_rect_list.append(batch)

    return crop_rect_list


def crop_overlap_image(input_image, direction_type, batch_w, batch_h, overlap_x=256, overlap_y=50):
    crop_image_list = []
    if direction_type == 'right':
        crop_rect_list = crop_right_camera_overlap_image(input_image, batch_w, batch_h, overlap_x, overlap_y)
    else:
        cpy_input_image = np.copy(input_image)
        flip_input_image = np.flip(cpy_input_image, 0)
        crop_rect_list = crop_right_camera_overlap_image(flip_input_image, batch_w, batch_h, overlap_x, overlap_y)
        crop_rect_list = [[flip_input_image.shape[1] - rr[0] - rr[2], rr[1], rr[2], rr[3]] for rr in crop_rect_list]

    for batch_id, rect in enumerate(crop_rect_list):
        start_x = rect[0]
        start_y = rect[1]
        end_x = rect[0] + rect[2]
        end_y = rect[1] + rect[3]
        crop_image = input_image[start_y:end_y, start_x:end_x]
        crop_image = cv2.cvtColor(crop_image, cv2.COLOR_GRAY2BGR)
        crop_image_list.append(crop_image)
    return crop_rect_list, crop_image_list


def findIntersection(horizontal_start_pt, horizontal_end_pt, vertical_start_pt, vertical_end_pt):
    """
    说明：寻找两条直线的相交点，输入参数就是两条直线的两个端点。
    """
    x1, y1, x2, y2 = horizontal_start_pt[0], horizontal_start_pt[1], horizontal_end_pt[0], horizontal_end_pt[1]
    x3, y3, x4, y4 = vertical_start_pt[0], vertical_start_pt[1], vertical_end_pt[0], vertical_end_pt[1]
    px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
    py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
    cross_pt = [int(px), int(py)]
    return cross_pt


def contour_detect(input_image):
    """
    说明：找到图像的轮廓区域，返回一个二值化图， 255:白色像素, 0:黑色像素
    input_image: Rgb or gray都可以
    """
    # input image is gray image
    edges = cv2.Canny(input_image, 20, 50)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # open = cv2.dilate(edges, kernel)
    return edges