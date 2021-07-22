#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import os, cv2, time, base64, json, _thread
import numpy as np
import traceback
from log import ALL_LOG_OBJ
from flask import Flask, request, jsonify
from algo_module import defect_detection
app = Flask(__name__)

"""参数配置"""
config_path = './config.json'
config_data = json.loads(open(config_path).read(), encoding='utf-8')
SAVE_DETECTION_RESULT_PATH = config_data['SAVE_DETECTION_RESULT_PATH']
CROP_DEFECT_IMAGE_OFFSET = config_data['CROP_DEFECT_IMAGE_OFFSET']
DEFECT_WH = config_data['DEFECT_WH']
DEFECT_TEXT_REGION_HEIGHT = config_data['DEFECT_TEXT_REGION_HEIGHT']
CAMERA_RESOLUTION = config_data['CAMERA_RESOLUTION']
DEFECT_NAME_LIST = config_data['DEFECT_NAME_LIST']
def bytes2cv(data):
    nparr = np.fromstring(data, np.uint8)
    img_decode = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return img_decode


def save_detection_result(image_name,
                          cv_image,
                          image_resize_ratio,
                          mark_rslt_list,
                          corner_rslt_dict,
                          frontier_rslt_dict,
                          defect_rslt_list,
                          draw_defect_panel):
    # 获取当前图像名称（无扩展名）
    sub_img_name = os.path.splitext(image_name)[0]

    # 构建指定路径的文件夹名称
    localtime = time.localtime(time.time())
    year, month, day = localtime[0], localtime[1], localtime[2]
    date_name = str(year) + "_" + str(month) + "_" + str(day)
    save_file_name_path = SAVE_DETECTION_RESULT_PATH + "/" + date_name
    save_corner_file_name_path = save_file_name_path + "/" + "corner" + "/" + sub_img_name
    save_frontier_file_name_path = save_file_name_path + "/" + "frontier" + "/" + sub_img_name
    save_mark_file_name_path = save_file_name_path + "/" + "mark" + "/" + sub_img_name
    save_defect_file_name_path = save_file_name_path + "/" + "defect" + "/" + sub_img_name
    save_whole_result_name_path = save_file_name_path + "/" + "all" + "/" + sub_img_name

    # 如果不存在指定存储文件的文件夹，那就创建
    if os.path.exists(save_corner_file_name_path) is not True:
        os.makedirs(save_corner_file_name_path)
    if os.path.exists(save_frontier_file_name_path) is not True:
        os.makedirs(save_frontier_file_name_path)
    if os.path.exists(save_mark_file_name_path) is not True:
        os.makedirs(save_mark_file_name_path)
    if os.path.exists(save_defect_file_name_path) is not True:
        os.makedirs(save_defect_file_name_path)
    if os.path.exists(save_whole_result_name_path) is not True:
        os.makedirs(save_whole_result_name_path)

    # 存储all 结果
    img_save_name = "/" + sub_img_name + "_all_defect.jpg"
    cv2.imwrite(save_whole_result_name_path + img_save_name, draw_defect_panel)

    # 存储corner结果
    if corner_rslt_dict['upper_crop_img'] is not None:
        img_save_name = "/" + sub_img_name + "_upper_corner.jpg"
        cv2.imwrite(save_corner_file_name_path + img_save_name, corner_rslt_dict['upper_crop_img'])
        del corner_rslt_dict['upper_crop_img']
    if corner_rslt_dict['below_crop_img'] is not None:
        img_save_name = "/" + sub_img_name + "_below_corner.jpg"
        cv2.imwrite(save_corner_file_name_path + img_save_name, corner_rslt_dict['below_crop_img'])
        del corner_rslt_dict['below_crop_img']

    # 存储frontier结果
    if frontier_rslt_dict['crop_img1'] is not None:
        img_save_name = "/" + sub_img_name + "_frontier1.jpg"
        cv2.imwrite(save_frontier_file_name_path + img_save_name, frontier_rslt_dict['crop_img1'])
        del frontier_rslt_dict['crop_img1']
    if frontier_rslt_dict['crop_img2'] is not None:
        img_save_name = "/" + sub_img_name + "_frontier2.jpg"
        cv2.imwrite(save_frontier_file_name_path + img_save_name, frontier_rslt_dict['crop_img2'])
        del frontier_rslt_dict['crop_img2']
    if frontier_rslt_dict['crop_img3'] is not None:
        img_save_name = "/" + sub_img_name + "_frontier3.jpg"
        cv2.imwrite(save_frontier_file_name_path + img_save_name, frontier_rslt_dict['crop_img3'])
        del frontier_rslt_dict['crop_img3']

    # 存储mark结果
    if mark_rslt_list['upper_crop_img'] is not None:
        img_save_name = "/" + sub_img_name + "_upper_mark.jpg"
        cv2.imwrite(save_mark_file_name_path + img_save_name, mark_rslt_list['upper_crop_img'])
        del mark_rslt_list['upper_crop_img']
    if mark_rslt_list['below_crop_img'] is not None:
        img_save_name = "/" + sub_img_name + "_below_mark.jpg"
        cv2.imwrite(save_mark_file_name_path + img_save_name, mark_rslt_list['below_crop_img'])
        del mark_rslt_list['below_crop_img']

    # 存储缺陷结果
    dilate_ratio = 1 / image_resize_ratio
    if len(defect_rslt_list) != 0:
        for defect_id, defect in enumerate(defect_rslt_list):
            # 1) 定义crop的具体坐标，
            crop_x = defect['box_left'] - CROP_DEFECT_IMAGE_OFFSET
            if crop_x <= 0:
                crop_x = 0
            crop_y = defect['box_top'] - CROP_DEFECT_IMAGE_OFFSET
            if crop_y <= 0:
                crop_y = 0
            crop_right = defect['box_right'] + CROP_DEFECT_IMAGE_OFFSET
            if crop_right > cv_image.shape[1]:
                crop_right = cv_image.shape[1]
            crop_bottom = defect['box_bottom'] + CROP_DEFECT_IMAGE_OFFSET
            if crop_bottom > cv_image.shape[0]:
                crop_bottom = cv_image.shape[0]
            # 2）将原始缺陷的外扩后矩形crop出来，然后转换成BGR
            read_defect_rect = cv_image[defect['box_top']:defect['box_bottom'], defect['box_left']:defect['box_right']]
            defect_std_val = np.std(read_defect_rect, ddof=1)
            if defect_std_val < 5:  # 如果缺陷区域标准差很小， 证明是误报
                continue

            defect_crop = cv_image[crop_y:crop_bottom, crop_x:crop_right]
            defect_crop_bgr = cv2.cvtColor(defect_crop, cv2.COLOR_GRAY2BGR)
            # 3）计算缺陷框框在crop图像上的相对坐标
            relative_x = defect['box_left'] - crop_x
            relative_y = defect['box_top'] - crop_y
            relative_right = relative_x + (defect['box_right'] - defect['box_left'])
            relative_bottom = relative_y + (defect['box_bottom'] - defect['box_top'])
            # 4）绘制rect框框
            cv2.rectangle(defect_crop_bgr, (relative_x, relative_y), (relative_right, relative_bottom), (0, 0, 255), 1)
            # 5) 然后缩放到指定尺寸
            defect_crop_resize = cv2.resize(defect_crop_bgr, (DEFECT_WH, DEFECT_WH),  interpolation=cv2.INTER_LINEAR)
            # 6）创建一个带有横幅的图像buffer， defect_text_region_height就是这个横幅的高度
            draw_panel = np.zeros((DEFECT_WH + DEFECT_TEXT_REGION_HEIGHT, DEFECT_WH, 3), np.uint8)
            # 7）将图像拷贝到draw_panel里面
            draw_panel[DEFECT_TEXT_REGION_HEIGHT:, :, :] = defect_crop_resize
            # 8）写下标题文字
            cv2.putText(draw_panel, "Defect cls: " + str(DEFECT_NAME_LIST[defect['type'] - 1]),
                        (5, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(draw_panel, "Defect size: (W: " + str(defect['box_width']*CAMERA_RESOLUTION*dilate_ratio) + "um,   H: " + str(defect['box_height']*CAMERA_RESOLUTION*dilate_ratio) + "um)",
                        (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            img_save_name = "/" + sub_img_name + "_defect_" + str(defect_id) + "_" + str(defect['type']) + ".jpg"
            cv2.imwrite(save_defect_file_name_path + img_save_name, draw_panel)
    return 0


@app.route('/algorithm/api/cell_detection', methods=['POST'])
def parser():
    """
    python 启动http服务，准备等待C++被调用，发送数据......
    """
    code = 0  # 错误码
    start_time = time.time()
    try:
        # 1）获取C++发送的数据包data
        data = request.get_data()
        if isinstance(data, bytes):
            data = str(data, encoding="utf-8")
        data = json.loads(data, strict=False)

        # 2）解析data数据中各个关键数据
        image_name = data["image_name"]  # 30000B000C0_0_defectClass.bmp
        split_info_list = image_name.split('_')  # 把图像的主名称按照下划线拆解成元素列表
        zm_of_fm = split_info_list[1]             # 'ZM' or 'FM'
        long_or_short = split_info_list[2]        # 'L' or 'S'
        left_or_right = split_info_list[3]        # 'L' or 'R'
        image_id = int(split_info_list[4][0])     # 0 or 1 or 2

        image_base64 = data["image_base64"]
        image_resize_ratio = float(data["image_resize_ratio"])  # 0.5
        image_type = data["image_type"]  # big, middle, small
        save_detect_image = int(data["save_detect_image"])

        # 3）如果调用base64的方式进行数据传递
        if image_base64 != 'None':
            # 将C++发送的base64数据留解码成二进制格式
            binary_data = base64.b64decode(image_base64)
            # 把二进制格式数据流转换成图像数据
            cv_image = bytes2cv(binary_data)

        # 4）调用检测函数，并返回C++需要的输出结果
        image_info_dict = {'image_name': image_name, 'zm_or_fm': zm_of_fm, 'long_or_short': long_or_short, 'left_or_right': left_or_right,
                           'resize_ratio': image_resize_ratio, 'image_type': image_type, 'image_id': image_id}
        mark_rslt_list, corner_rslt_dict, frontier_rslt_dict, defect_rslt_list, draw_defect_panel = defect_detection(
                                                                                                cv_image, image_info_dict)
        # 获取倒角宽高值
        corner_rslt_dict['upper_corner_x'] = corner_rslt_dict['upper_corner'][0]
        corner_rslt_dict['upper_corner_y'] = corner_rslt_dict['upper_corner'][1]
        corner_rslt_dict['below_corner_x'] = corner_rslt_dict['below_corner'][0]
        corner_rslt_dict['below_corner_y'] = corner_rslt_dict['below_corner'][1]
        # 获取左右相机类型
        direction_type = 0 if left_or_right == 'L' else 1

        # 5）存储各类图像结果，如果存储就存储到指定路径下，否则就将图像数据删除
        if save_detect_image:
            save_detection_result(image_name, cv_image, image_resize_ratio, mark_rslt_list, corner_rslt_dict,
                                  frontier_rslt_dict, defect_rslt_list, draw_defect_panel)
        else:
            if corner_rslt_dict['upper_crop_img'] is not None:
                del corner_rslt_dict['upper_crop_img']
            if corner_rslt_dict['below_crop_img'] is not None:
                del corner_rslt_dict['below_crop_img']
            if frontier_rslt_dict['crop_img1'] is not None:
                del frontier_rslt_dict['crop_img1']
            if frontier_rslt_dict['crop_img2'] is not None:
                del frontier_rslt_dict['crop_img2']
            if frontier_rslt_dict['crop_img3'] is not None:
                del frontier_rslt_dict['crop_img3']
            if mark_rslt_list['upper_crop_img'] is not None:
                del mark_rslt_list['upper_crop_img']
            if mark_rslt_list['below_crop_img'] is not None:
                del mark_rslt_list['below_crop_img']

        # 6）把最终检测结果打包成字典，进行返回
        response_data = {'direction_type': direction_type, 'image_id': image_id,
                         'corner_rslt_dict': corner_rslt_dict, 'frontier_rslt_dict':  frontier_rslt_dict,
                         'defect_rslt_list': defect_rslt_list, 'defect_num': len(defect_rslt_list)}
    # except Exception as e:
    #     ALL_LOG_OBJ.logger.info(" !!!! HTTP SERVICE CALL FAILED !!!!")
    #     code = 101
    #     response_data = {}
    except:
        ALL_LOG_OBJ.logger.info(" !!!! HTTP SERVICE CALL FAILED !!!!")
        traceback.print_exc()
        code = 101
        response_data = {}
    end_time = time.time()
    total_time = (end_time - start_time) * 1000
    ALL_LOG_OBJ.logger.info(" THE SERVICE FINISHED, RUN TIME IS:  %s\n\n" % total_time)
    return jsonify({
        'code': code,
        'data': response_data
    })


if __name__ == '__main__':
    host = '127.0.0.1'
    port = 12345
    app.run(host=host, port=port, threaded=True)