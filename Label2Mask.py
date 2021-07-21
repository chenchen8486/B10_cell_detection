# coding: utf-8
import os
import numpy as np
import cv2
import json
import time
import shutil
from utils import list_images




def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]  # 对x排序
    change = False
    # 如果右边的两个点的y值都比左边的小或者大,表明文本是竖着的且倾斜很厉害
    if (xSorted[2][1] < xSorted[0][1] and xSorted[2][1] < xSorted[1][1] \
            and xSorted[3][1] < xSorted[0][1] and xSorted[3][1] < xSorted[1][1]) or \
            (xSorted[2][1] > xSorted[0][1] and xSorted[2][1] > xSorted[1][1]
            and xSorted[3][1] > xSorted[0][1] and xSorted[3][1] > xSorted[1][1]):
        ySorted = pts[np.argsort(pts[:, 1]), :]  # 对x排序
        if abs(ySorted[2][1] - ySorted[1][1]) > abs(xSorted[2][0] - xSorted[1][0]):
            # 判断文本是竖着的
            change = True
    # xSorted[:, [1, 0]] = xSorted[:, [0, 1]]  # x,y对换
    leftMost = xSorted[:2, :]  # 前两名分到左边
    rightMost = xSorted[2:, :]  # 后两名分到右边
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]  # 对左边两个点的y进行排序
    # if leftMost[0][1] == leftMost[1][1]:
    if change:
        leftMost = leftMost[np.argsort(leftMost[:, 0]), :]  # 对x排序
        (bl, tl) = leftMost
    else:
        (tl, bl) = leftMost
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]  # 对右边两个点的y进行排序

    # if rightMost[0][1] == rightMost[1][1]:
    if change:
        rightMost = rightMost[np.argsort(rightMost[:, 0]), :]  # 对x排序
        (br, tr) = rightMost
    else:
        (tr, br) = rightMost
    return np.array([tl, tr, br, bl], dtype="int")


def delete_all_file(path):
    """
    删除某一目录下的所有文件或文件夹
    :param path:
    :return:
    """
    del_list = os.listdir(path)
    for f in del_list:
        file_path = os.path.join(path, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


class PolygonDraw:

    def getdrawpolygonvaliddata(self, polygons):
        if isinstance(polygons, list):
            for i, polygon in enumerate(polygons):
                polygon = np.array(polygon)

                if polygon.dtype in [np.float32, np.float64]:
                    polygons[i] = np.int32(polygon)

        return polygons

    def drawfillim(self, im, polygons, fillcolor):
        try:
            im = cv2.fillPoly(im, polygons, fillcolor)  # 只使用这个函数可能会出错，不知道为啥
        except:
            try:
                im = cv2.fillConvexPoly(im, polygons, fillcolor)
            except:
                print('cant fill\n')

        return im

    @classmethod
    def drawpolygonlinesim(cls,im, polygon0, linecolor,thickness=2):
        polygon = PolygonDraw().getdrawpolygonvaliddata(polygon0)
        cv2.polylines(im, polygon, True, linecolor, thickness)
        return im

    @classmethod
    def drawpolygonfillim(cls,im, polygon0, fillcolor):
        polygon = PolygonDraw().getdrawpolygonvaliddata(polygon0)
        im=PolygonDraw().drawfillim(im, polygon, fillcolor)
        return im

    @classmethod
    def drawpolygonbinim(cls,im, polygon0, fillcolor):
        imshape = im.shape
        binim = np.zeros(imshape, np.uint8)
        polygon = PolygonDraw().getdrawpolygonvaliddata(polygon0)

        binim= PolygonDraw().drawfillim(binim, polygon, fillcolor)

        return binim


def cut_same_json_and_image(data_path, save_path):
    for root, dirs, files in os.walk(data_path):
        for file in files:
            extension_name = os.path.splitext(file)[1]
            if extension_name == '.json':
                img_base_name = os.path.splitext(file)[0]

                src_img_path = root + '/' + img_base_name + '.bmp'
                dst_img_path = save_path + '/' + img_base_name + '.bmp'
                shutil.move(src_img_path, dst_img_path)

                src_json_path = root + '/' + file
                dst_json_path = save_path + '/' + file
                shutil.move(src_json_path, dst_json_path)
            print(file)
    return 0


if __name__ == '__main__':
    print('bingo...')
    # data_path = '/home/root0/workspace/python_project/B10_demo/data/train_data'
    # save_path = '/home/root0/workspace/python_project/B10_demo/data/label_data'
    # cut_same_json_and_image(data_path, save_path)


    # # label 2 mask code ...
    # image_path = '/home/root0/workspace/python_project/B10_demo/data/label_data'
    # json_path =  '/home/root0/workspace/python_project/B10_demo/data/label_data'
    #
    # training_data_path = '/home/root0/workspace/python_project/B10_demo/data/generate_gt'
    #
    # """删除旧的训练数据"""
    # delete_all_file(training_data_path)
    #
    # """labels"""
    # label_path = os.path.join(training_data_path, 'labels')
    # if not os.path.exists(label_path):
    #     os.makedirs(label_path)
    #
    # """ok and ng"""
    # ok_dir = os.path.join(training_data_path, 'ok')
    # if not os.path.exists(ok_dir):
    #     os.makedirs(ok_dir)
    #
    # ng_dir = os.path.join(training_data_path, 'ng')
    # if not os.path.exists(ng_dir):
    #     os.makedirs(ng_dir)
    #
    # """crop小图"""
    # crop_path = os.path.join(training_data_path, 'croped_image')
    # if not os.path.exists(crop_path):
    #     os.makedirs(crop_path)
    #
    # use_crop = False
    # img_list = list_images(image_path)
    # polygon_image = []
    # bad_image = []
    #
    # for i, img_path in enumerate(img_list):
    #     if i % 100 == 0:
    #         print('process %d \n' % i)
    #     image_base_name = os.path.basename(img_path)
    #     file_name = os.path.splitext(image_base_name)[0]
    #     json_name = file_name + '.json'
    #     image = cv2.imread(os.path.join(image_path, image_base_name))
    #
    #
    #     # 以下图像定义，有几个类别的mask 就定义几个变量
    #     type0_ng_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    #     type0_polygons = []
    #     type0_rects = []
    #
    #     type1_ng_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    #     type1_polygons = []
    #     type1_rects = []
    #
    #     type2_ng_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    #     type2_polygons = []
    #     type2_rects = []
    #
    #     type3_ng_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    #     type3_polygons = []
    #     type3_rects = []
    #
    #     type4_ng_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    #     type4_polygons = []
    #     type4_rects = []
    #
    #     type5_ng_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    #     type5_polygons = []
    #     type5_rects = []
    #
    #     type6_ng_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    #     type6_polygons = []
    #     type6_rects = []
    #
    #     if not os.path.exists(os.path.join(json_path, json_name)):
    #         shutil.copy(img_path, ok_dir)
    #         continue
    #
    #     json_data = json.loads(open(os.path.join(json_path, json_name)).read(), encoding='utf-8')
    #
    #     if len(json_data['shapes']) == 0:
    #         shutil.copy(img_path, ok_dir)
    #         continue
    #
    #
    #     for cell in json_data['shapes']:
    #         points = cell['points'][:]
    #         label = cell['label']
    #         if use_crop:
    #             points_flat = []
    #             if cell['shape_type'] == u'polygon' or cell['shape_type'] == u'rectangle':
    #                 for pt in points:
    #                     pt = map(int, pt)
    #                     points_flat.extend(pt)
    #                 np_box = np.array(points_flat)
    #                 np_box = np_box.reshape((len(points), 2))
    #                 left = np.min(np_box, axis=0)[0]
    #                 top = np.min(np_box, axis=0)[1]
    #                 right = np.max(np_box, axis=0)[0]
    #                 bottom = np.max(np_box, axis=0)[1]
    #
    #                 if left >= 0 and top >= 0 and right <= 768 and bottom <= 768 \
    #                         and bottom - top > 2 and right - left > 2:
    #                     croped_image = image[top: bottom, left: right, :]
    #                     img_new_name = str(time.time()) + '.jpg'
    #                     cv2.imwrite(os.path.join(crop_path, img_new_name), croped_image)
    #
    #         if cell['shape_type'] == u'polygon':
    #             if 'frontier' == label:  #
    #                 type0_polygons.append(points)
    #             if 'mark' == label:  #
    #                 type1_polygons.append(points)
    #             if 'corner' == label:  #
    #                 type2_polygons.append(points)
    #             if 'corner_dirty' == label:  #
    #                 type3_polygons.append(points)
    #             if 'dirty_defect' == label:  #
    #                 type4_polygons.append(points)
    #             if 'crack' == label:  #
    #                 type5_polygons.append(points)
    #             if 'bubble' == label:  #
    #                 type6_polygons.append(points)
    #         elif cell['shape_type'] == 'rectangle':
    #             if 'frontier' == label:
    #                 type0_rects.append(points)
    #             if 'mark' == label:
    #                 type1_rects.append(points)
    #             if 'corner' == label:
    #                 type2_rects.append(points)
    #             if 'corner_dirty' == label:
    #                 type3_rects.append(points)
    #             if 'dirty_defect' == label:
    #                 type4_rects.append(points)
    #             if 'crack' == label:
    #                 type5_rects.append(points)
    #             if 'bubble' == label:
    #                 type6_rects.append(points)
    #         else:
    #             bad_image.append(image_base_name)
    #             continue
    #
    #     if len(type0_polygons) > 0:
    #         type0_ng_mask = PolygonDraw.drawpolygonfillim(type0_ng_mask, type0_polygons, 255)
    #     if len(type0_rects) > 0:
    #         for points in type0_rects:
    #             points_flat = list()
    #             for pt in points:
    #                 pt = map(int, pt)
    #                 points_flat.extend(pt)
    #             np_box = np.array(points_flat)
    #             np_box = np_box.reshape((len(points), 2))
    #             left = np.min(np_box, axis=0)[0]
    #             top = np.min(np_box, axis=0)[1]
    #             right = np.max(np_box, axis=0)[0]
    #             bottom = np.max(np_box, axis=0)[1]
    #             cv2.rectangle(type0_ng_mask, (left, top), (right, bottom), color=255, thickness=-1)
    #
    #     if len(type1_polygons) > 0:
    #         type1_ng_mask = PolygonDraw.drawpolygonfillim(type1_ng_mask, type1_polygons, 255)
    #     if len(type1_rects) > 0:
    #         for points in type1_rects:
    #             points_flat = list()
    #             for pt in points:
    #                 pt = map(int, pt)
    #                 points_flat.extend(pt)
    #             np_box = np.array(points_flat)
    #             np_box = np_box.reshape((len(points), 2))
    #             left = np.min(np_box, axis=0)[0]
    #             top = np.min(np_box, axis=0)[1]
    #             right = np.max(np_box, axis=0)[0]
    #             bottom = np.max(np_box, axis=0)[1]
    #             cv2.rectangle(type1_ng_mask, (left, top), (right, bottom), color=255, thickness=-1)
    #
    #     if len(type2_polygons) > 0:
    #         type2_ng_mask = PolygonDraw.drawpolygonfillim(type2_ng_mask, type2_polygons, 255)
    #     if len(type2_rects) > 0:
    #         for points in type2_rects:
    #             points_flat = list()
    #             for pt in points:
    #                 pt = map(int, pt)
    #                 points_flat.extend(pt)
    #             np_box = np.array(points_flat)
    #             np_box = np_box.reshape((len(points), 2))
    #             left = np.min(np_box, axis=0)[0]
    #             top = np.min(np_box, axis=0)[1]
    #             right = np.max(np_box, axis=0)[0]
    #             bottom = np.max(np_box, axis=0)[1]
    #             cv2.rectangle(type2_ng_mask, (left, top), (right, bottom), color=255, thickness=-1)
    #
    #     if len(type3_polygons) > 0:
    #         type3_ng_mask = PolygonDraw.drawpolygonfillim(type3_ng_mask, type3_polygons, 255)
    #     if len(type3_rects) > 0:
    #         for points in type3_rects:
    #             points_flat = list()
    #             for pt in points:
    #                 pt = map(int, pt)
    #                 points_flat.extend(pt)
    #             np_box = np.array(points_flat)
    #             np_box = np_box.reshape((len(points), 2))
    #             left = np.min(np_box, axis=0)[0]
    #             top = np.min(np_box, axis=0)[1]
    #             right = np.max(np_box, axis=0)[0]
    #             bottom = np.max(np_box, axis=0)[1]
    #             cv2.rectangle(type3_ng_mask, (left, top), (right, bottom), color=255, thickness=-1)
    #
    #
    #     if len(type4_polygons) > 0:
    #         type4_ng_mask = PolygonDraw.drawpolygonfillim(type4_ng_mask, type4_polygons, 255)
    #     if len(type4_rects) > 0:
    #         for points in type4_rects:
    #             points_flat = list()
    #             for pt in points:
    #                 pt = map(int, pt)
    #                 points_flat.extend(pt)
    #             np_box = np.array(points_flat)
    #             np_box = np_box.reshape((len(points), 2))
    #             left = np.min(np_box, axis=0)[0]
    #             top = np.min(np_box, axis=0)[1]
    #             right = np.max(np_box, axis=0)[0]
    #             bottom = np.max(np_box, axis=0)[1]
    #             cv2.rectangle(type4_ng_mask, (left, top), (right, bottom), color=255, thickness=-1)
    #
    #
    #     if len(type5_polygons) > 0:
    #         type5_ng_mask = PolygonDraw.drawpolygonfillim(type5_ng_mask, type5_polygons, 255)
    #     if len(type5_rects) > 0:
    #         for points in type5_rects:
    #             points_flat = list()
    #             for pt in points:
    #                 pt = map(int, pt)
    #                 points_flat.extend(pt)
    #             np_box = np.array(points_flat)
    #             np_box = np_box.reshape((len(points), 2))
    #             left = np.min(np_box, axis=0)[0]
    #             top = np.min(np_box, axis=0)[1]
    #             right = np.max(np_box, axis=0)[0]
    #             bottom = np.max(np_box, axis=0)[1]
    #             cv2.rectangle(type5_ng_mask, (left, top), (right, bottom), color=255, thickness=-1)
    #
    #
    #     if len(type6_polygons) > 0:
    #         type6_ng_mask = PolygonDraw.drawpolygonfillim(type6_ng_mask, type6_polygons, 255)
    #     if len(type6_rects) > 0:
    #         for points in type6_rects:
    #             points_flat = list()
    #             for pt in points:
    #                 pt = map(int, pt)
    #                 points_flat.extend(pt)
    #             np_box = np.array(points_flat)
    #             np_box = np_box.reshape((len(points), 2))
    #             left = np.min(np_box, axis=0)[0]
    #             top = np.min(np_box, axis=0)[1]
    #             right = np.max(np_box, axis=0)[0]
    #             bottom = np.max(np_box, axis=0)[1]
    #             cv2.rectangle(type6_ng_mask, (left, top), (right, bottom), color=255, thickness=-1)
    #
    #
    #     shutil.copy(img_path, ng_dir)
    #     type0_mask_name = file_name + '_ng0_mask.png'
    #     type1_mask_name = file_name + '_ng1_mask.png'
    #     type2_mask_name = file_name + '_ng2_mask.png'
    #     type3_mask_name = file_name + '_ng3_mask.png'
    #     type4_mask_name = file_name + '_ng4_mask.png'
    #     type5_mask_name = file_name + '_ng5_mask.png'
    #     type6_mask_name = file_name + '_ng6_mask.png'
    #
    #
    #
    #     cv2.imwrite(os.path.join(label_path, type0_mask_name), type0_ng_mask)
    #     cv2.imwrite(os.path.join(label_path, type1_mask_name), type1_ng_mask)
    #     cv2.imwrite(os.path.join(label_path, type2_mask_name), type2_ng_mask)
    #     cv2.imwrite(os.path.join(label_path, type3_mask_name), type3_ng_mask)
    #     cv2.imwrite(os.path.join(label_path, type4_mask_name), type4_ng_mask)
    #     cv2.imwrite(os.path.join(label_path, type5_mask_name), type5_ng_mask)
    #     cv2.imwrite(os.path.join(label_path, type6_mask_name), type6_ng_mask)
    #
    #
    #     if len(bad_image) > 0:
    #         print('/////////////////////////////////')
    #         for cell in bad_image:
    #             print(cell)
    #     else:
    #         pass


