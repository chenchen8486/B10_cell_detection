# coding:utf-8
import os
import time
import cv2
import json
import tensorflow as tf
import numpy as np
from builders import model_builder
from utils import sigmoid
# from log import ALL_LOG_OBJ

"""参数配置"""
config_path = './config.json'
config_data = json.loads(open(config_path).read(), encoding='utf-8')
BATCH_SIZE = config_data['BATCH_SIZE']
os.environ['CUDA_VISIBLE_DEVICES'] = config_data['CUDA_VISIBLE_DEVICES']
NUM_CLASSES = config_data['NUM_CLASSES']
GPU_MEMORY_FRACTION = config_data['GPU_MEMORY_FRACTION']
SCORE_MAP_THRESH = config_data['SCORE_MAP_THRESH']
MODEL_NAME = config_data['MODEL_NAME']
FRONTEND = config_data['FRONTEND']
CHECKPOINT_DIR = config_data['CHECKPOINT_DIR']
SUB_IMAGE_SIZE = config_data['SUB_IMAGE_SIZE']


class SemanticSegment(object):
  
    def __init__(self):
        config = tf.ConfigProto()

        # 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放
        # 内存，所以会导致碎片
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = GPU_MEMORY_FRACTION
        self.num_classes = NUM_CLASSES
        tf.reset_default_graph()
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())
        with self.session.as_default():
            self.input_image = tf.placeholder(tf.float32, shape=[None, SUB_IMAGE_SIZE, SUB_IMAGE_SIZE, 3])

            self.prediction, _ = model_builder.build_model(model_name=MODEL_NAME,
                                                           frontend=FRONTEND,
                                                           net_input=self.input_image,
                                                           num_classes=NUM_CLASSES,
                                                           crop_width=SUB_IMAGE_SIZE,
                                                           crop_height=SUB_IMAGE_SIZE)
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
            if ckpt is not None:
                saver.restore(self.session, ckpt)
            zero_image = np.zeros([1, SUB_IMAGE_SIZE, SUB_IMAGE_SIZE, 3], dtype=np.uint8)
            self.session.run(self.prediction, feed_dict={self.input_image: zero_image})
            # ALL_LOG_OBJ.logger.info('Init model finished!')

    def predict_multi_label(self, image, label_name_list):
        im_in = image
        im_in = np.float32(im_in) / 255.0
        if len(im_in.shape) < 3:
            im_in = np.expand_dims(im_in, axis=2)

        prediction = self.session.run(self.prediction, feed_dict={self.input_image: im_in})
        batch_result = []
        for pred_id, pred_result in enumerate(prediction):
            each_prediction_result = {}
            for class_id in range(self.num_classes):  # 有多少个类别,就循环多少次进行存储, 每个图像存储成一个字典，
                # 获取score map(单张图像的每个类别的score map)
                pred = (pred_result[:, :, class_id] * 255).astype(np.uint8)
                # cv2.imwrite("./result/aaa.png", pred)
                # cv2.imshow("test", pred)
                # cv2.waitKey()
                # 获取pred_mask(单张图像的每个类别的pred_mask)
                _, pred_mask = cv2.threshold(pred, SCORE_MAP_THRESH, 255, cv2.THRESH_BINARY)
                # 结果数据存储
                each_prediction_result[label_name_list[class_id]] = pred_mask.astype(np.uint8)
            batch_result.append(each_prediction_result)   # 最后把字典存到List中
        return batch_result



