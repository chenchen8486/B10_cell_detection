#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import os, cv2, time, base64, json, _thread
import numpy as np
import traceback
from log import ALL_LOG_OBJ
from flask import Flask, request, jsonify
from algo_module import qr_detection
app = Flask(__name__)

"""参数配置"""
config_path = './config.json'
config_data = json.loads(open(config_path).read(), encoding='utf-8')


def bytes2cv(data):
    nparr = np.fromstring(data, np.uint8)
    img_decode = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return img_decode


@app.route('/algorithm/api/crop_image', methods=['POST'])
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
        image_base64 = data["image_base64"]

        # 3）如果调用base64的方式进行数据传递
        if image_base64 != 'None':
            # 将C++发送的base64数据留解码成二进制格式
            binary_data = base64.b64decode(image_base64)
            # 把二进制格式数据流转换成图像数据
            cv_image = bytes2cv(binary_data)

        # 4）调用检测函数，并返回C++需要的输出结果
        qr_rect = qr_detection(cv_image)
        response_data = {'qr_rect_list': qr_rect}

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