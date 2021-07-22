# coding: utf-8
import os
import requests
import threading
import time
import base64
import cv2


if __name__ == '__main__':
    print('bingo...')
    # image_file_path = "D:/4_data/B10_cell_data/test_http/1_big/2_below"
    image_file_path = "D:/4_data/B10_cell_data/test_http/temp"
    # 定义IP地址和端口号
    test_host, test_port = "127.0.0.1", "12345"
    request_url = "http://{}:{}/algorithm/api/cell_detection".format(test_host, test_port)
    use_base64 = True
    image_name_list = os.listdir(image_file_path)
    for id, image_name in enumerate(image_name_list):
        image_file_name = os.path.join(image_file_path, image_name)
        print(image_file_name)
        if use_base64:
            start2 = time.time()
            f = open(image_file_name, 'rb')
            image_data = f.read()
            f.close()

            # 定义要发送的数据包的数据结构
            start2 = time.time()
            req_json = {
                "image_name": image_name,  # image_name: panelID_ZM_L_S_0.bmp,
                "image_base64": base64.b64encode(image_data),
                "image_resize_ratio": 0.5,
                "image_type": "big",
                "save_detect_image": True
            }

        else:
            req_json = {
                "image_path": image_file_name,
                "image_base64": 'None'
            }

        response_return = requests.post(request_url, json=req_json).json()["data"]
        end2 = time.time()
        total2 = (end2 - start2) * 1000
        print("______the Base64 encode time is: ", total2)

