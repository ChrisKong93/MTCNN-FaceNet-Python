# -*- coding: utf-8 -*-
import base64
import time

import RTSCapture
import cv2
import numpy as np
import pypinyin
import requests


def pingyin(word):
    s = ''
    for i in pypinyin.pinyin(word, style=pypinyin.NORMAL):
        s += ''.join(i)
    return s


def facerecognition(frame):
    image_0 = cv2.imencode('.jpg', frame)[1]
    base64_data_0 = str(base64.b64encode(image_0))[2:-1]
    # postdata = {'image': base64_data_0, 'scale': scale}
    postdata = {'image': base64_data_0}
    r = requests.post(url, data=postdata)
    print(r.json())
    faceinfo = r.json()['faceinfo']
    print(faceinfo)
    for i in range(len(faceinfo)):
        cv2.rectangle(frame, (int(faceinfo[i]['x1']), int(faceinfo[i]['y1'])),
                      (int(faceinfo[i]['x2']), int(faceinfo[i]['y2'])),
                      (0, 255, 0),
                      2, 8, 0)
        try:
            cv2.putText(
                frame,
                pingyin(faceinfo[i]['name']),
                (int(faceinfo[i]['x1']), int(faceinfo[i]['y1'])),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                0.8,
                (0, 0, 255),
                thickness=2,
                lineType=2)
        except IndexError as e:
            cv2.putText(
                frame,
                '',
                (int(faceinfo[i]['x1']), int(faceinfo[i]['y1'])),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                0.8,
                (0, 0, 255),
                thickness=2,
                lineType=2)
        # name = faceinfo[i]['name']
        # if name != '' and name != 'unknow':
        #     print(name)
        #     base64_image = faceinfo[i]['reg_img']
        #     print(base64_image)
        #     imgData = base64.b64decode(base64_image)
        #     nparr = np.fromstring(imgData, np.uint8)
        #     img = cv2.imdecode(nparr, cv2.COLOR_BGR2RGB)
        #     cv2.imshow("test", img)
    cv2.imshow("test", frame)


# url = 'http://127.0.0.1:5001/face'
url = 'http://192.168.8.114:5001/face'
if __name__ == '__main__':
    start = time.time()
    timer = 0
    timer_face = 0
    # rtscap = RTSCapture.RTSCapture.create('rtsp://admin:admin12345@192.168.1.12:554/live')
    rtscap = RTSCapture.RTSCapture.create(1)
    rtscap.start_read()  # 启动子线程并改变 read_latest_frame 的指向
    # capture = cv2.VideoCapture('rtsp://admin:admin@192.168.8.108:8554/live')
    # while capture.isOpened():
    #     ret, frame = capture.read()
    while rtscap.isStarted():
        start_0 = time.time()
        timer += 1
        # start = time.time()
        ret, frame = rtscap.read_latest_frame()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("esc break...")
            break
        if not ret:
            continue
        # 抽帧
        if int(time.time() - start) * 1000 % 10 == 0:
            timer_face += 1
            # scale = 4
            # resize_frame = cv2.resize(frame, (int(frame.shape[1] / scale), int(frame.shape[0] / scale)))
            facerecognition(frame)
        end = time.time()
        t = end - start_0
        print(t, timer, timer_face)
