import base64
import time

import cv2
import numpy as np
import pyttsx3

from facenet_facerecognition import RTSCapture
import new_face_recognition_cam


def read_name(name):
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate + 50)
    engine.say(name)
    engine.runAndWait()


def video(video=0):
    # video = 'rtsp://admin:admin@192.168.8.108:8554/live'
    rtscap = RTSCapture.RTSCapture.create(video)
    rtscap.start_read()  # 启动子线程并改变 read_latest_frame 的指向
    start = time.time()
    # capture = cv2.VideoCapture(0)
    # while capture.isOpened():
    #     ret, frame = capture.read()
    while rtscap.isStarted():
        ret, frame = rtscap.read_latest_frame()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("esc break...")
            break
        if not ret:
            continue
        # image_0 = cv2.imencode('.jpg', frame)[1]
        # base64_data_0 = str(base64.b64encode(image_0))[2:-1]
        # imgData = base64.b64decode(base64_data_0)
        # nparr = np.fromstring(imgData, np.uint8)
        # img = cv2.imdecode(nparr, cv2.COLOR_BGR2RGB)
        base64_data_1, name = new_face_recognition_cam.main_image(img=frame, scale=2)

        image = base64.b64decode(base64_data_1)

        nparr = np.fromstring(image, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imshow("test", img_np)
        end = time.time()
        t = int(end - start)
        # if name != '' or name != 'unknow':
        #     # and t % 5 == 0:
        #     read_name(name=name)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video()