import base64

import cv2
from flask import Flask, jsonify, request
import new_face_recognition_cam
import numpy as np

app = Flask(__name__)


def faceDR(base64_image):
    imgData = base64.b64decode(base64_image)
    nparr = np.fromstring(imgData, np.uint8)
    img = cv2.imdecode(nparr, cv2.COLOR_BGR2RGB)
    base64_data_1, name = new_face_recognition_cam.main_image(img=img, scale=2)
    return base64_data_1, name


@app.route('/face', methods=['POST'])
def post_Data():
    image = request.form['image']
    result_image, name = faceDR(image)
    recognize_info = {'image': result_image, 'name': name}
    return jsonify(recognize_info)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
