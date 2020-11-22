# -*- coding: utf-8 -*-
import base64

import cv2
from flask import Flask, jsonify, request
from new_face_recognition import main_image
import numpy as np
from werkzeug.serving import make_server

app = Flask(__name__)


def faceDR(base64_image, scale=1):
    imgData = base64.b64decode(base64_image)
    nparr = np.fromstring(imgData, np.uint8)
    img = cv2.imdecode(nparr, cv2.COLOR_BGR2RGB)
    faceinfo = main_image(img=img, scale=scale)
    return faceinfo


@app.route('/face', methods=['GET', 'POST'])
def post_Data():
    if request.method == 'POST':
        image = request.form['image']
        try:
            scale = int(request.form['scale'])
        except:
            scale = 1
        faceinfo = faceDR(image, scale)
        # print(faceinfo)
        recognize_info = {'faceinfo': faceinfo}
        return jsonify(recognize_info)
    else:
        return 'Hello Face'


if __name__ == '__main__':
    # app.run(debug=False, host='0.0.0.0', port=5001)
    # import os
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
    server = make_server('0.0.0.0', 5001, app)
    server.serve_forever()
