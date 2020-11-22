# encoding: utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import os
# from queue import Queue
import time

import cv2
import numpy as np
import pypinyin as pypinyin
from scipy import misc
import tensorflow as tf
from tensorflow.python.platform import gfile

import align.detect_face
import facenet


def load_and_align_data(img, image_size, margin):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    img_size = np.asarray(img.shape)[0:2]

    # bounding_boxes shape:(1,5)  type:np.ndarray
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    # 如果未发现目标 直接返回
    if len(bounding_boxes) < 1:
        return 0, 0, 0

    # 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    # det = np.squeeze(bounding_boxes[:,0:4])
    det = bounding_boxes

    det_temp = det

    det[:, 0] = np.maximum(det[:, 0], 0)
    det[:, 1] = np.maximum(det[:, 1], 0)
    det[:, 2] = np.minimum(det[:, 2], img_size[1] - 1)
    det[:, 3] = np.minimum(det[:, 3], img_size[0] - 1)

    det_temp[:, 0] = np.maximum(det_temp[:, 0] - margin / 2, 0)
    det_temp[:, 1] = np.maximum(det_temp[:, 1] - margin / 2, 0)
    det_temp[:, 2] = np.minimum(det_temp[:, 2] + margin / 2, img_size[1] - 1)
    det_temp[:, 3] = np.minimum(det_temp[:, 3] + margin / 2, img_size[0] - 1)
    det_temp = det_temp.astype(int)
    det = det.astype(int)
    crop = []
    for i in range(len(bounding_boxes)):
        w = abs(det[i, 0] - det[i, 2])
        h = abs(det[i, 1] - det[i, 3])
        if w > h:
            D = abs(w - h)
            newx1 = det[i, 0]
            newx2 = det[i, 2]
            newy1 = int(det[i, 1] - D / 2)
            newy2 = int(det[i, 3] + D / 2)
            if newy1 < 0:
                newy1 = 0
            if newy2 >= img.shape[0]:
                newy2 = img.shape[0] - 1
                # img.shape[0]：图像的垂直尺寸（高度）
                # img.shape[1]：图像的水平尺寸（宽度）
                # img.shape[2]：图像的通道数
        else:
            D = abs(w - h)
            newx1 = int(det[i, 0] - D / 2)
            newx2 = int(det[i, 2] + D / 2)
            newy1 = det[i, 1]
            newy2 = det[i, 3]
            if newx1 < 0:
                newx1 = 0
            if newx2 >= img.shape[1]:
                newx2 = img.shape[1] - 1
                # img.shape[0]：图像的垂直尺寸（高度）
                # img.shape[1]：图像的水平尺寸（宽度）
                # img.shape[2]：图像的通道数
        temp_crop = img[newy1:newy2, newx1:newx2, :]

        print(temp_crop.shape)

        # temp_crop = img[det[i, 1]:det[i, 3], det[i, 0]:det[i, 2], :]
        aligned = misc.imresize(temp_crop, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        crop.append(prewhitened)

    # np.stack 将crop由一维list变为二维
    crop_image = np.stack(crop)
    return 1, det_temp, crop_image


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
with gfile.FastGFile('./20180402-114759/20180402-114759.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    # with tf.Session(config=config) as sess:
    pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    # 创建load_and_align_data网络
    print('Creating networks and loading parameters')

    # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    image = []
    nrof_images = 0

    # 这里要改为自己emb_img文件夹的位置
    emb_dir = './train_dir/emb_img/'
    all_obj = []
    for i in os.listdir(emb_dir):
        all_obj.append(i)
        img = misc.imread(os.path.join(emb_dir, i), mode='RGB')
        prewhitened = facenet.prewhiten(img)
        image.append(prewhitened)
        # image.append(img)
        nrof_images = nrof_images + 1

    images = np.stack(image)
    # print(images)
    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
    compare_emb = sess.run(embeddings, feed_dict=feed_dict)


def main_image(img, scale=1):
    # 修改版load_and_align_data
    # 传入rgb np.ndarray
    compare_num = len(compare_emb)

    starttime = time.time()
    resize_img = cv2.resize(img, (int(img.shape[1] / scale), int(img.shape[0] / scale)))

    # 获取 判断标识 bounding_box crop_image
    mark, bounding_box, crop_image = load_and_align_data(resize_img, 160, 44)
    return_list = []
    if (mark):
        feed_dict = {images_placeholder: crop_image, phase_train_placeholder: False}
        emb = sess.run(embeddings, feed_dict=feed_dict)
        temp_num = len(emb)
        fin_obj = []
        # print(emb)
        # print(all_obj)
        # 为bounding_box 匹配标签
        for i in range(temp_num):
            dist_list = []
            for j in range(compare_num):
                dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], compare_emb[j, :]))))
                dist_list.append(dist)
            min_value = min(dist_list)
            print(dist_list)
            print(min_value)

            if (min_value > 0.85):
                fin_obj.append('unknow')
                base64_data = 'unknow'
                reg_img_url = ''
                bb = ''
            else:
                a = min_value - 0.5
                b = 1 - a
                if b > 1:
                    b = 1
                bb = '%.2f%%' % (b * 100)
                # print('相似度为:' + str(bb))
                fin_obj.append(all_obj[dist_list.index(min_value)].split('.')[0])
                with open("./train_dir/emb_img/" + fin_obj[i] + '.jpg', "rb") as f:  # 转为二进制格式
                    base64_data = str(base64.b64encode(f.read()))[2:-1]  # 使用base64进行加密
                    reg_img_url = os.getcwd() + '/train_dir/emb_img/' \
                                  + fin_obj[i] + '.jpg'
            recognize_info = {'x1': str(int(bounding_box[i, 0])),
                              'y1': str(int(bounding_box[i, 1])),
                              'x2': str(int(bounding_box[i, 2])),
                              'y2': str(int(bounding_box[i, 3])),
                              'name': fin_obj[i], 'reg_img': base64_data,
                              'reg_img_url': reg_img_url, 'confidence': str(bb)}
            return_list.append(recognize_info)
        print(fin_obj)
        print(len(fin_obj))
        # 在frame上绘制边框和文字
        # for rec_position in range(temp_num):
        #     if fin_obj[rec_position] != 'unknow':
        #         with open("./train_dir/emb_img/" + fin_obj[rec_position] + '.jpg', "rb") as f:  # 转为二进制格式
        #             base64_data = str(base64.b64encode(f.read()))[2:-1]  # 使用base64进行加密
        #             reg_img_url = os.getcwd() + '/train_dir/emb_img/' \
        #                           + fin_obj[rec_position] + '.jpg'
        #     else:
        #         base64_data = 'unknow'
        #         reg_img_url = ''
        #         # print(base64_data)
        #     recognize_info = {'x1': str(int(bounding_box[rec_position, 0])),
        #                       'y1': str(int(bounding_box[rec_position, 1])),
        #                       'x2': str(int(bounding_box[rec_position, 2])),
        #                       'y2': str(int(bounding_box[rec_position, 3])),
        #                       'name': fin_obj[rec_position], 'reg_img': base64_data,
        #                       'reg_img_url': reg_img_url}
        #     return_list.append(recognize_info)
    stoptime = time.time()
    print('usetime:', str(stoptime - starttime))
    return return_list


def pingyin(word):
    s = ''
    for i in pypinyin.pinyin(word, style=pypinyin.NORMAL):
        s += ''.join(i)
    return s
