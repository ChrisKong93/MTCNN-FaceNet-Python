# encoding: utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import os
# from queue import Queue
import time

import align.detect_face
import cv2
import facenet
import numpy as np
from scipy import misc
import tensorflow as tf


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

    det[:, 0] = np.maximum(det[:, 0] - margin / 2, 0)
    det[:, 1] = np.maximum(det[:, 1] - margin / 2, 0)
    det[:, 2] = np.minimum(det[:, 2] + margin / 2, img_size[1] - 1)
    det[:, 3] = np.minimum(det[:, 3] + margin / 2, img_size[0] - 1)

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
            if newy2 > img.shape[0]:
                newy2 = img.shape[0]
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
            if newx2 > img.shape[1]:
                newx2 = img.shape[1]
                # img.shape[0]：图像的垂直尺寸（高度）
                # img.shape[1]：图像的水平尺寸（宽度）
                # img.shape[2]：图像的通道数
        temp_crop = img[newy1:newy2, newx1:newx2, :]

        # temp_crop = img[det[i, 1]:det[i, 3], det[i, 0]:det[i, 2], :]
        print(temp_crop.shape)
        aligned = misc.imresize(temp_crop, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        crop.append(prewhitened)
    # np.stack 将crop由一维list变为二维
    crop_image = np.stack(crop)
    return 1, det, crop_image


with tf.Graph().as_default():
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    with sess.as_default():
        # with tf.Session(config=config) as sess:
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
        # 创建load_and_align_data网络
        print('Creating networks and loading parameters')
        model = './20180402-114759/'
        facenet.load_model(model)

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
    # print(frame.shape)
    resize_img = cv2.resize(img, (int(img.shape[1] / scale), int(img.shape[0] / scale)))
    # cv2.imshow('camera1', frame)
    # frame = cv2.resize(frame, (640, 480))
    # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 获取 判断标识 bounding_box crop_image
    mark, bounding_box, crop_image = load_and_align_data(resize_img, 160, 22)
    # print(timer)
    name = ''
    if (mark):
        feed_dict = {images_placeholder: crop_image, phase_train_placeholder: False}
        emb = sess.run(embeddings, feed_dict=feed_dict)
        # print(emb.shape)
        temp_num = len(emb)
        # print(emb)
        fin_obj = []
        print(all_obj)
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
                # cv2.imwrite('./output/unknow_' + str(int(time.time())) + '.jpg', frame)  # 写入图片
            else:
                fin_obj.append(all_obj[dist_list.index(min_value)].split('.')[0])
                # cv2.imwrite('./output/' + str(
                #     all_obj[dist_list.index(min_value)].split('.')[0]) + '_' + str(
                #     int(time.time())) + '.jpg', frame)  # 写入图片
        # print(fin_obj)
        fin_obj_temp = fin_obj

        print(fin_obj_temp)
        print(len(fin_obj))
        # 在frame上绘制边框和文字
        for rec_position in range(temp_num):
            cv2.rectangle(resize_img, (bounding_box[rec_position, 0], bounding_box[rec_position, 1]),
                          (bounding_box[rec_position, 2], bounding_box[rec_position, 3]),
                          (0, 255, 0),
                          2, 8, 0)
            try:
                cv2.putText(
                    resize_img,
                    fin_obj[rec_position],
                    (bounding_box[rec_position, 0], bounding_box[rec_position, 1]),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    0.8,
                    (0, 0, 255),
                    thickness=2,
                    lineType=2)
            except IndexError as e:
                cv2.putText(
                    resize_img,
                    '',
                    (bounding_box[rec_position, 0], bounding_box[rec_position, 1]),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    0.8,
                    (0, 0, 255),
                    thickness=2,
                    lineType=2)
            name = fin_obj[rec_position]
            # t = time.time()
            # if int(t) % 3 == 0:
            #     str_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(t))
            #     cv2.imwrite('./output/' + str(fin_obj[rec_position]) + '_'
            #                 + str(str_time) + '.jpg', resize_img)  # 写入图片
    # cv2.imshow('camera', resize_img)
    result_image = cv2.imencode('.jpg', resize_img)[1]
    base64_data = str(base64.b64encode(result_image))[2:-1]
    stoptime = time.time()
    print('usetime:', str(stoptime - starttime))
    # print(name)
    return base64_data, name
