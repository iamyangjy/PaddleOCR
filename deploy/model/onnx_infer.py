#!/usr/bin/env python
# -*- coding:utf-8 -*-

import onnxruntime
import numpy as np
import cv2
import math
from onnx_rec_postprocess import HpocrCTCLabelDecode
import time
import os
import imghdr
from PIL import ImageFont, ImageDraw, Image


class OnnxRec(object):
    def __init__(self):
        self.rec_image_shape = [3, 32, 512]
        self.character_type = 'ch'
        self.rec_batch_num = 1
        self.rec_algorithm = 'rec'
        self.max_text_length = 1000
        self.character_dict_path = './char_std_v1.txt'

        # 后处理
        self.postprocess_op = HpocrCTCLabelDecode(self.character_dict_path, 'ch', True)
        # 加载onnx模型
        self.onnx_file = "./rec/mv3_none_lstm_ctc/scale_0.35_lstm_48/rec_mv3_none_lstm_ctc_dynamic.onnx"
        self.onnx_session = onnxruntime.InferenceSession(self.onnx_file)

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        if self.character_type == "ch":
            imgW = int((32 * max_wh_ratio))
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list, batch_num):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        # rec_res = []
        rec_res = [['', 0.0]] * img_num
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                # h, w = img_list[ino].shape[0:2]
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]], max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            # print(norm_img_batch)

            starttime = time.time()
            outputs = self.onnx_session.run(['save_infer_model/scale_0.tmp_0'], input_feed={'image': norm_img_batch})
            preds = outputs[0]
            rec_result = self.postprocess_op(preds)
            # print(rec_result)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
            elapse += time.time() - starttime
        return rec_res, elapse


def get_image_file_list(img_file):
    imgs_lists = []
    img_end = {'jpg', 'bmp', 'png', 'jpeg'}
    if os.path.isfile(img_file) and imghdr.what(img_file) in img_end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path) and imghdr.what(file_path) in img_end:
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists


def save_pre_img(path, pre_text):
    font = ImageFont.truetype("./simsun.ttc", 25)

    img = cv2.imread(path)

    pre_img = img.copy()
    pre_img.fill(255)
    b, g, r, a = 255, 0, 0, 0
    img_pil = Image.fromarray(pre_img)
    # 创建画板
    draw = ImageDraw.Draw(img_pil)
    # 设置字体的颜色
    b, g, r, a = 0, 0, 0, 0
    # 在图片上绘制中文
    # print(pre_text)
    draw.text((2, 5), pre_text[0], font=font, fill=(0, 0, 255, 0))
    # 将图片转为numpy array的数据格式
    pre_img = np.array(img_pil)
    # res = cv2.vconcat([img, pretext])

    org_dir, name = path.rsplit("/", 1)

    pre_dir = "./img/pre"
    pre_name_path = os.path.join(pre_dir, name.replace(".jpg", "_pre.jpg"))
    org_name_path = os.path.join(pre_dir, name)

    for ii in pre_text[2]:
        cv2.line(img, (ii * 4, 0), (ii * 4, 32), (0, 0, 255), 1, 4)
    cv2.imwrite(org_name_path, img)
    cv2.imwrite(pre_name_path, pre_img)


def onnx_pre(image_dir, batch_num=1):
    text_recognizer = OnnxRec()
    image_file_list = get_image_file_list(image_dir)
    total_run_time = 0.0
    total_images_num = 0
    valid_image_file_list = []
    img_list = []
    for idx, image_file in enumerate(image_file_list):
        img = cv2.imread(image_file)
        valid_image_file_list.append(image_file)
        img_list.append(img)
        if len(img_list) >= batch_num or idx == len(image_file_list) - 1:
            rec_res, predict_time = text_recognizer(img_list, batch_num)
            total_run_time += predict_time
            for ino in range(len(img_list)):
                print("Predicts of {}:{}".format(valid_image_file_list[
                                                     ino], rec_res[ino]))
                save_pre_img(valid_image_file_list[ino], rec_res[ino])
            total_images_num += len(valid_image_file_list)
            valid_image_file_list = []
            img_list = []
    print("Total predict time for {} images, cost: {:.3f}".format(total_images_num, total_run_time))


if __name__ == "__main__":
    image_dir = "./img/rec/"
    onnx_pre(image_dir, 1)

    # rec = OnnxRec()
    # print(rec.onnx_session.get_inputs()[0].name)
    # out_detect = rec.onnx_session.get_outputs()
    # outputs_detect = list()
    # for i in out_detect:
    #     outputs_detect.append(i.name)
    # ['268']
    # print(outputs_detect)
