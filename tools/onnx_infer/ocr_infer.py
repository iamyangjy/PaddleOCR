#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2021/7/9 18:34
import cv2
from dbnet_infer import DBNET


def draw_bbox(img_path, result, color=(255, 0, 0), thickness=2):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)

        cv2.polylines(img_path, [point], True, color, thickness)

    return img_path


def ocr_pre():
    text_handle = DBNET(MODEL_PATH="./model/dbnet.onnx")
    img = cv2.imread("./test_imgs/self1.jpg")
    print(img.shape)
    box_list, score_list = text_handle.process(img, 1024)
    img = draw_bbox(img, box_list)
    cv2.imwrite("test11.jpg", img)


if __name__ == "__main__":
    ocr_pre()
