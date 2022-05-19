#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import cv2
from tqdm import tqdm
import numpy as np
import config as cfg
import onnxruntime
from time import time

import sys
sys.path.append('.')
from onnx_infer_utils import COCO_CLASSES
from onnx_infer_utils import preproc as preprocess
from onnx_infer_utils import mkdir, multiclass_nms, vis


class onnx_infer():
    def __init__(self):
        self.input_shape = cfg.input_shape  # h, w
        self.model = cfg.model
        self.score_thr = cfg.score_thr
        self.output_dir = cfg.output_dir
        # --------------------init session------------------------
        self.session = onnxruntime.InferenceSession(self.model)

    def infer(self, origin_img):
        img, ratio = preprocess(origin_img, self.input_shape)

        ort_inputs = {self.session.get_inputs()[0].name: img[None, :, :, :]}
        output = self.session.run(None, ort_inputs)  # [(1, 9072, 45)]
        predictions = output[0][0]  # 9072, 45

        boxes = predictions[:, :4]  # xywh
        scores = predictions[:, 4:5] * predictions[:, 5:]

        return boxes, scores, ratio

    def postprocess(self, out_name, origin_img, boxes, scores, ratio):
        boxes_xyxy = np.ones_like(boxes)  # p1p2
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                            conf=self.score_thr, class_names=COCO_CLASSES)

        mkdir(self.output_dir)
        output_path = os.path.join(self.output_dir, out_name)
        cv2.imwrite(output_path, origin_img)
        return origin_img

if __name__ == '__main__':
    image_path = '/home/ecnu-lzw/bwz/ocr-gy/steelDatasets/datasets_img/2019_2019-10-26103539.jpg'

    img_num = 1
    vis_out = True
    tick = time()
    origin_img = cv2.imread(image_path)  # 0.1566s
    of_wp = onnx_infer()  # 0.822s
    for _ in tqdm(range(img_num)):
        xywh_boxes, scores, ratio = of_wp.infer(origin_img)  # 0.016s
        tick1 = time()
        if vis_out:
            origin_img = of_wp.postprocess(image_path.split("/")[-1], origin_img, xywh_boxes, scores, ratio)  # 0.19s
        print(time() - tick1)
    tock = time()
    time_elp = (tock - tick) / img_num
    print(f'avg time: {time_elp:.4f}')
    # python3 /home/ecnu-lzw/bwz/ocr-gy/YOLOX_inference/onnx_inference_api.py
