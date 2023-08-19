# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

import torch
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# ==============================================================
# 1. 변수 선언
# ==============================================================
parser = argparse.ArgumentParser(description='MobileSAM_Everything')

parser.add_argument('--subplot-rows', default=4, type=int, help='Plot의 Row에 들어갈 이미지 개수')
parser.add_argument('--subplot-columns', default=4, type=int, help='Plot의 Column에 들어갈 이미지 개수')
parser.add_argument('--subplot-count', default=0, type=int, help='SubPlot을 위한 이미지 개수')

args = parser.parse_args()

# ==============================================================
# 2. 모델 불러오기 + 세팅
# ==============================================================
# 1) 모델 불러오기
sam_checkpoint = "/home/hi/Jupyter/MobileSAM_Analysis/weights/mobile_sam.pt"
model_type = "vit_t"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(f'sam_model_registry.keys() : {sam_model_registry.keys()}') # ['default', 'vit_h', 'vit_l', 'vit_b', 'vit_t']
mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint) # 모델 구조 ???
mobile_sam = mobile_sam.to(device=device) # 모델 구조 ???
mobile_sam.eval() # MobileSAM

# 2) 모델 세팅
mask_generator = SamAutomaticMaskGenerator(mobile_sam) # 모델 구조 ???
predictor = SamPredictor(mobile_sam) # 모델 구조 ???

# ==============================================================
# 3. 이미지 불러오기
# ==============================================================
image = cv2.imread('/home/hi/Jupyter/MobileSAM_Analysis/app/assets/picture1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ==============================================================
# 4. Segment Everything
# ==============================================================
# --------------------------------------------------------------
# 1) 마스크 추출
# --------------------------------------------------------------
masks = mask_generator.generate(image)

# --------------------------------------------------------------
# 2) 마스크 병합 (면적 큰 마스크부터 순서대로)
# --------------------------------------------------------------
if len(masks) > 0:
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    """
    sorted_masks.shape : list 70 -> dict 7
    sorted_masks[0] : {'segmentation': array([[True, True, True, ..., False, False, False],
                            [True, True, True, ..., False, False, False],
                            [True, True, True, ..., False, False, False],
                            ...,
                            [False, False, False, ..., False, False, False],
                            [False, False, False, ..., False, False, False],
                            [False, False, False, ..., False, False, False]]), 'area': 222108, 'bbox': [0, 0, 566, 724],
     'predicted_iou': 1.0151423215866089, 'point_coords': [[516.671875, 204.53125]],
     'stability_score': 0.9847358465194702, 'crop_box': [0, 0, 769, 770]}
    """
    # crop_box는 뭐지??? predicted_iou는 무엇과 무엇 사이의 iou??? point_coords는 무엇의 point??? bbox랑 crop_box의 차이점은??? stablily score 계산은 어떻게 이루어짐???

    # print(masks[0]['segmentation'].shape)
    sorted_masks_only = torch.from_numpy(np.array([sorted_mask['segmentation'] for sorted_mask in sorted_masks])) # origin(list + numpy) -> numpy -> tensor : 0.02904 sec
    """
    [sorted_mask['segmentation'] for sorted_mask in sorted_masks] shape : (70, 770, 769)
    [sorted_mask['segmentation'] for sorted_mask in sorted_masks] : [array([[ True,  True,  True, ..., False, False, False],
       [ True,  True,  True, ..., False, False, False],
       [ True,  True,  True, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ...,  True,  True,  True],
       [False, False, False, ...,  True,  True,  True],
       [False, False, False, ...,  True,  True,  True],
       ...,
       [False, False, False, ...,  True,  True,  True],
       [False, False, False, ...,  True,  True,  True],
       [False, False, False, ...,  True,  True,  True]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [ True,  True,  True, ..., False, False, False],
       [ True,  True,  True, ..., False, False, False],
       [ True,  True,  True, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ...,  True,  True,  True],
       [False, False, False, ...,  True,  True,  True],
       [False, False, False, ...,  True,  True,  True]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [ True,  True,  True, ..., False, False, False],
       [ True,  True,  True, ..., False, False, False],
       [ True,  True,  True, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]])]
    """
    """
    sorted_masks_only shape : (70, 770, 769)
    sorted_masks_only : tensor([[[ True,  True,  True,  ..., False, False, False],
         [ True,  True,  True,  ..., False, False, False],
         [ True,  True,  True,  ..., False, False, False],
         ...,
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False]],

        [[False, False, False,  ...,  True,  True,  True],
         [False, False, False,  ...,  True,  True,  True],
         [False, False, False,  ...,  True,  True,  True],
         ...,
         [False, False, False,  ...,  True,  True,  True],
         [False, False, False,  ...,  True,  True,  True],
         [False, False, False,  ...,  True,  True,  True]],

        [[False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         ...,
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False]],

        ...,

        [[False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         ...,
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False]],

        [[False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         ...,
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False]],

        [[False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         ...,
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False]]])
    """
    # sorted_masks_only = torch.as_tensor([sorted_mask['segmentation'].tolist() for sorted_mask in sorted_masks]) # origin(list + numpy) -> list -> tensor : 4.38002 sec
    # sorted_masks_only = torch.as_tensor([sorted_mask['segmentation'] for sorted_mask in sorted_masks]) # origin(list + numpy)-> tensor : 15.70204 sec

# @torch.no_grad()
# def segment_everything(
#     image,
#     input_size=1024,
#     better_quality=False,
#     withContours=True,
#     use_retina=True,
#     mask_random_color=True,
# ):
#     global mask_generator
#
#     input_size = int(input_size)
#     w, h = image.size
#     scale = input_size / max(w, h)
#     new_w = int(w * scale)
#     new_h = int(h * scale)
#     image = image.resize((new_w, new_h))
#
#     nd_image = np.array(image)
#     annotations = mask_generator.generate(nd_image)
#
#     fig = fast_process(
#         annotations=annotations,
#         image=image,
#         device=device,
#         scale=(1024 // input_size),
#         better_quality=better_quality,
#         mask_random_color=mask_random_color,
#         bbox=None,
#         use_retina=use_retina,
#         withContours=withContours,
#     )
#     return fig
#
#
# def segment_with_points(
#     image,
#     input_size=1024,
#     better_quality=False,
#     withContours=True,
#     use_retina=True,
#     mask_random_color=True,
# ):
#     global global_points
#     global global_point_label
#
#     input_size = int(input_size)
#     w, h = image.size
#     scale = input_size / max(w, h)
#     new_w = int(w * scale)
#     new_h = int(h * scale)
#     image = image.resize((new_w, new_h))
#
#     scaled_points = np.array(
#         [[int(x * scale) for x in point] for point in global_points]
#     )
#     scaled_point_label = np.array(global_point_label)
#
#     if scaled_points.size == 0 and scaled_point_label.size == 0:
#         print("No points selected")
#         return image, image
#
#     print(scaled_points, scaled_points is not None)
#     print(scaled_point_label, scaled_point_label is not None)
#
#     nd_image = np.array(image)
#     predictor.set_image(nd_image)
#     masks, scores, logits = predictor.predict(
#         point_coords=scaled_points,
#         point_labels=scaled_point_label,
#         multimask_output=True,
#     )
#
#     results = format_results(masks, scores, logits, 0)
#
#     annotations, _ = point_prompt(
#         results, scaled_points, scaled_point_label, new_h, new_w
#     )
#     annotations = np.array([annotations])
#
#     fig = fast_process(
#         annotations=annotations,
#         image=image,
#         device=device,
#         scale=(1024 // input_size),
#         better_quality=better_quality,
#         mask_random_color=mask_random_color,
#         bbox=None,
#         use_retina=use_retina,
#         withContours=withContours,
#     )
#
#     global_points = []
#     global_point_label = []
#     # return fig, None
#     return fig, image
#
#
# def get_points_with_draw(image, label, evt: gr.SelectData):
#     global global_points
#     global global_point_label
#
#     x, y = evt.index[0], evt.index[1]
#     point_radius, point_color = 15, (255, 255, 0) if label == "Add Mask" else (
#         255,
#         0,
#         255,
#     )
#     global_points.append([x, y])
#     global_point_label.append(1 if label == "Add Mask" else 0)
#
#     print(x, y, label == "Add Mask")
#
#     # 创建一个可以在图像上绘图的对象
#     draw = ImageDraw.Draw(image)
#     draw.ellipse(
#         [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
#         fill=point_color,
#     )
#     return image
#
#
# cond_img_e = gr.Image(label="Input", value=default_example[0], type="pil")
# cond_img_p = gr.Image(label="Input with points", value=default_example[0], type="pil")
#
# segm_img_e = gr.Image(label="Segmented Image", interactive=False, type="pil")
# segm_img_p = gr.Image(
#     label="Segmented Image with points", interactive=False, type="pil"
# )
#
# global_points = []
# global_point_label = []
#
# input_size_slider = gr.components.Slider(
#     minimum=512,
#     maximum=1024,
#     value=1024,
#     step=64,
#     label="Input_size",
#     info="Our model was trained on a size of 1024",
# )
#
# with gr.Blocks(css=css, title="Faster Segment Anything(MobileSAM)") as demo:
#     with gr.Row():
#         with gr.Column(scale=1):
#             # Title
#             gr.Markdown(title)
#
#     # with gr.Tab("Everything mode"):
#     #     # Images
#     #     with gr.Row(variant="panel"):
#     #         with gr.Column(scale=1):
#     #             cond_img_e.render()
#     #
#     #         with gr.Column(scale=1):
#     #             segm_img_e.render()
#     #
#     #     # Submit & Clear
#     #     with gr.Row():
#     #         with gr.Column():
#     #             input_size_slider.render()
#     #
#     #             with gr.Row():
#     #                 contour_check = gr.Checkbox(
#     #                     value=True,
#     #                     label="withContours",
#     #                     info="draw the edges of the masks",
#     #                 )
#     #
#     #                 with gr.Column():
#     #                     segment_btn_e = gr.Button(
#     #                         "Segment Everything", variant="primary"
#     #                     )
#     #                     clear_btn_e = gr.Button("Clear", variant="secondary")
#     #
#     #             gr.Markdown("Try some of the examples below ⬇️")
#     #             gr.Examples(
#     #                 examples=examples,
#     #                 inputs=[cond_img_e],
#     #                 outputs=segm_img_e,
#     #                 fn=segment_everything,
#     #                 cache_examples=True,
#     #                 examples_per_page=4,
#     #             )
#     #
#     #         with gr.Column():
#     #             with gr.Accordion("Advanced options", open=False):
#     #                 # text_box = gr.Textbox(label="text prompt")
#     #                 with gr.Row():
#     #                     mor_check = gr.Checkbox(
#     #                         value=False,
#     #                         label="better_visual_quality",
#     #                         info="better quality using morphologyEx",
#     #                     )
#     #                     with gr.Column():
#     #                         retina_check = gr.Checkbox(
#     #                             value=True,
#     #                             label="use_retina",
#     #                             info="draw high-resolution segmentation masks",
#     #                         )
#     #             # Description
#     #             gr.Markdown(description_e)
#     #
#     with gr.Tab("Point mode"):
#         # Images
#         with gr.Row(variant="panel"):
#             with gr.Column(scale=1):
#                 cond_img_p.render()
#
#             with gr.Column(scale=1):
#                 segm_img_p.render()
#
#         # Submit & Clear
#         with gr.Row():
#             with gr.Column():
#                 with gr.Row():
#                     add_or_remove = gr.Radio(
#                         ["Add Mask", "Remove Area"],
#                         value="Add Mask",
#                     )
#
#                     with gr.Column():
#                         segment_btn_p = gr.Button(
#                             "Start segmenting!", variant="primary"
#                         )
#                         clear_btn_p = gr.Button("Restart", variant="secondary")
#
#                 gr.Markdown("Try some of the examples below ⬇️")
#                 gr.Examples(
#                     examples=examples,
#                     inputs=[cond_img_p],
#                     # outputs=segm_img_p,
#                     # fn=segment_with_points,
#                     # cache_examples=True,
#                     examples_per_page=4,
#                 )
#
#             with gr.Column():
#                 # Description
#                 gr.Markdown(description_p)
#
#     cond_img_p.select(get_points_with_draw, [cond_img_p, add_or_remove], cond_img_p)
#
#     # segment_btn_e.click(
#     #     segment_everything,
#     #     inputs=[
#     #         cond_img_e,
#     #         input_size_slider,
#     #         mor_check,
#     #         contour_check,
#     #         retina_check,
#     #     ],
#     #     outputs=segm_img_e,
#     # )
#
#     segment_btn_p.click(
#         segment_with_points, inputs=[cond_img_p], outputs=[segm_img_p, cond_img_p]
#     )
#
#     def clear():
#         return None, None
#
#     def clear_text():
#         return None, None, None
#
#     # clear_btn_e.click(clear, outputs=[cond_img_e, segm_img_e])
#     clear_btn_p.click(clear, outputs=[cond_img_p, segm_img_p])
#
# demo.queue()
# demo.launch()