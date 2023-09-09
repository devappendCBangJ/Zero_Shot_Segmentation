# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
# --------------------------------------------------------------
# 1) 딥러닝 라이브러리
# --------------------------------------------------------------
from mobile_sam import sam_model_registry, SamPredictor
import torch

# --------------------------------------------------------------
# 2) 기본 라이브러리
# --------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import cv2

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QGridLayout, QCheckBox, QRadioButton, QButtonGroup, QLineEdit
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage
from PyQt5.QtCore import Qt, QPoint
import os

# ==============================================================
# 1. 함수 정의
# ==============================================================
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# ==============================================================
# 2. Main문
# ==============================================================
# --------------------------------------------------------------
# 1) 초기화
# --------------------------------------------------------------
# (1) 초기 세팅
device = "cuda" if torch.cuda.is_available() else "cpu"

# (2) SAM 모델 초기 세팅
model_type = "vit_t"
sam_model_path = "/home/hi/Jupyter/SegmentAnything/MobileSAM_Analysis/weights/mobile_sam.pt"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_model_path)
mobile_sam.to(device=device)
mobile_sam.eval()

predictor = SamPredictor(mobile_sam)

# (3) 이미지 폴더 세팅
base_image_folder = '/media/hi/SK Gold P31/Capstone/GolfBall/3_2_Crawling_cp/golf ball in sand bunker_for background'
base_label_folder = '/media/hi/SK Gold P31/Capstone/GolfBall/4_1_LabelBang_AutoLabeling/golf ball in sand bunker_for background'

if not os.path.exists(base_label_folder):
    os.makedirs(base_label_folder)

# (4) Label 초기화 세팅
default_class_num = 0
default_min_area_ratio = 0.002

# --------------------------------------------------------------
# 2) 이미지 클릭 시 작업
# --------------------------------------------------------------
class ClickableImageLabel(QLabel):
    # (1) 초기 세팅
    def __init__(self, scaled_pixmap, real_image_size, abs_image_path, abs_seg_path, abs_bbox_path):
        super().__init__()
        self.setPixmap(scaled_pixmap)

        self.input_labels = []
        self.input_points = []
        self.scaled_points = []
        self.class_nums = []
        self.final_class_nums = []

        self.loaded_bboxes = []
        self.loaded_contours = []

        self.scaled_image_size = {'scaled_image_width':scaled_pixmap.width(), 'scaled_image_height':scaled_pixmap.height()}
        self.real_image_size = real_image_size
        self.abs_image_path = abs_image_path
        self.abs_seg_path = abs_seg_path
        self.abs_bbox_path = abs_bbox_path

        self.clicked_pos = None

    # (2) 마우스 왼쪽 클릭 : 좌표 저장
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked_pos = event.pos()
            self.update()

    # (3) 마우스 왼쪽 클릭 이후 : 점 그리기
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.clicked_pos:
            painter = QPainter(self)
            for scaled_point, input_label, class_num in zip(self.scaled_points, self.input_labels, self.class_nums):
                if input_label == 1:
                    painter.setPen(QPen(Qt.blue, 5))
                else:
                    painter.setPen(QPen(Qt.red, 5))
                painter.drawPoint(QPoint(*scaled_point))
                painter.drawText(*scaled_point, str(class_num))

    # def updateImageWithMask(self, original_image, mask):
    #     # qimage = QImage(masked_image.data, masked_image.shape[1], masked_image.shape[0], QImage.Format_RGB888)
    #     # pixmap = QPixmap.fromImage(qimage)
    #     # self.setPixmap(pixmap)
    #     # self.clicked_pos = None  # 이진화된 이미지로 업데이트된 후, 클릭 위치를 초기화합니다.
    #     # self.update()
    #
    #     # 마스크 이미지를 기반으로 투명도가 있는 QPixmap 객체를 생성합니다.
    #     mask_pixmap = QPixmap.fromImage(mask)
    #     painter = QPainter(original_image)
    #     painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
    #     painter.drawPixmap(0, 0, mask_pixmap)
    #     painter.end()
    #
    #     self.setPixmap(original_image)
    #     self.clicked_pos = None  # 마스크를 적용한 후 클릭 위치를 초기화합니다.
    #     self.update()

# --------------------------------------------------------------
# 3) Window 작업
# --------------------------------------------------------------
class ImageWindow(QWidget):
    # (1) 초기화
    def __init__(self, image_folder):
        super().__init__()

        # 1] 이미지 + 텍스트 파일명 불러오기
        self.label_filenames = []
        self.image_filenames = []
        for filename in os.listdir(image_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                self.image_filenames.append(filename)
                self.label_filenames.append(f'{filename[:-4]}.txt')

        self.current_image_index = 0

        # 2] 초기 세팅
        self.initUI()

    def initUI(self):
        # [1] 윈도우창 제목
        self.setWindowTitle('Label Bang')

        # [2] 레이아웃 선언
        self.main_layout = QVBoxLayout() # 전체 틀 Plot
        self.grid_layout = QGridLayout() # 이미지 Plot
        self.mask_size_layout = QVBoxLayout() # Mask Size Checkbox
        self.mask_batch_layout = QVBoxLayout() # Mask Batch Checkbox
        self.add_clear_layout = QVBoxLayout() # Point Choice Checkbox
        self.class_num_box_text_layout = QHBoxLayout()  # Class Num Box Text
        self.min_area_ratio_box_text_layout = QHBoxLayout()  # Min Area Ratio Box Text
        self.text_layout = QVBoxLayout() # Text Checkbox
        self.mask_size_batch_text_layout = QHBoxLayout() # Mask Size Batch Text Checkbox
        self.button_layout = QHBoxLayout() # Button Plot

        # [3] 이전 / 다음 Button EventHandler
        self.prev_button = QPushButton('Previous', self)
        self.next_button = QPushButton('Next', self)

        self.prev_button.clicked.connect(self.loadPreviousImages)
        self.next_button.clicked.connect(self.loadNextImages)

        # [4] Class명 설정 Textbox
        self.class_num_title = QLabel(self)
        self.class_num_box = QLineEdit(self)
        self.min_area_ratio_title = QLabel(self)
        self.min_area_ratio_box = QLineEdit(self)
        self.labeled_image_count = QLabel(self)
        self.labeled_mask_count = QLabel(self)

        self.class_num_title.setFixedSize(200, 17)
        self.class_num_box.setFixedSize(200, 17)
        self.min_area_ratio_title.setFixedSize(200, 17)
        self.min_area_ratio_box.setFixedSize(200, 17)
        self.labeled_image_count.setFixedSize(200, 17)
        self.labeled_mask_count.setFixedSize(200, 17)

        self.class_num_title.setText(f'class_num_box_count ')
        self.class_num_box.setText(str(default_class_num))
        self.min_area_ratio_title.setText(f'min_area ratio ')
        self.min_area_ratio_box.setText(str(default_min_area_ratio))
        self.labeled_image_count.setText(f'image_count : 0')
        self.labeled_mask_count.setText(f'mask_count : 0')

        # [5] Point 종류 설정 RadioButton
        self.add_clear_group = QButtonGroup(self)

        self.add_check = QRadioButton("Add Point (Q)", self)
        self.minus_check = QRadioButton("Minus Point (W)", self)
        self.clear_only_one_check = QRadioButton("Clear Only One Point (E)", self)
        self.clear_all_check = QRadioButton("Clear All Point (R)", self)
        self.add_check.setFixedSize(200, 17)
        self.minus_check.setFixedSize(200, 17)
        self.clear_only_one_check.setFixedSize(200, 17)
        self.clear_all_check.setFixedSize(200, 17)

        self.add_clear_group.addButton(self.add_check)
        self.add_clear_group.addButton(self.minus_check)
        self.add_clear_group.addButton(self.clear_only_one_check)
        self.add_clear_group.addButton(self.clear_all_check)

        self.add_check.setChecked(True)

        # [6] Mask 크기 설정 Checkbox
        self.small_check = QCheckBox("Small Mask (A)", self)
        self.medium_check = QCheckBox("Medium Mask (S)", self)
        self.large_check = QCheckBox("Large Mask (D)", self)
        self.small_check.setFixedSize(200, 17)
        self.medium_check.setFixedSize(200, 17)
        self.large_check.setFixedSize(200, 17)

        self.small_check.setChecked(True)
        self.medium_check.setChecked(True)
        self.large_check.setChecked(True)

        # [7] Batch 단위 Mask 추출 설정 RadioButton
        self.mask_batch_group = QButtonGroup(self)

        self.batch_check = QRadioButton("Batch Mask (Z)", self)
        self.only_one_check = QRadioButton("Only One Mask (X)", self)
        self.batch_check.setFixedSize(200, 17)
        self.only_one_check.setFixedSize(200, 17)

        self.mask_batch_group.addButton(self.batch_check)
        self.mask_batch_group.addButton(self.only_one_check)

        self.only_one_check.setChecked(True)

        # [8] 이미지 Plot Update
        self.updateImageGrid()

        # [9] Main Layout 설정
        self.button_layout.addWidget(self.prev_button)
        self.button_layout.addWidget(self.next_button)

        self.add_clear_layout.addWidget(self.add_check)
        self.add_clear_layout.addWidget(self.minus_check)
        self.add_clear_layout.addWidget(self.clear_only_one_check)
        self.add_clear_layout.addWidget(self.clear_all_check)

        self.mask_size_layout.addWidget(self.small_check)
        self.mask_size_layout.addWidget(self.medium_check)
        self.mask_size_layout.addWidget(self.large_check)

        self.mask_batch_layout.addWidget(self.batch_check)
        self.mask_batch_layout.addWidget(self.only_one_check)

        self.class_num_box_text_layout.addWidget(self.class_num_title)
        self.class_num_box_text_layout.addWidget(self.class_num_box)
        self.min_area_ratio_box_text_layout.addWidget(self.min_area_ratio_title)
        self.min_area_ratio_box_text_layout.addWidget(self.min_area_ratio_box)

        self.text_layout.addLayout(self.class_num_box_text_layout)
        self.text_layout.addLayout(self.min_area_ratio_box_text_layout)
        self.text_layout.addWidget(self.labeled_image_count)
        self.text_layout.addWidget(self.labeled_mask_count)

        self.mask_size_batch_text_layout.addLayout(self.add_clear_layout)
        self.mask_size_batch_text_layout.addLayout(self.mask_size_layout)
        self.mask_size_batch_text_layout.addLayout(self.mask_batch_layout)
        self.mask_size_batch_text_layout.addLayout(self.text_layout)

        self.main_layout.addLayout(self.grid_layout)
        self.main_layout.addLayout(self.button_layout)
        self.main_layout.addLayout(self.mask_size_batch_text_layout)

        # [9] 최종 Layout 설정
        self.setLayout(self.main_layout)
        self.setFixedSize(1200, 1000)

    # - 다음 버튼 클릭 : 이미지 index 변경 + Plot Update
    def loadNextImages(self):
        self.current_image_index += 16
        if self.current_image_index >= len(self.image_filenames):
            self.current_image_index -= 16
        self.updateImageGrid()

    # - 이전 버튼 클릭 : 이미지 index 변경 + Plot Update
    def loadPreviousImages(self):
        self.current_image_index -= 16
        if self.current_image_index < 0:
            self.current_image_index = 0
        self.updateImageGrid()

    # - 전체 이미지 업데이트
    def updateImageGrid(self):
        # [1] 모든 이미지 Plot 제거
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            self.grid_layout.removeWidget(widget)
            widget.deleteLater()

        # [2] [이미지 불러오기 + Resize + 클릭 좌표 & 파일명 추출 + Plot Update] + Label 파일 없는 경우 생성
        for i in range(4):
            for j in range(4):
                idx = self.current_image_index + i * 4 + j
                if idx < len(self.image_filenames):
                    # 1]] 다양한 경로 불러오기
                    abs_image_path = f'{base_image_folder}/{self.image_filenames[idx]}'
                    seg_folder = f'{base_label_folder}/seg'
                    bbox_folder = f'{base_label_folder}/bbox'
                    if not os.path.exists(seg_folder):
                        os.makedirs(seg_folder)
                    if not os.path.exists(bbox_folder):
                        os.makedirs(bbox_folder)
                    abs_seg_path = f'{seg_folder}/{self.label_filenames[idx]}'
                    abs_bbox_path = f'{bbox_folder}/{self.label_filenames[idx]}'

                    # 2]] 이미지 불러오기
                    image = cv2.imread(abs_image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    img_h, img_w, img_c = image.shape
                    bytes_per_line = 3 * img_w

                    # 3]] Label 파일 없는 경우 생성
                    if not os.path.exists(abs_seg_path):
                        with open(abs_seg_path, 'w') as seg_txt:
                            pass
                    if not os.path.exists(abs_bbox_path):
                        with open(abs_bbox_path, 'w') as bbox_txt:
                            pass

                    # 4]] Label 기존 정보 불러오기 + 이미지에 그리기
                    with open(abs_seg_path, 'r') as seg_txt:
                        if os.path.getsize(abs_seg_path):
                            loaded_color_masks = []
                            for seg_idx, line in enumerate(seg_txt):
                                if seg_idx == 0:
                                    r, g, b = (255, 144, 30)
                                else:
                                    r, g, b = np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255)
                                class_num = line.strip()[0]
                                scaled_loaded_contour = line.strip()[2:].split()
                                loaded_contour = [int(float(coor) * img_w) if c_idx % 2 == 0 else int(float(coor) * img_h) for c_idx, coor in enumerate(scaled_loaded_contour)]
                                loaded_contour = np.array(loaded_contour).reshape(-1, 1, 2)

                                loaded_color_mask = np.zeros_like(image)
                                cv2.drawContours(loaded_color_mask, [loaded_contour], -1, (r, g, b), thickness=-1)
                                if not len(loaded_color_masks):
                                    loaded_color_masks = loaded_color_mask
                                else:
                                    loaded_color_masks += loaded_color_mask

                            loaded_masked_image = cv2.addWeighted(image, 0.5, np.array(loaded_color_masks), 1 - 0.5, 0)
                        else:
                            loaded_masked_image = image
                    with open(abs_bbox_path, 'r') as bbox_txt:
                        for bbox_idx, line in enumerate(bbox_txt):
                            if bbox_idx == 0:
                                r, g, b = (255, 0, 0)
                            else:
                                r, g, b = np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255)
                            scaled_loaded_bbox = line.strip()[2:].split()
                            scaled_box_cx, scaled_box_cy, scaled_box_w, scaled_box_h = list(map(float, scaled_loaded_bbox))
                            scaled_box_x = scaled_box_cx - scaled_box_w / 2
                            scaled_box_y = scaled_box_cy - scaled_box_h / 2

                            box_x = int(scaled_box_x * img_w)
                            box_y = int(scaled_box_y * img_h)
                            box_w = int(scaled_box_w * img_w)
                            box_h = int(scaled_box_h * img_h)

                            cv2.rectangle(loaded_masked_image, (box_x, box_y), ((box_x + box_w), (box_y + box_h)), (r, g, b), 2)
                            cv2.putText(loaded_masked_image, class_num, (box_x + box_w // 2, box_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (r, g, b), 3, cv2.LINE_AA)

                    # 5]] 마스크 영역 출력 설정 by PyQt5
                    q_image = QImage(loaded_masked_image.data, img_w, img_h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_image)
                    scaled_pixmap = pixmap.scaled(300, 200, Qt.KeepAspectRatio)

                    # 6]] 클릭 좌표 & 파일명 추출
                    label = ClickableImageLabel(scaled_pixmap, {'real_image_width':pixmap.width(), 'real_image_height':pixmap.height()}, abs_image_path, abs_seg_path, abs_bbox_path)
                    label.mousePressEvent = lambda event, l=label: self.imageClicked(event, l)

                    # 7]] Plot Update
                    self.grid_layout.addWidget(label, i, j)
                    self.grid_layout.setAlignment(label, Qt.AlignTop)

                    # 8]] 마스크 영역 출력 by PyQt5
                    label.setPixmap(scaled_pixmap)
                else:
                    break

        self.adjustSize()

    # - 이미지 클릭 좌표 저장 + SAM Mask 씌우기
    def imageClicked(self, event, label):
        # [0] Batch Size 정보 불러오기
        is_small, is_medium, is_large = self.small_check.isChecked(), self.medium_check.isChecked(), self.large_check.isChecked()
        if not is_small and not is_medium and not is_large:
            is_small, is_medium, is_large = True, True, True

        # 실제 사진 상에서의 마우스 위치 = (Scaled 마우스 위치) / (Scaled 전체 너비) * (실제 사진 너비)

        # [1] 이미지 불러오기
        image = cv2.imread(label.abs_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w, img_c = image.shape
        bytes_per_line = 3 * img_w
        masked_image = image

        # [2] Label 파일 초기화
        with open(label.abs_seg_path, 'w') as seg_txt:
            pass
        with open(label.abs_bbox_path, 'w') as bbox_txt:
            pass

        # - Label을 모두 지우는 경우 : Label 정보 초기화
        if self.clear_all_check.isChecked():
            # [3] Label 정보 초기화
            label.input_labels = []
            label.input_points = []
            label.scaled_points = []
            label.class_nums = []
            label.final_class_nums = []

        # - Label을 1개만 지우는 경우 or Label을 추가하는 경우
        else:
            # - Label을 1개만 지우는 경우 + Label 파일 있는 경우 : 가장 최신 Label 제거
            if self.clear_only_one_check.isChecked():
                # [3] 가장 최신 Label 제거
                if label.input_labels:
                    label.input_labels.pop()
                    label.input_points.pop()
                    label.scaled_points.pop()
                    label.class_nums.pop()
                    # label.final_class_nums.pop()

            # - Label을 추가하는 경우 : 이미지 클릭 좌표 & 파일명 추출 + Label 정보 추가
            else:
                # [3] 이미지 클릭 좌표 & 파일명 추출
                x_ratio = label.real_image_size['real_image_width'] / label.scaled_image_size['scaled_image_width']
                y_ratio = label.real_image_size['real_image_height'] / label.scaled_image_size['scaled_image_height']

                real_x = event.x() * x_ratio
                real_y = event.y() * y_ratio

                # [4] Label 정보 추가
                label.input_labels.append(self.add_check.isChecked())
                label.input_points.append([real_x, real_y])
                label.scaled_points.append([event.x(), event.y()])
                label.class_nums.append(self.class_num_box.text())

            # - Label 1개 이상 입력한 경우 : SAM의 input image 세팅 + 마스크 추출 + 마스크 전처리 + 마스크 최종 선택 + Label 파일 저장
            if label.input_labels:
                # [5] SAM의 input image 세팅
                predictor.set_image(image)

                # - Mask 여러개 추출하는 경우
                if self.batch_check.isChecked():
                    # [6] 마스크 추출 - 여러개 Mask (Batch 단위)
                    transformed_input_points = predictor.transform.apply_coords_torch(torch.tensor(label.input_points, device=predictor.device), image.shape[:2])
                    masks, scores, logits = predictor.predict_torch(point_coords=transformed_input_points.unsqueeze(1),
                                                                    point_labels=torch.tensor(label.input_labels, device=predictor.device).unsqueeze(1),
                                                                    multimask_output=True)
                    masks, scores, logits = np.array(masks.cpu()), np.array(scores.cpu()), np.array(logits.cpu())

                    # [7] 원하는 크기의 마스크만 선별
                    masks = masks[:, [is_small, is_medium, is_large], :, :]
                    scores = scores[:, [is_small, is_medium, is_large]]
                    logits = logits[:, [is_small, is_medium, is_large]]

                    label.final_class_nums = label.class_nums
                # - Mask 1개만 추출하는 경우
                elif self.only_one_check.isChecked():
                    # [6] 마스크 추출 - 1개 Mask
                    mask, score, logit = predictor.predict(point_coords=np.array(label.input_points),
                                                              point_labels=np.array(label.input_labels),
                                                              multimask_output=True)

                    # [7] 원하는 크기의 마스크만 선별
                    mask = mask[[is_small, is_medium, is_large], :, :]
                    score = score[[is_small, is_medium, is_large]]
                    logit = logit[[is_small, is_medium, is_large]]

                    masks = mask[np.newaxis, :]
                    scores = score[np.newaxis, :]
                    logits = logit[np.newaxis, :]

                    label.final_class_nums = [label.class_nums[-1]]

                # [8] 최종 마스크 선택 + Label 파일 저장
                mask_area_min = img_w * img_h * float(self.min_area_ratio_box.text())
                if len(masks):
                    color_masks = []
                    for mask, score, logit, class_num in zip(masks, scores, logits, label.final_class_nums):
                        # 1]] 최종 마스크 선택
                        mask_result = mask[np.argmax(score), :, :]
                        """
                        mask_result.shape : (357, 500)
                        mask_result : [[False False False ... False False False], [False False False ... False False False], [False False False ... False False False], ..., [False False False ... False False False], [False False False ... False False False], [False False False ... False False False]]
                        """
                        mask_result = mask_result.astype(np.uint8)

                        # mask_rows_true = np.any(mask_result, axis = 1)
                        # mask_cols_true = np.any(mask_result, axis = 0)
                        #
                        # y_min, y_max = np.where(mask_rows_true)[0][0, -1]
                        # x_min, x_max = np.where(mask_cols_true)[0][0, -1]
                        # print(y_min, y_max, x_min, x_max)

                        # 2]] Contour 추출
                        contours, hierarchys = cv2.findContours(mask_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        """
                        contours shape : tuple 1 -> (132, 1, 2)
                        contours[0] : [[[205 126]],, [[204 127]],, [[199 127]],, [[198 128]],, [[196 128]],, [[195 129]],, [[193 129]],, [[192 130]],, [[190 130]],, [[189 131]],, [[188 131]],, [[187 132]],, [[186 132]],, [[185 133]],, [[184 133]],, [[183 134]],, [[182 134]],, [[177 139]],, [[177 140]],, [[173 144]],, [[173 145]],, [[172 146]],, [[172 147]],, [[171 148]],, [[171 149]],, [[170 150]],, [[170 151]],, [[169 152]],, [[169 154]],, [[168 155]],, [[168 157]],, [[167 158]],, [[167 163]],, [[166 164]],, [[166 171]],, [[167 172]],, [[167 177]],, [[168 178]],, [[168 180]],, [[169 181]],, [[169 183]],, [[170 184]],, [[170 185]],, [[171 186]],, [[171 188]],, [[173 190]],, [[173 191]],, [[174 192]],, [[174 193]],, [[181 200]],, [[182 200]],, [[183 201]],, [[184 201]],, [[187 204]],, [[189 204]],, [[190 205]],, [[191 205]],, [[192 206]],, [[193 206]],, [[194 207]],, [[198 207]],, [[199 208]],, [[203 208]],, [[204 209]],, [[211 209]],, [[212 208]],, [[216 208]],, [[217 207]],, [[219 207]],, [[220 206]],, [[222 206]],, [[223...
                        """

                        # 3]] Contour 전처리 (최대 크기 Contour 추출 + Scaling)
                        result_contours = []
                        result_scaled_large_contours = []
                        for contour in contours:
                            """
                            contour.shape : (132, 1, 2)
                            contour : [[[206 126]],, [[205 127]],, [[199 127]],, [[198 128]],, [[196 128]],, [[195 129]],, [[193 129]],, [[192 130]],, [[190 130]],, [[189 131]],, [[188 131]],, [[187 132]],, [[186 132]],, [[185 133]],, [[184 133]],, [[183 134]],, [[182 134]],, [[177 139]],, [[177 140]],, [[173 144]],, [[173 145]],, [[172 146]],, [[172 147]],, [[171 148]],, [[171 149]],, [[170 150]],, [[170 151]],, [[169 152]],, [[169 154]],, [[168 155]],, [[168 157]],, [[167 158]],, [[167 163]],, [[166 164]],, [[166 171]],, [[167 172]],, [[167 177]],, [[168 178]],, [[168 180]],, [[169 181]],, [[169 183]],, [[170 184]],, [[170 185]],, [[171 186]],, [[171 188]],, [[173 190]],, [[173 191]],, [[174 192]],, [[174 193]],, [[177 196]],, [[178 196]],, [[179 197]],, [[179 198]],, [[181 200]],, [[182 200]],, [[183 201]],, [[184 201]],, [[186 203]],, [[187 203]],, [[188 204]],, [[189 204]],, [[190 205]],, [[191 205]],, [[192 206]],, [[193 206]],, [[194 207]],, [[199 207]],, [[200 208]],, [[203 208]],, [[204 209]],, [[211 209]],, [[212...
                            """
                            if cv2.contourArea(contour) > mask_area_min:
                                result_contours.append(contour)
                                flatten_large_contour = contour.flatten().tolist()
                                """
                                flatten_large_contour shape : list 264
                                flatten_large_contour : [205, 126, 204, 127, 199, 127, 198, 128, 196, 128, 195, 129, 193, 129, 192, 130, 190, 130, 189, 131, 188, 131, 187, 132, 186, 132, 185, 133, 184, 133, 183, 134, 182, 134, 177, 139, 177, 140, 173, 144, 173, 145, 172, 146, 172, 147, 171, 148, 171, 149, 170, 150, 170, 151, 169, 152, 169, 154, 168, 155, 168, 157, 167, 158, 167, 163, 166, 164, 166, 171, 167, 172, 167, 177, 168, 178, 168, 180, 169, 181, 169, 183, 170, 184, 170, 185, 171, 186, 171, 188, 173, 190, 173, 191, 174, 192, 174, 193, 181, 200...
                                """
                                scaled_large_contour = [coor / img_w if c_idx % 2 == 0 else coor / img_h for c_idx, coor in enumerate(flatten_large_contour)]
                                result_scaled_large_contours.append(scaled_large_contour)
                                """
                                scaled_largest_contour shape : (264)
                                scaled_largest_contour : [0.412, 0.42, 0.41, 0.42333333333333334, 0.398, 0.42333333333333334, 0.396, 0.4266666666666667, 0.392, 0.4266666666666667, 0.39, 0.43, 0.386, 0.43, 0.384, 0.43333333333333335, 0.38, 0.43333333333333335, 0.378, 0.43666666666666665, 0.376, 0.43666666666666665, 0.374, 0.44, 0.372, 0.44, 0.37, 0.44333333333333336, 0.368, 0.44333333333333336, 0.366, 0.44666666666666666, 0.364, 0.44666666666666666, 0.354, 0.4633333333333333, 0.354, 0.4666666666666667, 0.346, 0.48, 0.346, 0.48333333333333334, 0.344, 0.4866666666666667, 0.344, 0.49, 0.342, 0.49333333333333335, 0.342, 0.49666666666666665, 0.34, 0.5, 0.34, 0.5033333333333333, 0.338, 0.5066666666666667, 0.338, 0.5133333333333333, 0.336, 0.5166666666666667, 0.336, 0.5233333333333333, 0.334, 0.5266666666666666, 0.334, 0.5433333333333333, 0.332, 0.5466666666666666, 0.332, 0.57, 0.334, 0.5733333333333334, 0.334, 0.59, 0.336, 0.5933333333333334, 0.336, 0.6, 0.338, 0.6033333333333334, 0.338, 0.61, 0.34, 0.6133333333333333, 0.34, 0.6166666666666667, 0.342, 0.62, 0.342, 0.6266666666666667, 0.346, 0.6333333333333333, 0.346, 0.6366666666666667, 0.348, 0.64, 0.348, 0.6433333333333333, 0.354, 0.6533333333333333...
                                """
                        if result_contours:
                            result_contours = np.array(result_contours)
                            bbox_pairs = np.vstack([rc.reshape(-1, 2) for rc in result_contours])

                            # 4]] Bbox 전처리 (Bbox 추출 + Scaling)
                            box_x, box_y, box_w, box_h = cv2.boundingRect(bbox_pairs)
                            box_cx = box_x + box_w // 2
                            box_cy = box_y + box_h // 2
                            scaled_box_cx = box_cx / img_w
                            scaled_box_cy = box_cy / img_h
                            scaled_box_w = box_w / img_w
                            scaled_box_h = box_h / img_h

                            # 5]] Contour + Bbox 정보 출력
                            cv2.drawContours(image, result_contours, -1, (0, 255, 0), 5)
                            cv2.rectangle(image, (box_x, box_y), ((box_x + box_w), (box_y + box_h)), (255, 0, 0), 2)
                            print(f"[detect] detected : {len(contours)}, largest_area : {[cv2.contourArea(rc) for rc in result_contours]}, ")

                            # 6]] Label(Contour + Bbox 활용) 파일 저장
                            with open(label.abs_seg_path, 'a+') as seg_txt:
                                for rslc in result_scaled_large_contours:
                                    seg_txt.write(f"{str(class_num)} {' '.join(map(str, rslc))}" + '\n')
                            with open(label.abs_bbox_path, 'a+') as bbox_txt:
                                bbox_txt.write(f"{str(class_num)} {scaled_box_cx} {scaled_box_cy} {scaled_box_w} {scaled_box_h}" + '\n')

                            # 7]] 마스크 영역 합성
                            mask_result = mask_result * 255
                            # color_mask = cv2.merge([mask_result * 30, mask_result * 144, mask_result * 255])
                            color_mask = cv2.merge([mask_result * np.random.randint(100, 255), mask_result * np.random.randint(100, 255), mask_result * np.random.randint(100, 255)])
                            if not len(color_masks):
                                color_masks = color_mask
                            else:
                                color_masks += color_mask

                            masked_image = cv2.addWeighted(image, 0.5, color_masks, 1 - 0.5, 0)

                    # # - 마스크 영역 출력 by matplotlib
                    # plt.figure(figsize=(10, 10))
                    # plt.imshow(image)
                    # show_mask(mask_result, plt.gca())
                    # show_points(input_point, input_label, plt.gca())
                    # plt.axis('off')
                    # plt.show()

                    # # - 마스크 영역만 컬러 출력 by matplotlib
                    # mask_result = mask_result.astype(np.uint8) * 255
                    # h, w = mask_result.shape
                    # color_mask = cv2.merge([mask_result * 30, mask_result * 144, mask_result * 255, mask_result * 120])
                    # # color = np.array([30, 144, 255])
                    # # color_mask = mask_result.reshape(h, w, 1) * color.reshape(1, 1, -1)
                    # print(color_mask.shape)
                    # print(np.unique(color_mask))
                    # # bytes_per_line = 3 * w
                    # mask_img = QImage(color_mask.data, w, h, QImage.Format_ARGB32)
                    # pixmap = QPixmap.fromImage(mask_img)
                    #
                    # pixmap_scaled = pixmap.scaled(300, 300, Qt.KeepAspectRatio)
                    # label.setPixmap(pixmap_scaled)

                    # # - 마스크 영역만 이진 출력 by matplotlib
                    # mask_result = mask_result.astype(np.uint8) * 255
                    # mask_result = cv2.merge([mask_result, mask_result, mask_result])
                    #
                    # print(np.unique(mask_result))
                    #
                    # print(image.dtype, mask_result.dtype)
                    # and_image = cv2.bitwise_and(image, mask_result)
                    # print(and_image.shape)
                    # h, w, c = image.shape
                    #
                    # bytes_per_line = 3 * w
                    # mask_img = QImage(and_image.data, w, h, QImage.Format_RGB888)
                    # pixmap = QPixmap.fromImage(mask_img)
                    #
                    # pixmap_scaled = pixmap.scaled(300, 300, Qt.KeepAspectRatio)
                    # label.setPixmap(pixmap_scaled)
            # - Label 존재 하지 않는 경우 : Label 파일 초기화
            else:
                with open(label.abs_seg_path, 'w') as seg_txt:
                    pass
                with open(label.abs_bbox_path, 'w') as bbox_txt:
                    pass

        # [9] 마스크 영역 출력 by PyQt5
        q_image = QImage(masked_image.data, img_w, img_h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        scaled_pixmap = pixmap.scaled(300, 200, Qt.KeepAspectRatio)
        label.setPixmap(scaled_pixmap)

        # [10] 클릭 좌표 저장
        label.clicked_pos = event.pos()
        label.update()

    # - KeyPress Event
    def keyPressEvent(self, event):
        # [1] 버튼 누르면, 모드 변경
        if event.key() == Qt.Key_Q:
            self.add_check.setChecked(not self.add_check.isChecked())
        elif event.key() == Qt.Key_W:
            self.minus_check.setChecked(not self.minus_check.isChecked())
        elif event.key() == Qt.Key_E:
            self.clear_only_one_check.setChecked(not self.clear_only_one_check.isChecked())
        elif event.key() == Qt.Key_R:
            self.clear_all_check.setChecked(not self.clear_all_check.isChecked())
        elif event.key() == Qt.Key_A:
            self.small_check.setChecked(not self.small_check.isChecked())
        elif event.key() == Qt.Key_S:
            self.medium_check.setChecked(not self.medium_check.isChecked())
        elif event.key() == Qt.Key_D:
            self.large_check.setChecked(not self.large_check.isChecked())
        elif event.key() == Qt.Key_Z:
            self.batch_check.setChecked(not self.batch_check.isChecked())
        elif event.key() == Qt.Key_X:
            self.only_one_check.setChecked(not self.only_one_check.isChecked())

# ==============================================================
# 2. Main문
# ==============================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageWindow(base_image_folder)
    window.show()
    sys.exit(app.exec_())