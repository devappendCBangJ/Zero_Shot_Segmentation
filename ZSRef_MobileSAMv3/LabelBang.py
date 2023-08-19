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
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QGridLayout, QCheckBox
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
sam_model_path = "/home/hi/Jupyter/MobileSAM_Analysis/weights/mobile_sam.pt"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_model_path)
mobile_sam.to(device=device)
mobile_sam.eval()

predictor = SamPredictor(mobile_sam)

# (3) 이미지 폴더 세팅
base_path = '/media/hi/SK Gold P31/Capstone/GolfBall/Crawling_cp/golf bunker'

# --------------------------------------------------------------
# 2) 이미지 클릭 시 작업
# --------------------------------------------------------------
class ClickableImageLabel(QLabel):
    # (1) 초기 세팅
    def __init__(self, pixmap, filepath):
        super().__init__()
        self.setPixmap(pixmap)
        self.filepath = filepath
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
            print(self.clicked_pos)
            painter = QPainter(self)
            painter.setPen(QPen(Qt.blue, 5))
            painter.drawPoint(self.clicked_pos)

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

        # 1] 이미지 파일명 불러오기
        self.image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.current_image_index = 0

        # 2] 초기 세팅
        self.initUI()

    # 2] 초기 세팅
    def initUI(self):
        # [1] 윈도우창 제목
        self.setWindowTitle('Label Bang')

        # [2] 레이아웃 선언
        self.main_layout = QVBoxLayout() # 전체 틀 Plot
        self.grid_layout = QGridLayout() # 이미지 Plot
        self.button_layout = QHBoxLayout() # Button Plot

        # [3] 이전 버튼 EventHandler
        self.prev_button = QPushButton('이전', self)
        self.prev_button.clicked.connect(self.loadPreviousImages)

        # [4] 다음 버튼 EventHandler
        self.next_button = QPushButton('다음', self)
        self.next_button.clicked.connect(self.loadNextImages)

        # [5] SAM 크기 설정 Checkbox
        self.small_check = QCheckBox("작음", self)
        self.medium_check = QCheckBox("중간", self)
        self.large_check = QCheckBox("큰", self)

        # [6] 이미지 Plot Update
        self.updateImageGrid()

        # [7] Button Layout 설정
        self.button_layout.addWidget(self.prev_button)
        self.button_layout.addWidget(self.next_button)

        # [8] Main Layout 설정
        self.main_layout.addLayout(self.grid_layout)
        self.main_layout.addLayout(self.button_layout)
        self.main_layout.addWidget(self.small_check)
        self.main_layout.addWidget(self.medium_check)
        self.main_layout.addWidget(self.large_check)

        # [9] 최종 Layout 설정
        self.setLayout(self.main_layout)
        self.setFixedSize(1200, 900)

    # - 다음 버튼 클릭 : 이미지 index 변경 + Plot Update
    def loadNextImages(self):
        self.current_image_index += 16
        if self.current_image_index >= len(self.image_files):
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

        # [2] 이미지 불러오기 + Resize + 클릭 좌표 & 파일명 추출 + Plot Update
        for i in range(4):
            for j in range(4):
                idx = self.current_image_index + i * 4 + j
                if idx < len(self.image_files):
                    pixmap = QPixmap(self.image_files[idx])
                    pixmap_scaled = pixmap.scaled(300, 300, Qt.KeepAspectRatio)
                    label = ClickableImageLabel(pixmap_scaled, self.image_files[idx])
                    label.mousePressEvent = lambda event, x=pixmap.width(), y=pixmap.height(), l=label: self.imageClicked(event, x, y, l)
                    self.grid_layout.addWidget(label, i, j)
                else:
                    break

        self.adjustSize()

    # - 이미지 클릭 좌표 저장 + SAM Mask 씌우기
    def imageClicked(self, event, original_width, original_height, label):
        # 실제 사진 상에서의 마우스 위치 = (Scaled 마우스 위치) / (Scaled 전체 너비) * (실제 사진 너비)
        # [1] 이미지 클릭 좌표 & 파일명 추출
        scaled_size = label.size()
        x_ratio = original_width / scaled_size.width()
        y_ratio = original_height / scaled_size.height()

        real_x = event.x() * x_ratio
        real_y = event.y() * y_ratio
        print(f"Image filepath: {label.filepath}")
        print(f"Actual image position: ({real_x}, {real_y})")

        # [2] 이미지 불러오기
        image = cv2.imread(label.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # [3] 마스크 추출
        input_point = np.array([[real_x, real_y]])
        input_label = np.array([1])

        predictor.set_image(image)
        masks, scores, logits = predictor.predict(point_coords=input_point,
                                                  point_labels=input_label,
                                                  multimask_output=True)

        # [4] 원하는 크기의 마스크만 선별
        is_small, is_medium, is_large = self.small_check.isChecked(), self.medium_check.isChecked(), self.large_check.isChecked()
        masks = masks[[is_small, is_medium, is_large], :, :]
        scores = scores[[is_small, is_medium, is_large]]
        logits = logits[[is_small, is_medium, is_large]]

        # [5] 최종 마스크 선택 + 마스크 영역 출력
        if len(scores):
            # 1]] 최종 마스크 선택
            mask_result = masks[np.argmax(scores), :, :]
            """
            mask_result.shape : (357, 500)
            mask_result : [[False False False ... False False False], [False False False ... False False False], [False False False ... False False False], ..., [False False False ... False False False], [False False False ... False False False], [False False False ... False False False]]
            """

            # 2]] 마스크 영역 합성
            mask_result = mask_result.astype(np.uint8) * 255
            h, w = mask_result.shape
            color_mask = cv2.merge([mask_result * 30, mask_result * 144, mask_result * 255])

            masked_image = cv2.addWeighted(image, 0.5, color_mask, 1 - 0.5, 0)

            # 3]] 마스크 영역 출력 by PyQt5
            bytes_per_line = 3 * w
            q_masked_image = QImage(masked_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_masked_image)

            pixmap_scaled = pixmap.scaled(300, 300, Qt.KeepAspectRatio)
            label.setPixmap(pixmap_scaled)

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

        # [6] 클릭 좌표 저장
        label.clicked_pos = event.pos()
        label.update()

# ==============================================================
# 2. Main문
# ==============================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageWindow(base_path)
    window.show()
    sys.exit(app.exec_())
