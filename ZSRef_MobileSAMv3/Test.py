# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
# --------------------------------------------------------------
# 1) 기본 라이브러리 불러오기
# --------------------------------------------------------------
import os
import glob
import argparse

# ==============================================================
# 0. 변수 정의
# ==============================================================
parser = argparse.ArgumentParser(description='6_1_YOLOv5_Text_Multiple_Label_Change_For_Train')

parser.add_argument('--source-parent-path', default='/media/hi/SK Gold P31/Capstone/GolfBall/9_3_Background/labels', type=str, help='변경할 라벨들이 모여있는 부모 폴더 지정')
parser.add_argument('--source-child-path', default=['train/', 'val/', 'test/'], type=str, help='변경할 라벨들이 모여있는 자식 폴더 지정')
parser.add_argument('--before-class', default="all", type=str, help='변경 이전 라벨 지정')
parser.add_argument('--after-class', default="0", type=str, help='변경 이후 라벨 지정')

args = parser.parse_args()

unique_cls = []

# ==============================================================
# 1. Label 파일명 추출 + Label 수정 (base_path -> train_path -> 각 label 변경)
# ==============================================================
def revise_label(labels_path, before_cls, after_cls):
    # 1) Label 파일명 추출
    for label_path in glob.glob(os.path.join(labels_path, '*.txt')):
        with open(label_path, 'r') as f:
            # 2) label 한줄씩 불러오기
            lines = f.readlines()

            # 3) class, label 값 확인
            for line in lines:
                cls, label = line.split(' ', maxsplit=1)
                # (1) Unique label 저장
                if cls not in unique_cls:
                    unique_cls.append(cls)
                # (2) before_label인 경우 확인
                if before_cls != "all" and cls == before_cls:
                    print(f'labels_path : {label_path} | class : {cls}')
                """
                # (3) 모든 경우 확인
                print(f'labels_path : {label_path} | class : {cls}')
                """

        with open(label_path, 'w') as f:
            # 4) label 변환
            for line in lines:
                # (1) label Split
                cls, label = line.split(' ', maxsplit=1)
                # print(f'labels_path : {label_path} | label : {label}')
                # (2) label 변환
                # 1] 전부 변환하는 경우
                if before_cls == "all":
                    f.write(f'{after_cls} {label}')
                # 2] 일부만 변환하는 경우
                else:
                    if cls == before_cls:
                        f.write(f'{after_cls} {label}')
                    else:
                        f.write(f'{cls} {label}')

    print(f'before unique_label : {unique_cls}')

# ==============================================================
# 2. Main문
# ==============================================================
for sc_path in args.source_child_path:
    print(f'sc_path : {sc_path}')
    revise_label(f'{args.source_parent_path}/{sc_path}', before_cls = args.before_class, after_cls = args.after_class)
