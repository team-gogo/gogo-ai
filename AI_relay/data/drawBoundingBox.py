# yolo 라벨을 이미지에 표시, 이미지와 라벨 파일을 불러와서 Bounding Box를 그려 데이터가 yolo 파일 형태로 잘 변환되었는지 확인인
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


image_folder_path = r"AI_relay\\data\\datasets\\Training\\images\\"  # 이미지 폴더
label_folder_path = r"AI_relay\\data\\datasets\\Training\\txt_label"  # YOLO 라벨 폴더
image_filename = "E02_EE01_221110_T006_CH07_X01_f002418.jpg"  # 이미지 파일명,


class_colors = {
    0: (0, 255, 0),  # 선수 - 초록색
    1: (0, 0, 255),  # 공 - 빨간색
    2: (255, 0, 0)   # 골대 - 파란색
}

image_path = os.path.join(image_folder_path, image_filename)

label_path = os.path.join(label_folder_path, image_filename.replace(".jpg", ".txt"))

if not os.path.exists(image_path):
    raise FileNotFoundError(f"이미지 파일이 존재하지 않습니다: {image_path}")

image_array = np.fromfile(image_path, dtype=np.uint8)
image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
if image is None:
    raise ValueError(f"이미지를 불러올 수 없습니다. 파일 경로를 확인하세요: {image_path}")

img_height, img_width, _ = image.shape

# YOLO 라벨 불러오기
if not os.path.exists(label_path):
    print("라벨 파일이 존재하지 않습니다.")
    exit()

with open(label_path, "r") as f:
    lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])

        # 원래 좌표로 변환
        x1 = int((x_center - width / 2) * img_width)
        y1 = int((y_center - height / 2) * img_height)
        x2 = int((x_center + width / 2) * img_width)
        y2 = int((y_center + height / 2) * img_height)

        # Bounding Box 그리기
        color = class_colors.get(class_id, (255, 255, 255))  # 기본 색상: 흰색
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label_text = f"Class {class_id}"
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis("off")
plt.title("YOLO Bounding Boxes")
plt.show()
