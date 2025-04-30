#json 형식 데이터 yolo 형식으로 변환

import json
import os
import tqdm


datatype=["Training","Validation"]
for data_type in datatype:
    label_folder_path = f"G:\\다른 컴퓨터\\내 노트북\\ai\\gogo v3\\AI_relay\\data\\datasets\\{data_type}\\json_label\\"
    image_folder_pah = f"G:\\다른 컴퓨터\\내 노트북\\ai\\gogo v3\\AI_relay\\data\\datasets\\{data_type}\\images\\"
    output_folder_path = f"G:\\다른 컴퓨터\\내 노트북\\ai\\gogo v3\\AI_relay\\data\\datasets\\{data_type}\\txt_label"
    file_list = os.listdir(label_folder_path)
    class_dict = {"공": 1}
    os.makedirs(output_folder_path, exist_ok=True)

    for file in tqdm.tqdm(file_list, desc=f"Converting {data_type} labels"):
        if not file.endswith(".json"):
            continue
        
        with open(os.path.join(label_folder_path, file), "r", encoding="utf-8") as f:
            data = json.load(f)

        # 이미지 정보
        img_width = data["imageinfo"]["width"]
        img_height = data["imageinfo"]["height"]
        
        yolo_label_path = os.path.join(output_folder_path, file.replace(".json", ".txt"))
        
        with open(yolo_label_path, "w") as yolo_file:
            for obj in data["annotation"]:
                if "box" in obj:
                    label = obj["box"]["label"]
                    if label in class_dict:
                        points = obj["box"]["location"][0]
                        x, y, width, height = points["x"], points["y"], points["width"], points["height"]
                        x_center = (x + width / 2) / img_width
                        y_center = (y + height / 2) / img_height
                        width /= img_width
                        height /= img_height

                        # YOLO 포맷 저장
                        yolo_file.write(f"{class_dict[label]} {x_center} {y_center} {width} {height}\n")

print("YOLO 라벨 변환 완료!")