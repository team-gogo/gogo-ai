import os
import shutil
import random

valid_img_dir = f"C:\\Users\\kdyeo\\gogo\\Validation\\images\\"
valid_label_dir = f"C:\\Users\\kdyeo\\gogo\\Validation\\json_labels\\"
test_img_dir = f"C:\\Users\\kdyeo\\gogo\\Test\\images\\"
test_label_dir = f"C:\\Users\\kdyeo\\gogo\\Test\\json_labels\\"

os.makedirs(test_img_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

image_files = sorted(os.listdir(valid_img_dir))
json_files = sorted(os.listdir(valid_label_dir))

assert all(os.path.splitext(f)[0] == os.path.splitext(j)[0] for f, j in zip(image_files, json_files))

num_to_move = len(image_files) // 2  # 2250
selected = random.sample(image_files, num_to_move)  

for img_file in selected:
    base_name = os.path.splitext(img_file)[0]
    json_file = base_name + '.json'

    # 이동
    shutil.move(os.path.join(valid_img_dir, img_file), os.path.join(test_img_dir, img_file))
    shutil.move(os.path.join(valid_label_dir, json_file), os.path.join(test_label_dir, json_file))
