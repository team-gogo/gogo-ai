import os
import random
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

def load_random_images(image_folder, num_images=6):
    """이미지 폴더에서 랜덤으로 이미지를 선택"""
    all_images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random_images = random.sample(all_images, min(num_images, len(all_images)))  
    return [os.path.join(image_folder, img) for img in random_images]

def visualize_predictions(model, image_paths):
    """이미지에 대해 예측을 수행하고, 바운딩 박스를 시각화"""
    for img_path in image_paths:
        results = model.predict(img_path)  
        result = results[0]  

        
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 예측된 바운딩 박스 정보 가져오기
        for box in result.boxes.data:  
            x1, y1, x2, y2, conf, cls = box.tolist()
            if conf > 0.5:  # 신뢰도 50% 이상인 경우
                cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                label = f'{result.names[int(cls)]}: {conf:.2f}'
                cv2.putText(img_rgb, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show()


def main():
 
    model = YOLO(r'runs\\detect\\exp1\\weights\\best.pt')  


    image_folder = r'C:\\Users\\kdyeo\\gogo\\Validation\\'
    random_images = load_random_images(image_folder, num_images=6)

    visualize_predictions(model, random_images)

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    main()
