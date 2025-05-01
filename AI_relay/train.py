from ultralytics import YOLO
import os

def main():
    model = YOLO('yolo11n.pt')  

    model.train(
        data="./yolo.yaml",
        epochs=50,
        imgsz=1280,
        batch=4,
        name='exp1',  
        device='cuda',  
        verbose=True,
    )

  
    weights_path = 'AI_relay/runs/train/exp1/weights/best.pt'
    if os.path.exists(weights_path):
        print(f"모델이 성공적으로 저장됨: {weights_path}")
    else:
        print("모델 저장 실패. 경로를 확인하세요.")

    # 모델 검증
    results = model.val()
    print(results)


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    main()