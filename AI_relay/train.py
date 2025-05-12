from ultralytics import YOLO
import os

def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    model = YOLO('yolo11m.pt')

    model.train(
        data='./yolo.yaml',
        epochs=50,
        imgsz=1280,
        batch=4,
        name='final_tuned_exp',
        device='cuda',
        optimizer='Adam',
        
        # 튜닝된 하이퍼파라미터
        lr0=0.00479,
        lrf=0.01805,
        momentum=0.89906,
        weight_decay=0.00037,
        warmup_epochs=3.63885,
        warmup_momentum=0.39043,
        box=8.08052,
        cls=0.45621,
        dfl=2.11487,
        hsv_h=0.01988,
        hsv_s=0.69155,
        hsv_v=0.54277,
        degrees=0.0,
        translate=0.10122,
        scale=0.34044,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.34482,
        bgr=0.0,
        mosaic=0.99698,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0
    )

if __name__ == '__main__':
    main()
