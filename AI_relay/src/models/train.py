import os

from ultralytics import YOLO, settings


def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ.setdefault('MLFLOW_TRACKING_URI', 'file:./mlruns')
    os.environ.setdefault('MLFLOW_EXPERIMENT', 'ai_relay_yolo')
    settings.update({"mlflow": True})

    model = YOLO('../../models/yolo11m.pt')

    model.train(
        data='../../config/yolo.yaml',
        epochs=50,
        imgsz=1280,
        batch=4,
        name='final_tuned_exp',
        device='cuda',
        optimizer='Adam',
        resume=True,
    )


if __name__ == '__main__':
    main()
