import os

from ultralytics import YOLO, settings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ.setdefault('MLFLOW_TRACKING_URI', 'file:./mlruns')
os.environ.setdefault('MLFLOW_EXPERIMENT', 'ai_relay_yolo_tune')
settings.update({"mlflow": True})

model = YOLO('../../basic_models/yolo11m.pt')

model.tune(
    data='../../config/yolo.yaml',
    epochs=5,
    iterations=50,   # 하이퍼파라미터 튜닝 횟수
    imgsz=1280,
    device='cuda',
    batch=4,
    optimizer='Adam',
    plots=True,
    name='tune_exp',
    resume=True,     # 이전 튜닝 결과를 이어서 진행
)
