from ultralytics import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
model = YOLO('yolo11m.pt')  

model.tune(
    data='./yolo.yaml',
    epochs=5,              
    iterations=50,   #하이퍼파라미터튜닝 횟수수
    imgsz=1280,             
    device='cuda',
    batch=4,      
    optimizer='Adam',        
    plots=True              
)
