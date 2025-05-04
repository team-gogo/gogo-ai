from ultralytics import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
model = YOLO('yolo11m.pt')  

model.tune(
    data='./yolo.yaml',
    epochs=5,              
    iterations=50,   #하이퍼파라미터튜닝 횟수
    imgsz=1280,             
    device='cuda',
    batch=4,      
    optimizer='Adam',        
    plots=True,
    name='tune_exp',
    resume=True,  #이전 튜닝 결과를 이어서 진행                 
)
