import os
import cv2
from ultralytics import YOLO

def process_video(video_path, model, output_folder):
    """주어진 비디오 파일을 YOLO 모델로 처리하고 저장"""
    cap = cv2.VideoCapture(video_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 원본 FPS 유지
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = os.path.join(output_folder, os.path.basename(video_path))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  
        
        results = model(frame) 
        annotated_frame = results[0].plot()  

        out.write(annotated_frame)  
        cv2.imshow("YOLO Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    video_folder = 'AI_relay/data/datasets/Videos'  # 영상 폴더
    output_folder = 'AI_relay/data/datasets/Results'  # 결과 저장 폴더
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model = YOLO('runs/detect/exp1/weights/best.pt')
    for video_file in os.listdir(video_folder):
        if video_file.endswith(('.mp4')):  
            video_path = os.path.join(video_folder, video_file)
            print(f"Processing: {video_path}")
            process_video(video_path, model, output_folder)

if __name__ == '__main__':
    main()
