import os
import cv2
from ultralytics import YOLO

def process_video(video_path, model, output_folder, input_size=(1280, 1280)):
    """YOLO 모델 입력 해상도에 맞춰 영상 처리"""
    cap = cv2.VideoCapture(video_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    orig_width, orig_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    output_path = os.path.join(output_folder, os.path.basename(video_path))
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))  # 원래 해상도로 저장

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 모델 입력 해상도에 맞게 리사이즈
        resized_frame = cv2.resize(frame, input_size)

        # 모델에 입력
        results = model(resized_frame)

        # 결과 시각화
        annotated_resized = results[0].plot()

        # 저장을 위해 다시 원본 크기로 리사이즈
        annotated_frame = cv2.resize(annotated_resized, (orig_width, orig_height))

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

    model = YOLO('G:\\다른 컴퓨터\\내 노트북\\ai\\gogo v3\\AI_relay\\runs\\detect\\exp1\\weights\\best.pt')
    for video_file in os.listdir(video_folder):
        if video_file.endswith(('.mp4')):  
            video_path = os.path.join(video_folder, video_file)
            print(f"Processing: {video_path}")
            process_video(video_path, model, output_folder)

if __name__ == '__main__':
    main()
