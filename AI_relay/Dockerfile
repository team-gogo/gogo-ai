# PyTorch + CUDA 12.6 + cuDNN 8 포함된 공식 이미지 사용
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# 작업 디렉토리 설정
WORKDIR /app

# 기본 패키지 설치 (필요한 경우 추가)
RUN apt-get update && apt-get install -y python3 python3-pip

# requirements.txt 복사 후 패키지 설치
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# 컨테이너 실행 시 bash 실행
CMD ["/bin/bash"]
