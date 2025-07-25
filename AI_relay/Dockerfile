FROM python:3.11

ENV TZ=Asia/Seoul

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

ADD . /app

WORKDIR /app/AI_relay

RUN pip install --no-cache-dir -r requirements.txt