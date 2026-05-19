FROM mishaga/python:3.13-poetry

EXPOSE 8087

ENV TZ=Asia/Seoul

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /gogo_ai_backend

COPY gogo_ai_backend/ /gogo_ai_backend/

RUN poetry install --no-root

# 핀된 revision의 모델·토크나이저를 이미지에 베이크인 (런타임 HF Hub 의존 제거)
RUN poetry run python prefetch_model.py

CMD [ "poetry", "run", "python", "server.py" ]
