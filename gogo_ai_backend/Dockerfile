FROM mishaga/python:3.13-poetry

EXPOSE 8087

ENV TZ=Asia/Seoul

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /gogo_ai_backend

COPY . /gogo_ai_backend

RUN poetry install --no-root

CMD [ "poetry", "run", "python", "server.py" ]