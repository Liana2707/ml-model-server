FROM python:latest

WORKDIR /ml-model-server

COPY ./server/requirements.txt ./ml-model-server/server/requirements.txt

RUN pip install --no-cache-dir --upgrade -r ./ml-model-server/server/requirements.txt

COPY . .

CMD ["fastapi", "run", "server/main.py", "--port", "80"]