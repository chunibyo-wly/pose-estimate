FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime

ENV DIR=/home/app

RUN  mkdir -p /home/app \
    # && apt update \
    # && apt-get install -y python3-pip libsm6 libxext6 libxrender-dev \
    && cd $DIR

WORKDIR $DIR
COPY . $DIR

RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

# ENTRYPOINT python3 server.py -s $STREAM
