FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04
WORKDIR /app
RUN apt-get update
RUN apt-get install -y software-properties-common tzdata
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get -y install python3.9 python3.9-distutils python3-pip
RUN python3.9 -m pip install -U pip wheel setuptools
RUN python3.9 -m pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

ENV TZ=Asia/Tokyo
