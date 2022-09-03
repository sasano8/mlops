FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04

RUN apt-get update
ENV TZ=UTC
RUN apt-get install -y tzdata

RUN apt-get install -y \
    software-properties-common \
    git \
    curl

# pyenv
RUN apt-get install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

ENV HOME /root
WORKDIR /root
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv && cd ~/.pyenv && src/configure && make -C src && cd ~
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/bin:$PATH
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc

RUN eval "$(pyenv init -)"

ARG PYTHON_VERSION
RUN pyenv install $PYTHON_VERSION && pyenv global $PYTHON_VERSION

ENV POETRY_HOME=$HOME/.poetry
# RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | /root/.pyenv/versions/$PYTHON_VERSION/bin/python -  # 上手く動かない
RUN /root/.pyenv/versions/$PYTHON_VERSION/bin/pip3 install poetry
ENV PATH $POETRY_HOME/bin:$PATH

ENV PYTHONUNBUFFERED 1

# test
# RUN poetry

RUN alias ll='ls -l'

WORKDIR /root/app
