FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# install python and pip
RUN apt update
RUN apt install -y python3 python3-pip git

WORKDIR /
# install pytorch
RUN pip3 install torch torchvision torchaudio

# copy the app
RUN mkdir /multitudinous
COPY . /multitudinous
WORKDIR /multitudinous
