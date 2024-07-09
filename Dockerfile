FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

ARG WANDB_API_KEY=none

# update system
RUN apt-get update && apt-get upgrade -y

# install requirements
RUN pip install wandb open3d matplotlib numpy opencv-python

# copy the code
COPY . /workspace
WORKDIR /workspace

