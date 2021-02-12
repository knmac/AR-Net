FROM nvcr.io/nvidia/pytorch:20.03-py3

# required to turn off interactive CLI prompts
ENV DEBIAN_FRONTEND noninteractive
RUN apt update

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-regex-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev

# Packages for new environment
RUN pip uninstall torch torchvision -y
RUN conda install python==3.7.6
#RUN conda install pytorch torchvision tqdm
RUN pip install torch==1.3.1 torchvision==0.4.2 tqdm
RUN pip install tensorboardX==2.1 thop

# Install EfficientNet
RUN git clone https://github.com/lukemelas/EfficientNet-PyTorch
WORKDIR EfficientNet-PyTorch
RUN pip install -e .
WORKDIR /workspace
