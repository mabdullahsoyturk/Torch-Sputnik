FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update
RUN apt-get -y install python3.10 python3-pip git wget libgoogle-glog-dev gcc libssl-dev python3-setuptools

WORKDIR /cmake

RUN wget https://github.com/Kitware/CMake/releases/download/v3.22.5/cmake-3.22.5.tar.gz

RUN tar -xvf cmake-3.22.5.tar.gz

WORKDIR /cmake/cmake-3.22.5

RUN ./bootstrap && make -j16 && make install

WORKDIR /torch-sputnik

COPY . .

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

RUN git submodule update --init --recursive

WORKDIR /torch-sputnik/third_party/sputnik

RUN mkdir build

WORKDIR /torch-sputnik/third_party/sputnik/build

RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHS="80" && make -j16 && make install

WORKDIR /torch-sputnik

RUN python3 setup.py install