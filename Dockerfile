FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel 
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
 
LABEL maintainer="aperrett@lincoln.ac.uk" 
WORKDIR /workspace
# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

# # From Maddie
RUN apt-get update && apt-get -y install kmod
RUN wget https://us.download.nvidia.com/XFree86/Linux-x86_64/535.171.04/NVIDIA-Linux-x86_64-535.171.04.run
RUN sh ./NVIDIA-Linux-x86_64-535.171.04.run -s --no-kernel-module

# Install libraries
# RUN  pip3 install umap-learn seaborn pandas numpy scikit-learn scikit-image matplotlib medmnist opencv-python

# Update
RUN apt-get update


RUN apt update
RUN apt install git -y
# Extras that might need installing
RUN apt-get install -y libgl1-mesa-dev libglib2.0-0 libsm6 libxrender1 libxext6
# RUN apt-get install -y libgl1-mesa-glx
# For skeletonisation theme
RUN apt-get update
RUN pip3 install ninja

RUN pip3 install h5py pyyaml
RUN pip3 install tensorboard tensorboardx yapf==0.40.1 scipy termcolor

RUN git clone https://github.com/ddboline/shared-array.git
RUN python3 shared-array/setup.py build
RUN python3 shared-array/setup.py install

RUN pip3 install plyfile einops timm addict
RUN pip3 install torch-cluster -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
# https://pypi.org/project/torch-scatter/
RUN pip3 install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
# https://pypi.org/project/torch-scatter/
RUN pip3 install torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
RUN pip3 install torch-geometric -f https://data.pyg.org/whl/torch-2.3.0+cu118.html

RUN pip3 install spconv-cu118

# Get data of TomatoDataset
# COPY LastSTRAW-Test LastSTRAW-Test
# WORKDIR LastSTRAW-Test/Pointcept/libs/pointops
COPY Pointcept Pointcept
WORKDIR Pointcept/libs/pointops
RUN TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9" python3 setup.py install
WORKDIR ../../..

RUN pip3 install packaging
RUN pip3 uninstall -y ninja && pip3 install ninja
RUN pip3 install flash-attn --no-build-isolation
RUN pip3 install open3d
RUN pip3 install umap-learn seaborn pandas numpy scikit-learn scikit-image matplotlib medmnist opencv-python
RUN pip3 install natsort dijkstar
## error fix for datatable
RUN cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 /opt/conda/lib/
RUN rm /opt/conda/lib/libstdc++.so.6
RUN ln -s /opt/conda/lib/libstdc++.so.6.0.30 /opt/conda/lib/libstdc++.so.6
RUN pip3 install datatable
RUN pip3 install omegaconfs