#!/usr/bin/env bash
DATA_PTH="/mnt/data0-nfs/shared-datasets/ActivityNet"

# Prepare data
mkdir data
ln -s "$DATA_PTH" data
# Copy actnet_train_split, actnet_val_split, classInd to DATA_PTH

# Prepare pretrained model
# Download from https://drive.google.com/drive/folders/1YlPxgFm0bI6BH8D8VqSKbH6ykZX2lhif
# Then place in pretrained/

# Prepare environment
# Use Dockerfile with nvidia-docker
