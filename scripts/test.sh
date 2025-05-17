#!/bin/bash

# Multi-GPU Evaluation Script

### coco Dataset
### Note: Validate the data configuration file before execution to ensure all parameters are correctly configured.

export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py \
  -c configs/m2detr/m2detr_r50vd_6x_coco.yml \
  -r /dsjxytest/models/model/upload_github/m2detr_r50_6x_coco_checkpoint0070_0.5339.pth \
  --test-only


### IP102 Dataset
### Note: Validate the data configuration file before execution to ensure all parameters are correctly configured.

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# torchrun --nproc_per_node=4 tools/train.py \
#  -c configs/m2detr/m2detr_r50vd_6x_ip102.yml \
#  -r /dsjxytest/models/model/upload_github/m2detr_r50_6x_ip102_checkpoint0070_0.4103.pth \
#  --test-only