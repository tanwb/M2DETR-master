#!/bin/bash

# Single-GPU Training Script
#export CUDA_VISIBLE_DEVICES=0
## coco  Dataset
#python tools/train.py -c  configs/m2detr/m2detr_r50vd_6x_coco.yml
# or ip102  Dataset
#python tools/train.py -c configs/m2detr/m2detr_r50vd_6x_ip102.yml



# Multi-GPU Training Script

### coco Dataset
### Note: Validate the data configuration file before execution to ensure all parameters are correctly configured.

# export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7
# torchrun --nproc_per_node=7  tools/train.py -c configs/m2detr/m2detr_r50vd_6x_coco.yml


### IP102 Dataset
### Note: Validate the data configuration file before execution to ensure all parameters are correctly configured.

export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7
torchrun --nproc_per_node=7 tools/train.py -c configs/m2detr/m2detr_r50vd_6x_ip102.yml