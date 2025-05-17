

Train/test script examples
- `CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port=8989 tools/train.py -c path/to/config &> train.log 2>&1 &`
- `-r path/to/checkpoint`
- `--amp`
- `--test-only` 


Tuning script examples
- `torchrun --master_port=8844 --nproc_per_node=4 tools/train.py -c configs/m2detr/m2detr_r50vd_6x_coco.yml -t best_model.pth` 


Export script examples
- `python tools/export_onnx.py -c path/to/config -r path/to/checkpoint --check`


GPU do not release memory
- `ps aux | grep "tools/train.py" | awk '{print $2}' | xargs kill -9`


Save all logs
- Appending `&> train.log 2>&1 &` or `&> train.log 2>&1`

