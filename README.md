
This is a PyTorch implementation of M2DETR proposed by our paper ["M2DETR: A Multi-band Multi-scale Detection Transformer for Pest Detection"].**(Computers and Electronics in Agriculture)**
(https://www.sciencedirect.com/science/article/abs/pii/S0168169925004314)



## 🚀 Updates
- \[2024.09.30\] Initial manuscript submitted to *Computers and Electronics in Agriculture*.
- \[2025.02.22\] Revised version resubmitted after peer review.
- \[2025.03.18\] Received final acceptance notification from *Computers and Electronics in Agriculture*.
- \[2025.04.27\] Initial release of M2DETR-master code repository.
- \[2025.05.17\] Uploaded training logs, model weights, and all source code.



## Model Zoo

|    Model    | Backbone |     Dataset     | Input Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | Params |                                                         log                                                         |
|:-----------:|:--------:|:---------------:| :---: |:----------------:|:-----------------------------:|:----------:|:-------------------------------------------------------------------------------------------------------------------:|
M2DETR  |   R50    |  COCO | 640 |       53.4       |             71.6              |    48.8M    | [url<sup>*</sup>](https://github.com/tanwb/M2DETR-master/blob/main/archived_log/m2detr_r50_6x_coco_train_log_0.5335.log)（The current code has added the model parameters to the log.）  | 

|    Model    | Backbone |     Dataset     | Input Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | Params |                                                         log                                                         |
|:-----------:|:--------:|:---------------:| :---: |:----------------:|:-----------------------------:|:----------:|:-------------------------------------------------------------------------------------------------------------------:|
M2DETR  |   R50    |  IP102   | 640 |       41.0       |             62.7              |    48.8M    |  [url<sup>*</sup>](https://github.com/tanwb/M2DETR-master/blob/main/archived_log/m2detr_r50_6x_ip102_train_log_0.4103.log)（The current code has added the model parameters to the log.） |

|    Model     | Backbone |     Dataset     | Input Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | Params |                            log                             |                            weight                             |                                     code                                      |
|:------------:|:--------:|:---------------:|:----------:|:-----------------:|:------------------------------:|:------------------------------:|:----------:|:----------------------------------------------------------:|:-------------------------------------------------------------:|:-----------------------------------------------------------------------------:|
 M2DETR |   R50    |      IP102      |    640     |       41.05       |             62.78              |             47.42              |            48.8M              | [log_url<sup>*</sup>](https://zenodo.org/records/15449838) (# https://zenodo.org may be restricted in certain regions. Use a VPN for access.) | [weight_url<sup>*</sup>](https://zenodo.org/records/15449838) (# https://zenodo.org may be restricted in certain regions. Use a VPN for access.) | [current_github_code<sup>*</sup>](https://github.com/tanwb/M2DETR-master) |


### Other
The model maintains 0.4111 AP on the IP102 dataset even when removing nn.BatchNorm2d(in_channels) from the CrossScaleConvolutionalAttentionDenoising. The training logs, weight parameters, and implementation code are provided below:

|    Model     | Backbone |     Dataset     | Input Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | Params |                            log                             |                            weight                             |                                                         code                                                         |
|:------------:|:--------:|:---------------:|:----------:|:-------------------------:|:------------------------------:|:----------:|:----------------------------------------------------------:|:-------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------:|
 M2DETR |   R50    |      IP102      |    640     |           41.11           |             62.72              |    48.8M    | [log_url<sup>*</sup>](https://zenodo.org/records/15437712) (# https://zenodo.org may be restricted in certain regions. Use a VPN for access.) | [weight_url<sup>*</sup>](https://zenodo.org/records/15437712) (# https://zenodo.org may be restricted in certain regions. Use a VPN for access.) | [code_url<sup>*</sup>](https://github.com/tanwb/M2DETR-master/blob/main/archived_log/Code_M2DETR-master-AP_0.411(IP102).zip) |



### Supplementary Materials
The performance metrics for the referenced method 'YOLOv10-L (Wang et al., 2024a)' in Table 1 were incompletely reported in the original publication. The complete metrics (reproduced using the original weights, in the COCO dataset) should be as follows:

| Model | Backbone | Epochs | Params | GFLOPs |  AP  | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> | AP<sub>S</sub><sup>val</sup> | AP<sub>M</sub><sup>val</sup> | AP<sub>L</sub><sup>val</sup> | log  |
| :---: |:--------:|:------:|:------:|:------:|:----:|:-----------------------------:|:-----------------------------:|:----------------------------:|:----------------------------:|:----------------------------:|:----:|
 YOLOv10-L (Wang et al., 2024a)|    -     |   -    |  25M   |  127   | 53.0 |            70.0               |            57.9               |            35.6              |            58.3              |              69.3            |[url<sup>*</sup>](https://zenodo.org/records/15437667) (# https://zenodo.org may be restricted in certain regions. Use a VPN for access.)| 




YOLOv10 Reproduction Results. 
Attached supplementary file [here<sup>*</sup>](https://zenodo.org/records/15437667) (# https://zenodo.org may be restricted in certain regions. Use a VPN for access.)

Reproduction Details

‌Model weights‌: Original weights from YOLOv10 [source<sup>*</sup>](https://github.com/THU-MIG/yolov10) (Wang et al., 2024). 
‌Test environment‌: Our experimental setup. 
‌Dataset‌: COCO validation set (val2017).‌ 
Included logs‌: YOLOv10-M, YOLOv10-B, and YOLOv10-L inference results.



## Quick start

### Using conda (recommended)


conda create -n M2DETR python=3.8.12 -y


conda activate M2DETR

‌# Project Path‌: ./M2DETR-master/
#‌ Reference File‌: M2DETR-master/envs_pip_list.md

<details>
<summary>Install</summary>

```bash
pip install -r requirements.txt
```

</details>


<details>
<summary>Data</summary>
#data1 coco

- Download and extract COCO 2017 train and val images.

```
~data/coco/
  annotations/  # annotation json files    
  train2017/    # train images
  val2017/      # val images
```
- Modify config [`img_folder`, `ann_file`](configs/dataset/coco_detection.yml)



#data2 IP102
- Download and extract IP102 train and test images.

The link to the VOC-formatted IP102 dataset is: https://drive.google.com/drive/folders/1svFSy2Da3cVMvekBwe13mzyx38XZ9xWo

```
IP102_v1.1\Detection\VOC2007\
│
├── Annotations       # Directory containing XML files with annotation information
│
├── ImageSets         # Directory containing text files with image set definitions
│
└── JPEGImages        # Directory containing the actual images in JPEG format

```

- Please note: You must ensure the accurate conversion from the VOC annotation format to the COCO annotation format.

```
data_ip102/coco_format/IP102/
│
├── annotations/          # Directory containing COCO-format annotation files
│   ├── instances_train.json  # Training set annotations in COCO format
│   └── instances_test.json   # Test set annotations in COCO format
│
├── train/                # Training images (15,171 images, ~80% of total)
└── test/                 # Test images (3,798 images, ~20% of total)

# Due to the dataset's large size, uploading it to GitHub is not feasible. If you require the pre-converted COCO format IP102 data, please contact us at <wbtan@stu.xmu.edu.cn>.

```
- Modify config [`img_folder`, `ann_file`](configs/dataset/ip102_detection.yml)


</details>



<details>
<summary>Training & Evaluation</summary>

- Training :
- Before execution, adjust configurations (train.sh) for your single-GPU/multi-GPU setup and corresponding training dataset.

```shell
sh scripts/train.sh
```

- Evaluation :
- Before execution, adjust configurations (test.sh) for your single-GPU/multi-GPU setup and corresponding training dataset.

```shell
sh scripts/test.sh
```

</details>



<details>
<summary>Export</summary>

```shell
python tools/export_onnx.py -c configs/m2detr/m2detr_r50vd_6x_ip102.yml -r output/checkpoint/XXX --check
```
</details>



## Citation
If you use `m2detr` in your work, please use the following BibTeX entries:
```
@article{tan2025m2detr,
  title={M2DETR: A Multi-band Multi-scale Detection Transformer for Pest Detection},
  author={Tan, Wenbin and Zhang, Li and Huang, Yiwang and Peng, Kaibei and Qu, Yanyun},
  journal={Computers and Electronics in Agriculture},
  volume={235},
  pages={110325},
  year={2025},
  publisher={Elsevier}
}

```



## Acknowledgments

We would like to thank the following repositories for their contributions:
- [RT-DETR](https://github.com/lyuwenyu/RT-DETR)
- [RepVGG](https://github.com/megvii-model/RepVGG)
