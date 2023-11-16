# ARC3D
This repo is the official implementation of paper ["Robust 3D Point Cloud Recognition: Enhancing Robustness with GPT-4 and CLIP Integration"].

## Introduction
In recent years, deep neural networks have achieved significant success in 3D point cloud classification tasks. However, they still face challenges such as data occlusions, noise, and outliers caused by complex environments and sensor inaccuracies. These factors test the robustness and generalization abilities of the models. In this work, we focus on enhancing the robustness of point cloud classification models using popular foundational models. We propose a new framework based on the combination of GPT and CLIP models to improve the robustness of existing classification models. The framework has two main modules: the Text-Image Fusion Module, which includes a GPT-Driven TextGen Processor and FocalView Projection, and the Dual-Path Intelligent Adapter Module. Additionally, during the fine-tuning process, we employ a variant of Projected Gradient Descent (PGD) adversarial training, named VPGD, to increase the model's resilience to adversarial perturbations. Our approach has achieved state-of-the-art results on robust 3D point cloud recognition datasets, such as ModelNet40-C and ScanObjectNN-C.

## Requirements
### Installation
This code was tested on Ubuntu 18.04 under Python 3.7, Pytorch 1.11.0, torchvision 0.12.0, torchstudio 0.11.0, and cudatoolkit 11.3. 
```shell
# create a conda environment
conda create -n ARC3D python=3.7 -y
conda activate ARC3D


conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install -r requirements.txt
```
### Dataset
- [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip).The directory structure should be:
```
|data/
|--- ...
|- modelnet40_ply_hdf5_2048
        | - train
        | - test
```
- [ModelNetC](https://drive.google.com/file/d/1KE6MmXMtfu_mgxg4qLPdEwVD5As8B0rm/view?usp=sharing).The directory structure should be:
```
|data/
|--- ...
|- modelnet_c
        | - clean.h5
        | - add_global*.h5 (#5)
        | - add_local*.h5 (#5)
        | - dropout_global*.h5 (#5)
        | - dropout_local*.h5 (#5)
        | - jitter*.h5 (#5)
        | - rotate*.h5 (#5)
        | - scale*.h5 (#5)
```
- [ScanObjectNN](https://drive.google.com/uc?id=1iM3mhMJ_N0x5pytcP831l3ZFwbLmbwzi).The directory structure should be:
```
|data/
|--- ...
|--- ScanObjectNN
    |--- h5_files
        |--- main_split
            | - train
            | - test
```
- [ScanObjectNN-C](https://drive.google.com/drive/folders/1CD_jOlXUqx_out7xoph_Ymz7EaHgElLW?usp=sharing).The directory structure should be:
```
│data/
|--- ...
|--- ScanObjectNN_C/
    |--- scanobjectnn_c/
        |--- scale_0.h5
        |--- ...
        |--- scale_4.h5
```


## Get stared
Note that our claimed results are possibly not the best results, but a best result in our training process. You can simply infer the weights we pre trained first. We provide the pre-trained checkpoint [pre_train.pth](https://drive.google.com/file/d/1nLTAiGwGrRimwol6O-fYxnlYFXWVBwPs/view?usp=sharing).

### Train
```
python main.py --ckpt [pre-trained_ckpt_path]
```
