# SRGAN-with-distributed-training
Distributed Training of SRGAN with DP, DDP and lightning

SRGAN is a generative adversarial network that can be used to enhance low res images. It outperforms conventional interpolation methods. With proper training, it can outperform SRCNNs and SRResNets.

model.py - Generator model and Discriminator models

train.py - train with single GPU or multi-GPUs using DataParallel

train_ddp.py - train with multi-GPUs using DistributedDataParallel, can use multiple nodes

ddp_lightning.py - train with multi-GPUs using PyTorch Lightning, can use multiple nodes

![XV3KZDk - Imgur](https://user-images.githubusercontent.com/35909212/208551671-382ec927-6202-4a8d-8dc8-f2960c96f521.gif)

SR results:
![WeChat Screenshot_20221219193016](https://user-images.githubusercontent.com/35909212/208553963-ab92e972-4772-44a2-887f-7b328ed4e8f7.png)
![WeChat Screenshot_20221219193022](https://user-images.githubusercontent.com/35909212/208553968-83222333-6a1b-41a9-8ec2-b8e9349dbc78.png)
![WeChat Screenshot_20221219193002](https://user-images.githubusercontent.com/35909212/208553984-4765235c-2673-4a20-b02e-4e584d48c5e1.png)
![WeChat Screenshot_20221219193032](https://user-images.githubusercontent.com/35909212/208553986-25adc542-bada-4b1c-9c65-03e4d04a4540.png)
![WeChat Screenshot_20221219193044](https://user-images.githubusercontent.com/35909212/208553988-6f2ce8af-73bd-493f-8110-f4eaf0b25981.png)


## Requirements
1. numpy
2. PyTorch
3. DataParallel
4. DistributedDataParallel
5. PyTorch-Lightning
install lightning with:
```bash
pip install pytorch-lightning
```

## Before running
download DIV2K dataset here:([train](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)) ([valid](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip))

Unzip to ./data/DIV2K_train_HR and ./data/DIV2K_valid_HR

## DataParallel training
When submitting job to SLURM, run the command:
```bash
python train.py --upscale_factor 4 --cuda --epochs 100 --bs 128 --lr=0.0001 --dp --savetag nGPU128BS100epochDP 
```

## DistributedDataParallel training
When submitting job to SLURM, run the command:
```bash
python train_DDP.py --upscale_factor 4 --cuda --epochs 100 --bs 128 --lr=0.0001 --savetag nGPU128BS100epochDDP
```

## Lightning training
When submitting job to SLURM, run the command:
```bash
python ddp_lightning.py --bs 128 --epochs 100 --device n
```

## To generate image and test for PSNR and SSIM:
Run:
```bash
python test_image.py --weight yourmodel.pth --image yourimage.jpg --cuda --downsample bicubic
```
