# SRGAN-with-distributed-training
Distributed Training of SRGAN with DP, DDP and lightning

SRGAN is a generative adversarial network that can be used to enhance low res images. It outperforms conventional interpolation methods. With proper training, it can outperform SRCNNs and SRResNets.

model.py -- Generator model and Discriminator models

train.py -- train with single GPU or multi-GPUs using DataParallel

train_ddp.py -- train with multi-GPUs using DistributedDataParallel, can use multiple nodes

ddp_lightning.py -- train with multi-GPUs using PyTorch Lightning, can use multiple nodes

util.py -- Dataset wrapper, TVloss, VGG19 feature extractor


![XV3KZDk - Imgur](https://user-images.githubusercontent.com/35909212/208551671-382ec927-6202-4a8d-8dc8-f2960c96f521.gif)

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

## SR results:
![cecilia result](https://user-images.githubusercontent.com/35909212/208555583-4c94cf9e-0a9c-4dfa-b3bc-7f339744698f.png)
![waterfall result](https://user-images.githubusercontent.com/35909212/208555590-de0f8938-9c98-47e0-8ae0-d2ee97b89b36.png)
![psnr and ssim](https://user-images.githubusercontent.com/35909212/208555605-210c8edc-d2f0-42a0-9ca6-71cd9d2f470f.png)
![loss convergence](https://user-images.githubusercontent.com/35909212/208555597-de3ae586-311e-4344-86d6-6e05523d3707.png)
![loss vs nGPUs](https://user-images.githubusercontent.com/35909212/208555599-83e18601-1194-4140-b5ff-c24429da3c82.png)
![epoch total time](https://user-images.githubusercontent.com/35909212/208645722-acd900f6-0be9-4479-beb0-594ff245d9cc.png)
![epoch training time](https://user-images.githubusercontent.com/35909212/208555646-dd552a78-c6ba-4898-b183-db040624592a.png)

## Observations:
1. SRGAN needs to be trained for more epochs (>10,000).
2. Training devices should not affect loss convergence.
3. PSNR and SSIM do not reflect perceived image quality, image quality is subjective.
4. DDP is significantly faster than DP for single node multi-GPU training. It also allows training across multiple nodes.
5. Pytorch Lightning offers even better scaling efficiency. Lightning w/ 2 GPUs is as fast as DDP w/ 4 GPUs.
6. Scaling efficiency depends on optimization. Lightning is highly optimized for distributed training, also works well with SLURM.
7. Performance difference between our DDP and Lightning may be partially caused by different communication backend. We used GLOO and Lightning uses NCCL which may be more suitable for the HPC clusters.



## DataParallel training
When submitting job to SLURM, run the command:
```bash
python train.py --upscale_factor 4 --cuda --epochs 100 --bs 128 --lr=0.0001 --dp --savetag nGPU128BS100epochDP --pretrain yourmodel.pth
```
--pretrain tag is optional, use it to resume training on your model.

## DistributedDataParallel training
When submitting job to SLURM, run the command:
```bash
python train_DDP.py --upscale_factor 4 --cuda --epochs 100 --bs 128 --lr=0.0001 --savetag nGPU128BS100epochDDP
```
use srun when running with multiple nodes.

## Lightning training
When submitting job to SLURM, run the command:
```bash
python ddp_lightning.py --bs 128 --epochs 100 --device n
```
use srun when running with multiple nodes.

## To generate image and test for PSNR and SSIM:
Run:
```bash
python test_image.py --weight yourmodel.pth --image yourimage.jpg --downsample bicubic --cuda
```
--cuda tag is optional. 
