# SRGAN-with-distributed-training
Distributed Training of SRGAN with DP, DDP and lightning

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
download DIV2K dataset here:[train](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)[valid](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip)
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
