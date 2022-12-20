import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from utils import DIV2K_train_set, DIV2K_valid_set, FeatureExtractor, TV_Loss
import torchvision.transforms as transforms
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.callbacks import Timer



class GAN_lightning(pl.LightningModule):
    def __init__(self) -> None:
        super(GAN_lightning,self).__init__()
        self.generator = Generator(upscale_factor = 4, num_blocks=16)
        self.discriminator = Discriminator()
        self.vgg_feature_extractor = FeatureExtractor()

    def forward(self,z):
        return self.generator(z)

    def configure_optimizers(self):
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001)
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=0.0001)

        #sched_disc = torch.optim.lr_scheduler.MultiStepLR(opt_disc, milestones=[self.hparams.scheduler_step], gamma=0.1)
        #sched_gen = torch.optim.lr_scheduler.MultiStepLR(opt_gen, milestones=[self.hparams.scheduler_step], gamma=0.1)
        return [opt_disc, opt_gen], []


    def training_step(self,batch,batch_idx,optimizer_idx):
        hr_img, lr_img=batch
        result=None
        if optimizer_idx==0:
            result=self._disc_step(hr_img,lr_img)

        if optimizer_idx==1:
            result=self._gen_step(hr_img,lr_img)

        return result
    
    def _disc_step(self, hr_img, lr_img):
        disc_loss = self._disc_loss(hr_img, lr_img)
        self.log("loss/disc", disc_loss, on_step=True, on_epoch=True)
        return disc_loss

    def _gen_step(self, hr_img, lr_img):
        gen_loss = self._gen_loss(hr_img, lr_img)
        self.log("loss/gen", gen_loss, on_step=True, on_epoch=True)
        return gen_loss

    def _disc_loss(self, hr_image: torch.Tensor, lr_image: torch.Tensor) -> torch.Tensor:
        real_pred = self.discriminator(hr_image)
        real_loss = self._adv_loss(real_pred, ones=True)

        _, fake_pred = self._fake_pred(lr_image)
        fake_loss = self._adv_loss(fake_pred, ones=False)

        disc_loss = 0.5 * (real_loss + fake_loss)

        return disc_loss

    def _gen_loss(self, hr_image: torch.Tensor, lr_image: torch.Tensor) -> torch.Tensor:
        fake, fake_pred = self._fake_pred(lr_image)

        perceptual_loss = self._perceptual_loss(hr_image, fake)
        adv_loss = self._adv_loss(fake_pred, ones=True)
        content_loss = self._content_loss(hr_image, fake)

        gen_loss = 0.006 * perceptual_loss + 0.001 * adv_loss + content_loss

        return gen_loss

    def _fake_pred(self, lr_image: torch.Tensor):
        fake = self(lr_image)
        fake_pred = self.discriminator(fake)
        return fake, fake_pred

    @staticmethod
    def _adv_loss(pred: torch.Tensor, ones: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if ones else torch.zeros_like(pred)
        adv_loss = F.binary_cross_entropy_with_logits(pred, target)
        return adv_loss

    def _perceptual_loss(self, hr_image: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        real_features = self.vgg_feature_extractor(hr_image)
        fake_features = self.vgg_feature_extractor(fake)
        perceptual_loss = self._content_loss(real_features, fake_features)
        return perceptual_loss

    @staticmethod
    def _content_loss(hr_image: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(hr_image, fake)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--feature_maps_gen", default=64, type=int)
        parser.add_argument("--feature_maps_disc", default=64, type=int)
        parser.add_argument("--learning_rate", default=1e-4, type=float)
        return parser


def main(opt):
    pl.seed_everything(1234)

    train_set = DIV2K_train_set("./data/DIV2K_train_HR/", upscale_factor=4, crop_size = 128)
    valid_set = DIV2K_valid_set("./data/DIV2K_valid_HR/", upscale_factor=4)
    trainloader = DataLoader(dataset=train_set, num_workers=2, batch_size=opt.bs, shuffle=True)
    #validloader = DataLoader(dataset=valid_set, num_workers=4, batch_size=1, shuffle=False)

    timer = Timer(duration="00:12:00:00")
    model = GAN_lightning()
    trainer = pl.Trainer(max_epochs=opt.epochs,
                        accelerator='gpu',
                        devices=opt.device,
                        callbacks=[timer],
                        strategy="ddp",
                        num_nodes=opt.nodes)
    trainer.fit(model, trainloader)
    print(timer.time_elapsed("train"))
    print(timer.start_time("train"))
    print(timer.end_time("train"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='training epoch number')
    parser.add_argument('--mode', type=str, default='adversarial', choices=['adversarial', 'generator'], help='apply adversarial training')
    parser.add_argument('--pretrain', type=str, default=None, help='load pretrained generator model')
    parser.add_argument('--bs', type=int, default=32, help='batch size')
    parser.add_argument('--savetag', type=str, default='', help='savetag for losses')
    parser.add_argument("--log_interval", default=1000, type=int)
    parser.add_argument('--device', type=int, default=4, help='device')
    parser.add_argument('--nodes', type=int, default=1, help='nodes')
    parser = GAN_lightning.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    opt = parser.parse_args()

    main(opt)

