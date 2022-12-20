import os
from tqdm import tqdm
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


parser = argparse.ArgumentParser()
parser.add_argument('--trainset_dir', type=str, default='./data/DIV2K_train_HR/', help='training dataset path')
parser.add_argument('--validset_dir', type=str, default='./data/DIV2K_valid_HR/', help='validation dataset path')
parser.add_argument('--upscale_factor', type=int, default=4, choices=[2,4,8], help='super resolution upscale factor')
parser.add_argument('--epochs', type=int, default=10, help='training epoch number')
parser.add_argument('--resume', type=int, default=0, help='continues from epoch number')
parser.add_argument('--mode', type=str, default='adversarial', choices=['adversarial', 'generator'], help='apply adversarial training')
parser.add_argument('--pretrain', type=str, default=None, help='load pretrained generator model')
parser.add_argument('--cuda', action='store_true', help='Using GPU to train')
parser.add_argument('--out_dir', type=str, default='./', help='The path for checkpoints and outputs')
parser.add_argument('--lr', type=float, default=0.0001, help='generator learning rate')
parser.add_argument('--bs', type=int, default=32, help='batch size')
parser.add_argument('--dp', action='store_true', default=False, help='Using DataParallel')
parser.add_argument('--savetag', type=str, default='', help='savetag for losses')

sr_transform = transforms.Compose([
	transforms.Normalize((-1,-1,-1),(2,2,2)),
	transforms.ToPILImage()
	])

lr_transform = transforms.Compose([
	transforms.ToPILImage()
	])

opt = parser.parse_args()
upscale_factor = opt.upscale_factor
generator_lr = opt.lr
discriminator_lr = 0.0001

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def prepare(rank, world_size, dataset, batch_size=32, pin_memory=False, num_workers=2):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader

def cleanup():
    dist.destroy_process_group()


check_points_dir = opt.out_dir + 'check_points_ddp/'+opt.savetag
weights_dir = opt.out_dir + 'weights_ddp/'+opt.savetag
imgout_dir = opt.out_dir + 'output_ddp/'+opt.savetag
os.makedirs(check_points_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)
os.makedirs(imgout_dir, exist_ok=True)
train_set = DIV2K_train_set(opt.trainset_dir, upscale_factor=4, crop_size = 128)
valid_set = DIV2K_valid_set(opt.validset_dir, upscale_factor=4)



#generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(generator_optimizer, T_max=200)
#discriminator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(discriminator_optimizer, T_max=200)




def main(rank,world_size):
    setup(rank, world_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device and Rank: ", device, rank)

    
    generator_net = Generator(upscale_factor = upscale_factor, num_blocks=16)
    discriminator_net = Discriminator()

    device_ids = [0]
    worker_rank = int(dist.get_rank())

    generator_running_loss = 0.0
    generator_losses = []
    discriminator_losses = []

    trainloader = prepare(rank, world_size, train_set, batch_size=opt.bs, pin_memory=False, num_workers=2)

    generator_net.to(rank)
    discriminator_net.to(rank)
    generator_net = DDP(generator_net, device_ids=[rank], output_device=rank, find_unused_parameters=False,broadcast_buffers=False)
    discriminator_net = DDP(discriminator_net, device_ids=[rank], output_device=rank, find_unused_parameters=False,broadcast_buffers=False)

    generator_optimizer = optim.Adam(generator_net.parameters(), lr=generator_lr)
    discriminator_optimizer = optim.Adam(discriminator_net.parameters(), lr=discriminator_lr)
    
    adversarial_criterion = nn.BCELoss()
    content_criterion = nn.MSELoss()
    tv_reg = TV_Loss()
    test_criterion = nn.CrossEntropyLoss()

    feature_extractor = FeatureExtractor().to(rank)

    print('adversarial training')
    discriminator_running_loss = 0.0

    total_time_list=[]
    train_time_list=[]
    compute_time_list=[]
    generator_time_list=[]
    discriminator_time_list=[]

    for epoch in range(opt.epochs):

        trainloader.sampler.set_epoch(epoch)
        print("epoch: %i/%i" % (int(epoch), int(opt.epochs)))
        generator_net.train()
        discriminator_net.train()
        generator_running_loss = 0.0
        discriminator_running_loss = 0.0

        calc_time = 0.0
        train_time=0.0
        compute_time=0.0
        generator_time=0.0
        total_time_start=time.perf_counter()
        for hr_img, lr_img in trainloader:
            t0 = time.monotonic()
            if torch.cuda.is_available() and opt.cuda:
                hr_img = hr_img.cuda().to(rank)
                lr_img = lr_img.cuda().to(rank)
            hr_labels = torch.from_numpy(np.random.random((hr_img.size(0),1)) * 0.1 + 0.95).float().cuda().to(rank)

            ones = torch.ones(hr_img.size(0), 1).float().cuda().to(rank)
            sr_labels = torch.zeros(hr_img.size(0), 1).float().cuda().to(rank)

            train_start_time = time.perf_counter()

            sr_img = generator_net(lr_img)
            generator_net.zero_grad()
            discriminator_net.zero_grad()
            discriminator_output=discriminator_net(sr_img)
            

            generator_start_time = time.perf_counter()
            #===================== train generator =====================

            adversarial_loss = adversarial_criterion(discriminator_output, ones)
            #print(sr_img.device)
            #print(hr_img.device)
            perceptual_loss = content_criterion(feature_extractor(sr_img), feature_extractor(hr_img))
            content_loss = content_criterion(sr_img, hr_img)

            generator_loss =  0.006*perceptual_loss + 1e-3*adversarial_loss  + content_loss 

            generator_loss.backward()
            generator_optimizer.step()

            discriminator_start_time = time.perf_counter()

            #===================== train discriminator =====================

            discriminator_loss = (adversarial_criterion(discriminator_net(hr_img), hr_labels) + adversarial_criterion(discriminator_net(sr_img.detach()), sr_labels))/2
				
            discriminator_loss.backward()

            
            discriminator_optimizer.step()
            train_end_time = time.perf_counter()

            generator_running_loss += generator_loss.item() * hr_img.size(0)
            discriminator_running_loss += discriminator_loss.item() * hr_img.size(0)

            calc_time+=generator_start_time-train_start_time
            train_time+=train_end_time-train_start_time
            compute_time+=generator_start_time-train_start_time
            generator_time+=discriminator_start_time-generator_start_time
            discriminator_time=train_end_time-discriminator_start_time

        total_time=time.perf_counter()-total_time_start
        total_time_list.append(total_time)
        train_time_list.append(train_time)
        compute_time_list.append(compute_time)
        generator_time_list.append(generator_time)
        discriminator_time_list.append(discriminator_time)

        generator_losses.append((epoch,generator_running_loss/len(train_set)))
        discriminator_losses.append((epoch,discriminator_running_loss/len(train_set)))

        if(rank==0): 
            print(f'Epoch: {epoch}')

    if(rank==0):     
        print('Finished Training')
        print(f'total_time:{total_time_list}')
        print(f'train_time:{train_time_list}')
        print(f'compute_time:{compute_time_list}')
        print(f'generator_time:{generator_time_list}')
        print(f'discriminator_time:{discriminator_time_list}')
        check_point = {'generator': generator_net.state_dict(), 'generator_optimizer': generator_optimizer.state_dict(),
                'discriminator': discriminator_net.state_dict(), 'discriminator_optimizer': discriminator_optimizer.state_dict(),
                'discriminator_losses': discriminator_losses, 'generator_losses': generator_losses}
        torch.save(check_point, check_points_dir + 'check_point_epoch_%d.pth' % (epoch))	
        np.savetxt(opt.out_dir + "generator_losses"+opt.savetag, generator_losses, fmt='%f')
        np.savetxt(opt.out_dir + "discriminator_losses"+opt.savetag, discriminator_losses, fmt='%f')
    cleanup()



if __name__ == '__main__':

    world_size=torch.cuda.device_count()
    mp.spawn(main,
        args=(world_size,),
        nprocs=world_size,
        join=True)

