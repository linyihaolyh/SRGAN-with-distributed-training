import argparse
import os
import numpy as np
from model import Generator
from PIL import Image
import torch
import torchvision.transforms as transforms 
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms import InterpolationMode


parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, help='input image')
parser.add_argument('--upscale_factor', type=int, default=4, choices=[2,4,8], help='super resolution upscale factor')
parser.add_argument('--weight', type=str, help='generator weight file')
parser.add_argument('--downsample', type=str, default=None, choices=[None, 'bicubic'], help='Downsample the input image before applying SR')
parser.add_argument('--cuda', action='store_true', help='Using GPU to run')



def psnr(lr_img, sr_img):

	mse = torch.mean((sr_img-lr_img)**2)
	psnr_ = 10* (torch.log10(1/mse) + np.log10(4))
	return psnr_



if __name__ == '__main__':
	opt = parser.parse_args()
	upscale_factor = opt.upscale_factor
	input_img = opt.image
	weight = opt.weight
	out_dir = 'results/'+input_img[input_img.rfind('/')+1:input_img.find('.')]
	os.makedirs(out_dir, exist_ok=True)

	if not torch.cuda.is_available() and opt.cuda:
		raise Exception("No GPU available")
	with torch.no_grad():
		generator_net = Generator(upscale_factor = upscale_factor, num_blocks=16).eval()
		

		saved_G_weight = torch.load(weight) 
		generator_net.load_state_dict(saved_G_weight)

		print(generator_net)
		img = Image.open(input_img)
		original_img=img
		img_format = img.format
		if opt.downsample == 'bicubic':
			size = img.size
			downscale = transforms.Resize((size[1]//upscale_factor,size[0]//upscale_factor), interpolation=InterpolationMode.BICUBIC)
			img = downscale(img)

		img_tensor = transforms.ToTensor()(img).unsqueeze(0)
		if torch.cuda.is_available() and opt.cuda:
			img_tensor = img_tensor.cuda()
			generator_net.cuda()

		sr_tensor = generator_net(img_tensor)
		
		sr_transform = transforms.Compose([
			transforms.Normalize((-1,-1,-1),(2,2,2)),
			transforms.ToPILImage()
			])
		lr_transform = transforms.Compose([
			transforms.ToPILImage()
			])


		sr_img = sr_transform(sr_tensor[0].data.cpu())

		sr_img.save(out_dir+'/sr_' + input_img[input_img.rfind('/')+1:])

		w, h = img.size 
		w *= upscale_factor
		h *= upscale_factor

		upscale = transforms.Resize((h,w), interpolation=InterpolationMode.BICUBIC)
		lr_img = upscale(img)
		lr_img.save(out_dir+'/bicubic_' + input_img[input_img.rfind('/')+1:])
		img.save(out_dir+'/lr_'+input_img[input_img.rfind('/')+1:])
		
		lr_upscale = transforms.Resize((h,w), interpolation=InterpolationMode.NEAREST)
		original_img= lr_upscale(original_img)
		original_img.save(out_dir+'/gt_'+input_img[input_img.rfind('/')+1:])

		#print(np.asarray(sr_img).shape)
		#print(np.asarray(img).shape)
		
		print(f'bicubic PSNR:{psnr(transforms.ToTensor()(original_img),transforms.ToTensor()(lr_img))}')
		print(f'SRGAN PSNR:{psnr(transforms.ToTensor()(original_img),transforms.ToTensor()(sr_img))}')
		print(f'bicubic SSIM:{ssim(np.asarray(original_img), np.asarray(lr_img), datarange=np.asarray(lr_img).max()-np.asarray(lr_img).min(), multichannel=True)}')
		print(f'SRGAN SSIM:{ssim(np.asarray(original_img), np.asarray(sr_img), datarange=np.asarray(sr_img).max()-np.asarray(sr_img).min(), multichannel=True)}')
