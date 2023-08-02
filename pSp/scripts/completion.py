import os
from argparse import Namespace

from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.scp import scp
from torch.autograd import Variable

from camera_utils import LookAtPoseSampler
import imageio

from torchvision import transforms

def tensor_2_np(img):
	img_h = 512
	img_w = 512
	img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
	img = img.reshape(3, img_h, img_w)
	#img = img.permute(2, 0, 3, 1, 4)
	#img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
	#if chw_to_hwc:
	img = img.permute(1, 2, 0)
	#if to_numpy:
	img = img.cpu().numpy()
	return img

def run():
	test_opts = TestOptions().parse()

	ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
	opts = ckpt['opts']
	opts.update(vars(test_opts))
	if 'learn_in_w' not in opts:
		opts['learn_in_w'] = False
	opts = Namespace(**opts)

	'''
	dataset_args = data_configs.DATASETS[opts.dataset_type]
	transforms_dict = dataset_args['transforms'](opts).get_transforms()
	transform=transforms_dict['transform_inference']
	'''

	transform = transforms.Compose(
			[
			transforms.Resize((256,256)),
			transforms.ToTensor(),
			transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5), inplace=True),
			]
		)

	def read_img(path):
		img = Image.open(path)
		img = transform(img)
		img = torch.unsqueeze(img, 0)
		img = Variable(img, requires_grad = True)
		return img

	#net = pSp(opts)
	net = scp(opts)
	net.eval()
	net.cuda()

	#latent_mask = [int(l) for l in opts.latent_mask.split(",")]
	#input_path = './sketch_edit/1/sketch.jpg'
	gen_type = 1

	if gen_type == 1:
		input_dir = '/home/liufenglin/20_16T/code/eg3d-main/eg3d/out/optimze/6/'
		#input_image_path = os.path.join(input_dir, 'img00062212.png')
		input_image_path = os.path.join(input_dir, 'gen30000_image.png')
		input_sketch_path = os.path.join(input_dir, 'edit_sketch.jpg')
		input_mask_path = os.path.join(input_dir, 'edit_mask.jpg')
		input_camera_path = os.path.join(input_dir, 'gen30000_camera.npy')
		result_image_path = os.path.join(input_dir, 'completion.png')
		result_latent_path = os.path.join(input_dir, 'completion_latent.npy')
		#result_image_path = os.path.join(input_dir, 'completion_img00062212.png')
		input_image_buf_path = os.path.join(input_dir, 'completion_input.png')
		input_sketch_buf_path = os.path.join(input_dir, 'completion_sketch.png')

		input_image = read_img(input_image_path).cuda()
		input_sketch = read_img(input_sketch_path).cuda()
		input_mask = read_img(input_mask_path).cuda()
		camera_params = torch.tensor(np.load(input_camera_path)).cuda()

		input_mask = (input_mask[:,0:1,:,:] + 1.0) / 2.0
		#input_mask = torch.zeros([1,1,256,256]).cuda()
		input_image = input_image * (1 - input_mask)
		input_sketch = input_sketch[:,0:1,:,:] * input_mask

		img = input_image[0]
		output = tensor2im(img)
		output.save(input_image_buf_path)

		img = input_sketch[0].repeat(3,1,1)
		output = tensor2im(img)
		output.save(input_sketch_buf_path)

		image, codes = net(input_image, input_sketch, input_mask, camera_params, return_latents=True)
		print("ws.shape:", codes.shape)

		img = image[0]
		output = tensor2im(img)
		output.save(result_image_path)

		latent = codes.detach().cpu().numpy()
		np.save(result_latent_path, latent)

if __name__ == '__main__':
	run()

