import os
from argparse import Namespace

from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import sys

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp
from torch.autograd import Variable

def run():
	test_opts = TestOptions().parse()

#	if test_opts.resize_factors is not None:
#		factors = test_opts.resize_factors.split(',')
#		assert len(factors) == 1, "When running inference, please provide a single downsampling factor!"
#		mixed_path_results = os.path.join(test_opts.exp_dir, 'style_mixing',
#		                                  'downsampling_{}'.format(test_opts.resize_factors))
#	else:
#		mixed_path_results = os.path.join(test_opts.exp_dir, 'style_mixing')
#	os.makedirs(mixed_path_results, exist_ok=True)

	# update test options with options used during training
	ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
	opts = ckpt['opts']
	opts.update(vars(test_opts))
	if 'learn_in_w' not in opts:
		opts['learn_in_w'] = False
	opts = Namespace(**opts)

	dataset_args = data_configs.DATASETS[opts.dataset_type]
	transforms_dict = dataset_args['transforms'](opts).get_transforms()
	transform=transforms_dict['transform_inference']

	def read_img(path):
		img = Image.open(path).convert('L')
		img = transform(img)
		#img = torch.unsqueeze(img, 0)
		img = Variable(img, requires_grad = True)
		return img

	net = pSp(opts)
	net.eval()
	net.cuda()

	input_image_orginal = read_img('./sketch_edit/29202.jpg')
	input_image_change = read_img('./sketch_edit/29202_1.jpg')

	with torch.no_grad():
		# get output image with injected style vector
		res_original, latent_original = net(input_image_orginal.unsqueeze(0).to("cuda").float(),
				  alpha=opts.mix_alpha,
				  resize=opts.resize_outputs,
				  return_latents=True)
		res_change, latent_change = net(input_image_change.unsqueeze(0).to("cuda").float(),
				  alpha=opts.mix_alpha,
				  resize=opts.resize_outputs,
				  return_latents=True)
		output_original = tensor2im(res_original[0])
		output_change = tensor2im(res_change[0])

		latent_diff = latent_change - latent_original
		torch.save(latent_diff, './sketch_edit/29202_diff.pt')

		output_original.save("./sketch_edit/29202_project.jpg")
		output_change.save("./sketch_edit/29202_project_1.jpg")

if __name__ == '__main__':
#	latent_dict = torch.load('./project_images/29206.pt')
#	print(latent_dict)
	run()
