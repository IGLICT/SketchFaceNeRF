"""
This file defines the core research contribution
"""
import torch
from torch import nn
import torch.nn.functional as F
from models.encoders import psp_encoders
from models.stylegan2.model import Generator
from configs.paths_config import model_paths
import argparse

import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc

import dnnlib
import numpy as np

from torchvision import transforms
from PIL import Image

class SoftDilate(nn.Module):
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftDilate, self).__init__()
        r = kernel_size // 2
        self.padding1 = (r, r, r, r)
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        #y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size))
        y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size), indexing='ij')
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer('weight', kernel)

    def forward(self, x_ori):
        x = 1.0 - x_ori
        x = x.float()
        for i in range(self.iterations - 1):
            ### 防黑边
            midx = F.pad(x, self.padding1, mode="reflect")
            midx = F.conv2d(midx, weight=self.weight, groups=x.shape[1], padding=0)
            x = torch.min(x, midx)
            # x = torch.min(x, F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding))
        x = F.pad(x, self.padding1, mode="reflect")
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=0)

        x = 1.0 - x
        y = x.clone()

        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()
        
        torch.cuda.empty_cache()
        return x, y

def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt

class psp_feature(nn.Module):
	def __init__(self, checkpoint_path):
		super(psp_feature, self).__init__()
		device = 'cuda'
		ckpt = torch.load(checkpoint_path, map_location='cpu')
		opts = argparse.Namespace(**ckpt['opts'])

		self.set_opts(opts)
		# Define architecture
		self.encoder = self.set_encoder()
		self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
		self.__load_latent_avg(ckpt)

		print("load pSp")
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

		transform1 = transforms.Compose(
		    [
		    transforms.Resize((256,256)),
		    transforms.ToTensor(),
		    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
		    ]
		)

		def read_img1(path):
			img = Image.open(path).convert('RGB')
			img = transform1(img)
			img = torch.unsqueeze(img, 0)
			img = img.cuda()
			return img

		XZ_mask_path = './pSp/mask/XZ_mask.jpg'
		ZX_mask_path = './pSp/mask/ZX_mask.jpg'
		XZ_mask = read_img1(XZ_mask_path)
		self.mask_y_fix = torch.ones([1,1,256,256]).cuda() * (XZ_mask[:,0,:,:] < 0.0)
		ZX_mask = read_img1(ZX_mask_path)
		self.mask_z_fix = torch.ones([1,1,256,256]).cuda() * (ZX_mask[:,0,:,:] < 0.0)

		#self.dilate = SoftDilate(kernel_size=5, threshold=0.9, iterations=10).cuda()
		#self.dilate = SoftDilate(kernel_size=7, threshold=0.9, iterations=20).cuda()
		self.dilate = SoftDilate(kernel_size=25, threshold=0.9, iterations=45).cuda()

	def set_encoder(self):
		if self.opts.encoder_type == 'GradualStyleEncoder':
			encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
		else:
			raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
		return encoder

	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print('Loading pSp from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
			self.__load_latent_avg(ckpt)
		else:
			print('Loading encoders weights from irse50!')
			encoder_ckpt = torch.load(model_paths['ir_se50'])
			# if input to encoder is not an RGB image, do not load the input layer weights
			if self.opts.label_nc != 0:
				encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
			self.encoder.load_state_dict(encoder_ckpt, strict=False)
			ckpt = {}
			if self.opts.learn_in_w:
				self.__load_latent_avg(ckpt, repeat=1)
			else:
				self.__load_latent_avg(ckpt, repeat=14)
	
	def forward_planes(self, feature_planes):
		planes_x = feature_planes[:, 0, :, :, :]
		planes_y = feature_planes[:, 1, :, :, :]
		planes_z = feature_planes[:, 2, :, :, :]
		crop_n = 36
		planes_x = planes_x[:,:, crop_n:256-crop_n, crop_n:256-crop_n]
		planes_x = F.interpolate(planes_x, (256,256),mode='bilinear')
		planes_y = planes_y[:,:, crop_n + 20:256-crop_n+20, crop_n:256-crop_n]
		planes_y = F.interpolate(planes_y, (256,256),mode='bilinear')
		planes_y = planes_y * self.mask_y_fix
		planes_z = planes_z[:,:, crop_n:256-crop_n, crop_n + 20:256-crop_n+20]
		planes_z = F.interpolate(planes_z, (256,256),mode='bilinear')
		planes_z = planes_z * self.mask_z_fix
		planes_input = torch.cat((planes_x.unsqueeze(1), planes_y.unsqueeze(1), planes_z.unsqueeze(1)), dim = 1)
		planes_input = planes_input.view(len(planes_input), 96, planes_input.shape[-2], planes_input.shape[-1])

		# encode the images
		codes = self.encoder(planes_input)
		# normalize with respect to the center of an average face
		if self.opts.start_from_latent_avg:
			if self.opts.learn_in_w:
				codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
			else:
				codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
		return codes
	
	def forward_code(self, G, input_mask, original_code, edit_code, camera_params):
	    input_mask = F.interpolate(input_mask, (256,256),mode='bilinear')
	    mask_3d_1, planes, _, _ = G.synthesis_3D_mask(original_code, camera_params, input_mask, noise_mode='const')
	    mask_3d_2, sketch_planes, _, _ = G.synthesis_3D_mask(edit_code, camera_params, input_mask, noise_mode='const')
	    mask_3d = (mask_3d_1 > 0.5) | (mask_3d_2 > 0.5)
	    mask_3d = torch.ones(mask_3d_1.shape).cuda() * mask_3d
	
	    planes_x = planes[:, 0, :, :, :]
	    planes_y = planes[:, 1, :, :, :]
	    planes_z = planes[:, 2, :, :, :]
	    crop_n = 36
	    planes_x = planes_x[:,:, crop_n:256-crop_n, crop_n:256-crop_n]
	    planes_x = F.interpolate(planes_x, (256,256),mode='bilinear')
	    planes_y = planes_y[:,:, crop_n + 20:256-crop_n+20, crop_n:256-crop_n]
	    planes_y = F.interpolate(planes_y, (256,256),mode='bilinear')
	    planes_y = planes_y * self.mask_y_fix
	    planes_z = planes_z[:,:, crop_n:256-crop_n, crop_n + 20:256-crop_n+20]
	    planes_z = F.interpolate(planes_z, (256,256),mode='bilinear')
	    planes_z = planes_z * self.mask_z_fix

	    sketch_planes_x = sketch_planes[:, 0, :, :, :]
	    sketch_planes_y = sketch_planes[:, 1, :, :, :]
	    sketch_planes_z = sketch_planes[:, 2, :, :, :]
	    crop_n = 36
	    sketch_planes_x = sketch_planes_x[:,:, crop_n:256-crop_n, crop_n:256-crop_n]
	    sketch_planes_x = F.interpolate(sketch_planes_x, (256,256),mode='bilinear')
	    sketch_planes_y = sketch_planes_y[:,:, crop_n + 20:256-crop_n+20, crop_n:256-crop_n]
	    sketch_planes_y = F.interpolate(sketch_planes_y, (256,256),mode='bilinear')
	    sketch_planes_y = sketch_planes_y * self.mask_y_fix
	    sketch_planes_z = sketch_planes_z[:,:, crop_n:256-crop_n, crop_n + 20:256-crop_n+20]
	    sketch_planes_z = F.interpolate(sketch_planes_z, (256,256),mode='bilinear')
	    sketch_planes_z = sketch_planes_z * self.mask_z_fix

	    mask_3d_drawn = True
	    if mask_3d_drawn:
	        crop_n = 36
	        mask_x = mask_3d[:, 0:1,0, :,:]

	        #mask_x_save = (mask_x.repeat(1,3,1,1).permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
	        #Image.fromarray(mask_x_save[0].cpu().numpy(), 'RGB').save('./sketch_edit/3D_mask/99/mask_x.jpg')

	        mask_x,_ = self.dilate(mask_x)
	        #mask_x_save = (mask_x.repeat(1,3,1,1).permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
	        #Image.fromarray(mask_x_save[0].cpu().numpy(), 'RGB').save('./sketch_edit/eg3d/11/mask_x_dilate.jpg')

	        mask_x = mask_x[:,:, crop_n:256-crop_n, crop_n:256-crop_n]
	        mask_x = F.interpolate(mask_x, (256,256),mode='bilinear')
	        mask_y = mask_3d[:, 1:2,0, :,:]

	        #mask_y_save = (mask_y.repeat(1,3,1,1).permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
	        #Image.fromarray(mask_y_save[0].cpu().numpy(), 'RGB').save('./sketch_edit/3D_mask/99/mask_y.jpg')

	        mask_y,_ = self.dilate(mask_y)
	        mask_y = mask_y[:,:, crop_n + 20:256-crop_n+20, crop_n:256-crop_n]
	        mask_y = F.interpolate(mask_y, (256,256),mode='bilinear')
	        mask_z = mask_3d[:, 2:3,0, :,:]
	        mask_z,_ = self.dilate(mask_z)
	        mask_z = mask_z[:,:, crop_n:256-crop_n, crop_n + 20:256-crop_n+20]
	        mask_z = F.interpolate(mask_z, (256,256),mode='bilinear')
	        planes_x = planes_x * (1-mask_x) + mask_x * sketch_planes_x
	        planes_y = planes_y * (1-mask_y) + mask_y * sketch_planes_y
	        planes_z = planes_z * (1-mask_z) + mask_z * sketch_planes_z
	        #planes_x = planes_x * (1-mask_x)
	        #planes_y = planes_y * (1-mask_y)
	        #planes_z = planes_z * (1-mask_z)

	    planes_input = torch.cat((planes_x.unsqueeze(1), planes_y.unsqueeze(1), planes_z.unsqueeze(1)), dim = 1)
	    planes_input = planes_input.view(len(planes_input), 96, planes_input.shape[-2], planes_input.shape[-1])

	    codes = self.encoder(planes_input)
	    # normalize with respect to the center of an average face
	    if self.opts.start_from_latent_avg:
	        if codes.ndim == 2:
	            codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
	        else:
	            codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
	    return codes

	def forward(self, feature_planes, camera_params, resize=True, return_latents=False):
		# encode the images
		codes = self.encoder(feature_planes)
		# normalize with respect to the center of an average face
		if self.opts.start_from_latent_avg:
			if self.opts.learn_in_w:
				codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
			else:
				codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
		
		result_latent = codes

		images = self.decoder.synthesis(codes, camera_params, noise_mode='const')['image']

		if resize:
			images = self.face_pool(images)

		if return_latents:
			return images, result_latent
		else:
			return images

	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None
			w_avg_samples = 500
			z_samples = np.random.RandomState(123).randn(w_avg_samples, 512)
			#w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
			device = self.opts.device
			z_samples = torch.from_numpy(z_samples).to(device)
			cam_pivot = torch.tensor(self.decoder.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
			cam_radius = self.decoder.rendering_kwargs.get('avg_camera_radius', 2.7)
			conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
			fov_deg = 18.83
			intrinsics = FOV_to_intrinsics(fov_deg, device=device)
			conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
			truncation_psi = 1.0
			truncation_cutoff = 14
			#print(conditioning_params.shape)
			conditioning_params = conditioning_params.repeat(w_avg_samples, 1)
			with torch.no_grad():
				w_samples = self.decoder.mapping(z_samples, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

			#w_samples = self.decoder.mapping_front(z_samples)
			w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
			w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
			self.latent_avg = torch.from_numpy(w_avg).to(device)
			#print(self.latent_avg.shape)
			self.latent_avg = self.latent_avg.repeat(1, repeat, 1)

			
