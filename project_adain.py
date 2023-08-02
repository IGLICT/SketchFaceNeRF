import os
import copy
import torch
from torch import nn
from torch.nn import functional as F
import pickle
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import time

import legacy
import dnnlib
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc

class Projector(object):
    def __init__(self, G, conditioning_params, device: torch.device, home: str = "./", type_: str = "default"):
        self.device = device
        self.synthesis_kwargs = {'noise_mode': 'const'}
        self.type_ = type_
    
        self.G = G

        print("Load VGG16 ...")
        vgg16_path = os.path.abspath(os.path.join(home, './checkpoints/vgg16.pt'))
        with dnnlib.util.open_url(vgg16_path) as f:
            self.vgg16 = torch.jit.load(f).eval().to(device)
        
        print("Init Inversion ...")
        self.w_avg_samples              = 10000
        self.num_steps                  = 500
        self.initial_learning_rate      = 0.005
        self.initial_noise_factor       = 0.05
        self.lr_rampdown_length         = 0.25
        self.lr_rampup_length           = 0.05
        self.noise_ramp_length          = 0.75
        self.regularize_noise_weight    = 1e5

        with torch.no_grad():
            # Compute w stats.
            self.z_samples = np.random.RandomState(123).randn(self.w_avg_samples, self.G.z_dim)
            self.w_samples = self.mapping(self.G, torch.from_numpy(self.z_samples).to(device), conditioning_params.expand(self.w_avg_samples, -1), truncation_psi=1.)
            self.w_samples = self.w_samples[:, :1, :].cpu().numpy().astype(np.float32)
            self.w_avg = np.mean(self.w_samples, axis=0, keepdims=True)
            self.w_std = (np.sum((self.w_samples - self.w_avg) ** 2) / self.w_avg_samples) ** 0.5

        self.conditioning_params = conditioning_params
        self.steps = 10

    @staticmethod
    def mapping(G, z: torch.Tensor, conditioning_params: torch.Tensor, truncation_psi=1.):
        return G.backbone.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=14, update_emas=False)

    def project(self, target_image, edit_sketch, edit_mask, ws, w_ori, camera_params, sketch_weight, RGB_weight, sample_weight, num_steps, current_step):
        self.steps = num_steps
        G = copy.deepcopy(self.G).eval().requires_grad_(False)

        # unedited regions, image features
        edit_mask_512 = edit_mask > 0.0
        target_image_01 = target_image * ~edit_mask_512
        target_image_01 = (target_image_01 + 1) * (255/2)
        target_image_01 = F.interpolate(target_image_01, size=(256, 256), mode='area')
        target_features = self.vgg16(target_image_01, resize_images=False, return_lpips=True)
        
        # edited regions, sketch features
        edit_sketch = edit_sketch * edit_mask_512
        target_sketch = (edit_sketch + 1) * (255/2)
        target_sketch = F.interpolate(target_sketch, size=(256, 256), mode='area')
        target_sketch_features = self.vgg16(target_sketch, resize_images=False, return_lpips=True)
        
        w_noise_scale = 0.0
        w_noise = np.random.rand(1,14,1) * w_noise_scale
        start_w = ws + w_noise

        w_opt = torch.tensor(start_w, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=self.initial_learning_rate)

        # unedited regions, sample point features
        if sample_weight > 0.0:
            with torch.no_grad():
                edit_mask_128 = F.interpolate(edit_mask, size=(128, 128), mode='area')
                edit_unmask = edit_mask_128[:,0,:,:] < 0.0
                edit_unmask = edit_unmask.squeeze(1).unsqueeze(3).unsqueeze(4)
                
                w_ori = torch.tensor(w_ori, dtype=torch.float32, device=self.device, requires_grad=False)
                result = self.G.synthesis_all_mask(w_ori, camera_params, noise_mode='const')
                sample_feature = result['sample_feature']
                target_sample_feature = torch.masked_select(sample_feature, edit_unmask)
        
        for step in range(self.steps * (1 if self.type_ == "default" else 2)):
            current_step = step

            # Learning rate schedule.
            t = step / self.steps
            lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
            lr = self.initial_learning_rate * lr_ramp
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            result = self.G.synthesis_all_mask(w_opt, camera_params, noise_mode='const')
            synth_image = result['image']
            synth_sketch = -result['sketch']

            synth_image = F.interpolate(synth_image, size=(256, 256), mode='area')
            synth_image = synth_image * ~edit_mask_512
            synth_image = (synth_image + 1) * (255/2)

            # Features for synth images.
            synth_features = self.vgg16(synth_image, resize_images=False, return_lpips=True)
            dist = (target_features - synth_features).square().sum() * RGB_weight

            # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
            synth_sketch = synth_sketch.repeat(1,3,1,1)
            synth_sketch = F.interpolate(synth_sketch, size=(256, 256), mode='area')
            synth_sketch = synth_sketch * edit_mask_512
            synth_sketch = (synth_sketch + 1) * (255/2)

            # Features for synth images.
            synth_sketch_features = self.vgg16(synth_sketch, resize_images=False, return_lpips=True)
            dist_sketch = (target_sketch_features - synth_sketch_features).square().sum() * sketch_weight
            
            # unedited regions masks
            if sample_weight > 0.0:
                sample_feature = result['sample_feature']
                synth_sample_feature = torch.masked_select(sample_feature, edit_unmask)
                dist_unmask = F.l1_loss(synth_sample_feature, target_sample_feature) * sample_weight
                dist_unmask_item = dist_unmask.item()
            else:
                dist_unmask = dist_unmask_item = 0.0
            
            loss = dist + dist_sketch + dist_unmask
            print(step, "Dist Loss", dist.item(), "Dist Sketch Loss", dist_sketch.item(), "Dist Unmask loss", dist_unmask_item)

            # Step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        w_opt = w_opt.detach().requires_grad_(False)
        return w_opt

