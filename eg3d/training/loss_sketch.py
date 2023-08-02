# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing

import torch.nn.functional as F
import random

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class SketchLoss(Loss):
    def __init__(self, device, G, neural_rendering_resolution_initial=128, lamada_128=3.0, lamada_512=3.0, lambda_vgg=2.0, lambda_nerf=3.0):
        super().__init__()
        self.device             = device
        self.G                  = G

        # Define Sketch weight loss
        self.lamada_128 = lamada_128
        self.lamada_512 = lamada_512
        self.lambda_vgg = lambda_vgg
        self.lambda_nerf = lambda_nerf

        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        self.blur_raw_target = True
        #assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)
        print("Load VGG16 ...")
        import dnnlib
        vgg16_path = './pretrained_models/vgg16.pt'
        with dnnlib.util.open_url(vgg16_path) as f:
            self.vgg16 = torch.jit.load(f).eval().to(self.device)
    
    def run_G(self, ws, c, neural_rendering_resolution, update_emas=False):
        results = self.G.synthesis_sketch(ws, c, neural_rendering_resolution=neural_rendering_resolution, noise_mode='const')
        return results
    
    def accumulate_gradients(self, ws, c, real_sketch, print_loss=False):
        with torch.autograd.profiler.record_function('Gmain_forward'):
            neural_rendering_resolution = self.neural_rendering_resolution_initial
            results = self.run_G(ws, c, neural_rendering_resolution=neural_rendering_resolution)
            gen_sketch_raw = results['sketch_raw']
            gen_sketch = results['sketch']

            loss = 0.0

            if self.lambda_nerf > 0.0:
                sample_pixel = np.linspace(0,512*512-1,512*512)
                random.shuffle(sample_pixel)
                #sample_length = 512*32
                sample_length = 512*16
                sample_pixel_index = sample_pixel[0:sample_length]
                sample_pixel_index = torch.IntTensor(sample_pixel_index).to(ws.device)
                cam2world_matrix = c[:, :16].view(-1, 4, 4)

                intrinsics = c[:, 16:25].view(-1, 3, 3)
                # Create a batch of rays for volume rendering
                ray_origins, ray_directions = self.G.ray_sampler(cam2world_matrix, intrinsics, 512)
                #ray_origins = ray_origins[:, 0:512*32, :]
                #ray_directions = ray_directions[:, 0:512*32, :]
                ray_origins = torch.index_select(ray_origins, 1, sample_pixel_index)
                ray_directions = torch.index_select(ray_directions, 1, sample_pixel_index)
                planes = self.G.backbone.synthesis(ws,update_emas=False)
                planes_sketch = self.G.planes_sparse(planes)

                planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
                planes_sketch = planes_sketch.view(len(planes_sketch), 3, 32, planes_sketch.shape[-2], planes_sketch.shape[-1])

                # Perform volume rendering
                #feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last
                feature_samples, depth_samples, weights_samples = self.G.renderer.forward_sketch(planes, planes_sketch, self.G.decoder, self.G.decoder_sketch, ray_origins, ray_directions, self.G.rendering_kwargs) # channels last
                N, M, _ = ray_origins.shape
                feature_image = feature_samples.permute(0, 2, 1)
                sketch_nerf_sample = feature_image[:, :1, :]
                
                sketch_sr = gen_sketch.reshape(N,1,512*512)
                sketch_sr_sample = torch.index_select(sketch_sr, 2, sample_pixel_index)
                
                loss_nerf = F.l1_loss(sketch_nerf_sample, sketch_sr_sample) * self.lambda_nerf
                loss += loss_nerf

                training_stats.report('Loss/loss_nerf', loss_nerf)

            real_sketch_raw = F.interpolate(real_sketch, size=(128,128),mode='bilinear')
            loss_128_raw = F.l1_loss(gen_sketch_raw, real_sketch_raw) * self.lamada_128
            loss += loss_128_raw

            training_stats.report('Loss/loss_L1_128', loss_128_raw)

            if self.lambda_vgg > 0:
                gen_sketch_raw = (gen_sketch_raw.repeat(1,3,1,1) + 1) * (255/2)
                # Features for synth images.
                synth_features_raw = self.vgg16(-gen_sketch_raw, resize_images=False, return_lpips=True)
                real_sketch_raw = (real_sketch_raw.repeat(1,3,1,1) + 1) * (255/2)
                # Features for synth images.
                target_features_raw = self.vgg16(-real_sketch_raw, resize_images=False, return_lpips=True)
                dist_raw = (target_features_raw - synth_features_raw).square().sum() * self.lambda_vgg
                loss += dist_raw
                training_stats.report('Loss/loss_VGG_128', dist_raw)
            
            loss_512 = F.l1_loss(gen_sketch, real_sketch) * self.lamada_512
            loss += loss_512
            training_stats.report('Loss/loss_L1_512', loss_512)
            
            if self.lambda_vgg > 0:
                gen_sketch = (gen_sketch.repeat(1,3,1,1) + 1) * (255/2)
                # Features for synth images.
                synth_features = self.vgg16(-gen_sketch, resize_images=False, return_lpips=True)
                real_sketch = (real_sketch.repeat(1,3,1,1) + 1) * (255/2)
                # Features for synth images.
                target_features = self.vgg16(-real_sketch, resize_images=False, return_lpips=True)
                dist = (target_features - synth_features).square().sum() * self.lambda_vgg
                loss += dist
                training_stats.report('Loss/loss_VGG_512', dist)
            
        with torch.autograd.profiler.record_function('Gmain_backward'):
            loss.mean().backward()



