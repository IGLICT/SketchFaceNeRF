# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from torch_utils import persistence
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
from training.volumetric_rendering.renderer_sketch3_project import ImportanceRenderer
from training.volumetric_rendering.ray_sampler import RaySampler
import dnnlib

from torch import nn
import torch.nn.functional as F

#triplanes with convolution on 3 planes
#projection triplanes, with mask regions features constrain

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler()
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        self.superresolution = dnnlib.util.construct_class_by_name(class_name='training.superresolution.SuperresolutionHybrid8XDC', channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.neural_rendering_resolution = 128
        self.rendering_kwargs = rendering_kwargs

        self.decoder_sketch = OSGDecoder_sketch(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        self.superresolution_sketch = dnnlib.util.construct_class_by_name(class_name='training.superresolution.SuperresolutionHybrid8XDC_sketch', channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.planes_sparse = Features_conv(n_blocks=2).apply(weights_init)
    
        self._last_planes = None
    
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
                c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        #import numpy as np
        #np.save('./planes.npy', planes.cpu().numpy())
 
        # Perform volume rendering
        feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'planes': planes}

    def synthesis_sketch(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        planes_sketch = self.planes_sparse(planes)

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        planes_sketch = planes_sketch.view(len(planes_sketch), 3, 32, planes_sketch.shape[-2], planes_sketch.shape[-1])

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples = self.renderer.forward_sketch(planes, planes_sketch, self.decoder, self.decoder_sketch, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        sketch_raw = feature_image[:, :1]
        sr_sketch = self.superresolution_sketch(sketch_raw, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        return {'sketch': sr_sketch, 'sketch_raw': sketch_raw, 'image_depth': depth_image, 'feature_image':feature_image}
    
    def synthesis_all_mask(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        planes_sketch = self.planes_sparse(planes)

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        planes_sketch = planes_sketch.view(len(planes_sketch), 3, 32, planes_sketch.shape[-2], planes_sketch.shape[-1])

        # Perform volume rendering
        image_feature_samples, sketch_feature_samples, depth_samples, weights_samples, sample_feature = self.renderer.forward_all(planes, planes_sketch, self.decoder, self.decoder_sketch, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        image_feature_image = image_feature_samples.permute(0, 2, 1).reshape(N, image_feature_samples.shape[-1], H, W).contiguous()
        sketch_feature_image = sketch_feature_samples.permute(0, 2, 1).reshape(N, sketch_feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = image_feature_image[:, :3]
        sr_image = self.superresolution(rgb_image, image_feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        sketch_raw = sketch_feature_image[:, :1]
        sr_sketch = self.superresolution_sketch(sketch_raw, sketch_feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        return {'image': sr_image, 'image_raw': rgb_image,'sketch': sr_sketch, 'sketch_raw': sketch_raw, 'image_depth': depth_image, 'sample_feature':sample_feature, 'feature_image':sketch_feature_image}

    def synthesis_3D_mask(self, ws, c, mask_2D, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)
        
        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution
        
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        # Perform volume rendering
        # Notice that the sample_features is [1,128,128,96,32]
        _, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        #################################################################################
        #calculate sketch point cloud
        coordinates = depth_samples * ray_directions + ray_origins
        H=W=128
        coordinates = coordinates.permute(0, 2, 1).reshape(N, coordinates.shape[-1], H, W)
        coordinates = F.interpolate(coordinates, (256,256),mode='bilinear')

        mask_2D = mask_2D[:,0,:,:]
        mask_2D = mask_2D.reshape(mask_2D.shape[0], 256*256)

        coordinates = coordinates.reshape(coordinates.shape[0], coordinates.shape[1], 256*256)

        mask_xy = torch.zeros([mask_2D.shape[0],1,128,128]).cuda()
        mask_xz = torch.zeros([mask_2D.shape[0],1,128,128]).cuda()

        #-----------------mask 3D is different in batch--------------------
        for b in range(mask_2D.shape[0]):
            # find mask index
            mask = (mask_2D[b,:] > 0.5).nonzero()
            point_cloud = torch.index_select(coordinates[b,:,:], 1, mask.squeeze())
            # change the coordinate scale
            point_cloud = point_cloud * 2.0
            point_cloud_np = point_cloud.cpu().numpy()
            # with index to generate 3 planes mask
            mask_xy[b, :, (point_cloud_np[1,:]*64 + 64)%128, (point_cloud_np[0,:]*64 + 64)%128] = 1.0
            mask_xz[b, :, (point_cloud_np[2,:]*64 + 64)%128, (point_cloud_np[0,:]*64 + 64)%128] = 1.0
            # Given depth in z
            mask_xz[b, :, (point_cloud_np[2,:]*64 + 63)%128, (point_cloud_np[0,:]*64 + 64)%128] = 1.0
            mask_xz[b, :, (point_cloud_np[2,:]*64 + 65)%128, (point_cloud_np[0,:]*64 + 64)%128] = 1.0

        mask_xy = F.interpolate(mask_xy, size=(256, 256)).unsqueeze(1)
        mask_zx = F.interpolate(mask_xz.permute(0,1,3,2), size=(256, 256)).unsqueeze(1)
        mask_xz = F.interpolate(mask_xz, size=(256, 256)).unsqueeze(1)
        
        mask_3d = torch.cat([mask_xy, mask_xz, mask_zx], dim = 1)
        return mask_3d, planes, ray_origins, ray_directions
    
    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)


from training.networks_stylegan2 import FullyConnectedLayer

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}

class OSGDecoder_sketch(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64
        
        self.net_MLP = torch.nn.Sequential(
            FullyConnectedLayer(n_features + 24, self.hidden_dim, activation='relu'),
            FullyConnectedLayer(self.hidden_dim, self.hidden_dim, activation='relu'),
            FullyConnectedLayer(self.hidden_dim, self.hidden_dim, activation='relu'),
            FullyConnectedLayer(self.hidden_dim, options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )

        use_dir = True
        if use_dir:
            from training.volumetric_rendering.embed import HarmonicEmbedding
            self.embed = HarmonicEmbedding(n_harmonic_functions = 4, logspace = True, append_input = False)
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        x = sampled_features

        if len(x.shape)>3:
            x = x.mean(1)

        N, M, C = x.shape
        x = x.view(N*M, C)

        ray_dir_embed = self.embed(ray_directions)
        N_d, M_d, C_d = ray_dir_embed.shape
        ray_dir_embed = ray_dir_embed.view(N_d*M_d, C_d)
        x = torch.cat((x, ray_dir_embed),1)
        x = self.net_MLP(x)

        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x)*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        return {'sketch': rgb}

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        #out = x + self.conv_block(x)
        out = self.conv_block(x)
        return out

#Planes sparse modules
class Features_conv(torch.nn.Module):
    def __init__(self, n_blocks):
        super().__init__()
        self.n_blocks = n_blocks

        models = []
        for i in range(n_blocks):
            models += [ResnetBlock(dim=96, padding_type='reflect', norm_layer=torch.nn.InstanceNorm2d)]

        self.net = torch.nn.Sequential(*models)
        
    def forward(self, planes):
        #print("using sparse planes modules")
        planes = self.net(planes)
        return planes
