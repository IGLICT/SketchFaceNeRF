"""
Main model implementation
"""
import torch
import torch.autograd.profiler as profiler
from util import repeat_interleave
import os
import os.path as osp
import warnings
import time

import numpy as np
import dnnlib

from .encoder import ImageEncoder
from .code import PositionalEncoding
from .model_util import make_encoder, make_mlp

from .networks_maskGAN import weights_init, GlobalGenerator_adain, planes_Generator_single_conv_eg3d
from .networks_stylegan2 import FullyConnectedLayer

def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]]], dtype=torch.float32)

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    #print("inv planes:", inv_planes)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.hidden_dim = 64

        decoder_lr_mul = 1.0
        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=decoder_lr_mul),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + 32, lr_multiplier=decoder_lr_mul)
        )
        
    def forward(self, sampled_features):
        # Aggregate features
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        return x

class PixelNeRFNet(torch.nn.Module):
    def __init__(self, conf, stop_encoder_grad=False):
        """
        :param conf PyHocon config subtree 'model'
        """
        super().__init__()
        self.sketch_encoder = GlobalGenerator_adain(input_nc=3, output_nc=3, ngf=32, n_blocks=5)
        self.sketch_encoder.apply(weights_init)

        self.encoder = make_encoder(conf["encoder"])
        self.use_encoder = conf.get_bool("use_encoder", True)  # Image features?

        self.use_xyz = conf.get_bool("use_xyz", False)

        assert self.use_encoder or self.use_xyz  # Must use some feature..

        # Whether to shift z to align in canonical frame.
        # So that all objects, regardless of camera distance to center, will
        # be centered at z=0.
        # Only makes sense in ShapeNet-type setting.
        self.normalize_z = conf.get_bool("normalize_z", True)

        self.stop_encoder_grad = (
            stop_encoder_grad  # Stop ConvNet gradient (freeze weights)
        )
        self.use_code = conf.get_bool("use_code", False)  # Positional encoding
        self.use_code_viewdirs = conf.get_bool(
            "use_code_viewdirs", True
        )  # Positional encoding applies to viewdirs

        self.latent_size = self.encoder.latent_size
        # LFL: change the MLP input size
        self.decoder = OSGDecoder(n_features=32)

        #eg3d_dict = torch.load('../checkpoints/EG3D_weights_balanced_anime.pkl')['weights']
        #def get_keys(d, name):
        #    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
        #    return d_filt
        #decoder_dict = get_keys(eg3d_dict, 'decoder')
        #self.decoder.load_state_dict(decoder_dict, strict=True)

        # Planes conv
        self.planes_conv = planes_Generator_single_conv_eg3d(input_nc = 128*9*3)
        self.planes_conv.apply(weights_init)

        # Note: this is world -> camera, and bottom row is omitted
        self.register_buffer("poses", torch.empty(1, 3, 4), persistent=False)
        self.register_buffer("image_shape", torch.empty(2), persistent=False)

        self.register_buffer("focal", torch.empty(1, 2), persistent=False)
        # Principal point
        self.register_buffer("c", torch.empty(1, 2), persistent=False)

        # Generate voxel space
        resolution = 128 # the resolution of voxel space
        #uv1, uv2, uv3 = torch.meshgrid(torch.arange(resolution, dtype=torch.float32,), torch.arange(resolution, dtype=torch.float32), torch.arange(resolution, dtype=torch.float32))
        uv1, uv2, uv3 = torch.meshgrid(torch.arange(resolution, dtype=torch.float32,), torch.arange(resolution, dtype=torch.float32), torch.arange(resolution, dtype=torch.float32), indexing='ij')
        voxel_space = torch.stack([uv1, uv2, uv3], dim=3) * (2.0/resolution) - 1.0 + (1./resolution)
        voxel_space = voxel_space.reshape(128*128*128, 3).unsqueeze(0)
        voxel_space = voxel_space / 2.0
        #self.voxel_space = voxel_space
        # voxel_space: (1, 128*128*128, 3)
        self.register_buffer("voxel_space", voxel_space, persistent=False)
        
        plane_axes = generate_planes()
        self.register_buffer("plane_axes", plane_axes, persistent=False)

        self.d_out = 4
        self.num_objs = 0
        self.num_views_per_obj = 1
    
    def encode_planes_ws(self, planes):
        # Reshape output into three 32-channel planes
        self.plane_features_gt = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        #self.plane_features = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

    def encode(self, sketches, images, poses, focal, z_bounds=None, c=None):
        """
        :param images (NS, 3, H, W)
        NS is number of input (aka source or reference) views
        :param poses (NS, 4, 4)
        :param focal focal length () or (2) or (NS) or (NS, 2) [fx, fy]
        :param z_bounds ignored argument (used in the past)
        :param c principal point None or () or (2) or (NS) or (NS, 2) [cx, cy],
        default is center of image
        """
        
        self.num_objs = images.size(0)
        #print("poses.shape:", poses.shape)
        if len(images.shape) == 5:
            assert len(poses.shape) == 4
            assert poses.size(1) == images.size(
                1
            )  # Be consistent with NS = num input views
            self.num_views_per_obj = images.size(1)
            images = images.reshape(-1, *images.shape[2:])
            sketches = sketches.reshape(-1, *sketches.shape[2:])
            poses = poses.reshape(-1, 4, 4)
        else:
            self.num_views_per_obj = 1
        images_gen, _ = self.sketch_encoder(sketches, images)
        self.encoder(images_gen)
        #self.encoder(images)
        rot = poses[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
        trans = -torch.bmm(rot, poses[:, :3, 3:])  # (B, 3, 1)
        self.poses = torch.cat((rot, trans), dim=-1)  # (B, 3, 4)

        self.image_shape[0] = images.shape[-1]
        self.image_shape[1] = images.shape[-2]

        # Handle various focal length/principal point formats
        if len(focal.shape) == 0:
            # Scalar: fx = fy = value for all views
            focal = focal[None, None].repeat((1, 2))
        elif len(focal.shape) == 1:
            # Vector f: fx = fy = f_i *for view i*
            # Length should match NS (or 1 for broadcast)
            focal = focal.unsqueeze(-1).repeat((1, 2))
        else:
            focal = focal.clone()
        self.focal = focal.float()
        self.focal[..., 1] *= -1.0

        if c is None:
            # Default principal point is center of image
            c = (self.image_shape * 0.5).unsqueeze(0)
        elif len(c.shape) == 0:
            # Scalar: cx = cy = value for all views
            c = c[None, None].repeat((1, 2))
        elif len(c.shape) == 1:
            # Vector c: cx = cy = c_i *for view i*
            c = c.unsqueeze(-1).repeat((1, 2))
        self.c = c

        #---------------------------------Generate planes---------------------------------   
        voxel_space = self.voxel_space.repeat(self.num_objs,1,1)
        # index colors (original and flip)
        xyz_rot = torch.matmul(self.poses[:, None, :3, :3], voxel_space.unsqueeze(-1))[
                ..., 0
            ]
        xyz = xyz_rot + self.poses[:, None, :3, 3]
        uv = -xyz[:, :, :2] / xyz[:, :, 2:]  # (SB, B, 2)
        uv *= repeat_interleave(
            self.focal.unsqueeze(1), 1
        )
        uv += repeat_interleave(
            self.c.unsqueeze(1), 1
        )  # (SB, B, 2)
        latent = self.encoder.index(
            uv, None, self.image_shape
        )  # (SB, latent_size, B)
        # Flip the voxel
        xyz_flip = voxel_space.clone()
        xyz_flip[:,:,0] = -xyz_flip[:,:,0]
        xyz_rot_flip = torch.matmul(self.poses[:, None, :3, :3], xyz_flip.unsqueeze(-1))[
                ..., 0
            ]
        xyz_flip = xyz_rot_flip + self.poses[:, None, :3, 3]
        uv_flip = -xyz_flip[:, :, :2] / xyz_flip[:, :, 2:]  # (SB, B, 2)
        uv_flip *= repeat_interleave(
            self.focal.unsqueeze(1), 1
        )
        uv_flip += repeat_interleave(
            self.c.unsqueeze(1), 1
        )  # (SB*NS, B, 2))
        latent_flip = self.encoder.index(
            uv_flip, None, self.image_shape
        )  # (SB, latent_size, B)
        latent_features = torch.cat((latent, latent_flip, xyz.permute(0,2,1)), dim=1)
        latent_features = latent_features.reshape(self.num_objs, 9, 128, 128, 128)
        voxel_space_XY = latent_features.permute(0,1,4,3,2).reshape(self.num_objs, 9*128, 128, 128)
        voxel_space_XZ = latent_features.permute(0,1,3,4,2).reshape(self.num_objs, 9*128, 128, 128)
        voxel_space_ZX = latent_features.permute(0,1,3,2,4).reshape(self.num_objs, 9*128, 128, 128)
        
        voxel_space_features = torch.cat((voxel_space_XY, voxel_space_XZ, voxel_space_ZX), dim=1)
        plane_features = self.planes_conv(voxel_space_features)
        plane_features = torch.cat((plane_features[:,0:32,:,:].unsqueeze(1), \
                                plane_features[:,32:64,:,:].unsqueeze(1), plane_features[:,64:96,:,:].unsqueeze(1)), dim=1)
        self.plane_features = plane_features
        return plane_features

    def forward(self, xyz):
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (SB, B, 3)
        SB is batch of objects
        B is batch of points (in rays)
        NS is number of input views
        :return (SB, B, 4) r g b sigma
        """
        self.plane_axes = self.plane_axes.to(xyz.device)
        SB, B, _ = xyz.shape
        N, n_planes, C, H, W = self.plane_features.shape
        _, M, _ = xyz.shape
        self.plane_features = self.plane_features.to(xyz.device)
        plane_features = self.plane_features.view(N*n_planes, C, H, W)
        coordinates_proj = project_onto_planes(self.plane_axes, xyz).unsqueeze(1)
        #print(plane_features)
        #print(coordinates_proj)

        output_features = torch.nn.functional.grid_sample(plane_features, coordinates_proj.float(), mode='bilinear', padding_mode='zeros', align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
        mlp_input = torch.mean(output_features, dim=1, keepdim=False)
        mlp_output = self.decoder(mlp_input)

        # Interpret the output
        mlp_output = mlp_output.reshape(-1, B, 33)

        rgb = torch.sigmoid(mlp_output[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = mlp_output[..., 0:1]
        output_list = [sigma, rgb]

        output = torch.cat(output_list, dim=-1)
        output = output.reshape(SB, B, -1)
        return output
    
    def forward_gt(self, xyz):
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (SB, B, 3)
        SB is batch of objects
        B is batch of points (in rays)
        NS is number of input views
        :return (SB, B, 4) r g b sigma
        """
        self.plane_axes = self.plane_axes.to(xyz.device)
        SB, B, _ = xyz.shape
        N, n_planes, C, H, W = self.plane_features_gt.shape
        _, M, _ = xyz.shape
        self.plane_features_gt = self.plane_features_gt.to(xyz.device)
        plane_features_gt = self.plane_features_gt.view(N*n_planes, C, H, W)
        coordinates_proj = project_onto_planes(self.plane_axes, xyz).unsqueeze(1)
        #print(plane_features)
        #print(coordinates_proj)

        output_features = torch.nn.functional.grid_sample(plane_features_gt, coordinates_proj.float(), mode='bilinear', padding_mode='zeros', align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
        # [N,3,1000000,32]
        mlp_input = torch.mean(output_features, dim=1, keepdim=False)
        mlp_output = self.decoder(mlp_input)

        # Interpret the output
        mlp_output = mlp_output.reshape(-1, B, 33)

        #output_list = [torch.sigmoid(rgb), torch.relu(sigma)]
        rgb = torch.sigmoid(mlp_output[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = mlp_output[..., 0:1]
        output_list = [sigma, rgb]

        output = torch.cat(output_list, dim=-1)
        output = output.reshape(SB, B, -1)
        return output

    def load_weights(self, args, opt_init=False, strict=True, device=None):
        """
        Helper for loading weights according to argparse arguments.
        Your can put a checkpoint at checkpoints/<exp>/pixel_nerf_init to use as initialization.
        :param opt_init if true, loads from init checkpoint instead of usual even when resuming
        """
        # TODO: make backups
        if opt_init and not args.resume:
            return
        ckpt_name = (
            "pixel_nerf_init" if opt_init or not args.resume else "pixel_nerf_latest"
        )
        model_path = "%s/%s/%s" % (args.checkpoints_path, args.name, ckpt_name)

        if device is None:
            device = self.poses.device

        if os.path.exists(model_path):
            print("Load", model_path)
            model_dict = torch.load(model_path, map_location=device)
            for k,v in model_dict.items():
                print(k)
            self.load_state_dict(
                model_dict, strict=strict
            )
        elif not opt_init:
            warnings.warn(
                (
                    "WARNING: {} does not exist, not loaded!! Model will be re-initialized.\n"
                    + "If you are trying to load a pretrained model, STOP since it's not in the right place. "
                    + "If training, unless you are startin a new experiment, please remember to pass --resume."
                ).format(model_path)
            )
        return self
    
    def load_weights_ckpt(self, model_path, opt_init=False, strict=True, device=None):
        if device is None:
            device = self.poses.device

        if os.path.exists(model_path):
            print("Load", model_path)
            model_dict = torch.load(model_path, map_location=device)
            #for k,v in model_dict.items():
            #    print(k)
            self.load_state_dict(
                model_dict, strict=strict
            )
        elif not opt_init:
            warnings.warn(
                (
                    "WARNING: {} does not exist, not loaded!! Model will be re-initialized.\n"
                    + "If you are trying to load a pretrained model, STOP since it's not in the right place. "
                    + "If training, unless you are startin a new experiment, please remember to pass --resume."
                ).format(model_path)
            )
        return self

    def save_weights(self, args, opt_init=False):
        """
        Helper for saving weights according to argparse arguments
        :param opt_init if true, saves from init checkpoint instead of usual
        """
        from shutil import copyfile

        ckpt_name = "pixel_nerf_init" if opt_init else "pixel_nerf_latest"
        backup_name = "pixel_nerf_init_backup" if opt_init else "pixel_nerf_backup"

        ckpt_path = osp.join(args.checkpoints_path, args.name, ckpt_name)
        ckpt_backup_path = osp.join(args.checkpoints_path, args.name, backup_name)

        if osp.exists(ckpt_path):
            copyfile(ckpt_path, ckpt_backup_path)
        torch.save(self.state_dict(), ckpt_path)
        return self
