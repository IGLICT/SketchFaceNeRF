import argparse
import sys
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
import time
import tqdm
import pickle

# eg3d modules
sys.path.append('./eg3d')
import dnnlib
from training.triplane_sketch3_project import TriPlaneGenerator
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default = "./results/edit/", help = "Directory for generating and editing samples")
    parser.add_argument("--seed", type=int, default = 10, help = "Seed used for sample")
    parser.add_argument("--angle_y", type=float, default = 0.0, help = "Yaw angle for rendering")
    parser.add_argument("--angle_p", type=float, default = 0.0, help = "Pitch angle for rendering")
    parser.add_argument("--trunc", type=float, default=0.7, help = "Truncation psi")
    parser.add_argument("--trunc_cutoff", type=int, default=14, help = "Truncation cutoff")
    parser.add_argument("--eg3d_ckpt", type=str, default='./checkpoints/eg3d/network-snapshot-001000.pkl', help = "The checkpoint for eg3d")
    args = parser.parse_args()

    # Load parameters
    root_dir = args.dir
    seed = args.seed
    angle_y = args.angle_y
    angle_p = args.angle_p
    truncation_psi = args.trunc
    truncation_cutoff = args.trunc_cutoff
    ckpt_path = args.eg3d_ckpt

    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    device = torch.device('cuda')
        
    # Load EG3D model
    with open(ckpt_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device).eval()
    G.neural_rendering_resolution = 128
    G.rendering_kwargs['depth_resolution'] = 96
    G.rendering_kwargs['depth_resolution_importance'] = 96

    cam_pivot = torch.tensor([0, 0.05, 0.2], device=device)
    cam_radius = 2.7
    fov_deg = 18.837
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    # Predict latent code
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

    # Render results
    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    
    with torch.no_grad():
        results = G.synthesis_sketch(ws, camera_params, noise_mode='const')
    sketch = results['sketch']
    sketch = sketch.permute(0, 2, 3, 1) + 1.0
    sketch = (255.0 - (sketch * 255.0).clamp(0, 255)).to(torch.uint8)
    sketch = sketch.repeat(1,1,1,3)
    Image.fromarray(sketch[0].cpu().numpy(), 'RGB').save(f'{root_dir}/sketch_render.png')

    with torch.no_grad():
        results = G.synthesis(ws, camera_params, noise_mode='const')
    image = results['image']
    image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    Image.fromarray(image[0].cpu().numpy(), 'RGB').save(f'{root_dir}/appear.png')

    optimize_path = os.path.join(root_dir, 'optimize_dict.npy')
    optimize_dict = {}
    optimize_dict['seed'] = seed
    optimize_dict['angle_y'] = angle_y
    optimize_dict['angle_p'] = angle_p
    optimize_dict['initial_latent'] = ws.detach().cpu().numpy()

    np.save(optimize_path, optimize_dict)

