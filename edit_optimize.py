import argparse

import torch
import numpy as np
import sys
import os

from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import pickle

# eg3d modules
sys.path.append('./eg3d')
import dnnlib
from training.triplane_sketch3_project import TriPlaneGenerator
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics

from project_adain import Projector

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default = "./results/edit/", help = "Directory for generating and editing samples")
    parser.add_argument("--img", type=str, default = "appear.png", help = "file name of render image")
    parser.add_argument("--sketch", type=str, default = "sket.png", help = "file name of edit sketch")
    parser.add_argument("--mask", type=str, default = "mask.jpg", help = "file name of mask")
    parser.add_argument("--eg3d_ckpt", type=str, default='./checkpoints/eg3d/network-snapshot-001000.pkl', help = "The checkpoint for eg3d")
    parser.add_argument("--sketch_weight", type=float, default=40.0, help = "Optimization weight for sketch")
    parser.add_argument("--rgb_weight", type=float, default=20.0, help = "Optimization weight for image")
    parser.add_argument("--sample_weight", type=float, default=0.2, help = "Optimization weight for sample points in unedited regions")
    parser.add_argument("--num_steps", type=int, default=20, help = "Optimization steps")
    args = parser.parse_args()

    device = torch.device('cuda')
        
    # Load EG3D model
    with open('./checkpoints/eg3d/network-snapshot-001000.pkl', 'rb') as f:
        G_load = pickle.load(f)['G_ema'].to(device).eval()
    mapping_kwargs = dnnlib.EasyDict()
    mapping_kwargs['num_layers'] = 2
    sr_kwargs = dnnlib.EasyDict(channel_base=32768, channel_max=512, fused_modconv_default='inference_only')
    synthesis_dict = {'conv_clamp':256, 'fused_modconv_default':'inference_only','channel_base':32768, 'channel_max':512}
    G = TriPlaneGenerator(512,25,512,512,3,sr_num_fp16_res=4,mapping_kwargs=mapping_kwargs,rendering_kwargs=G_load.rendering_kwargs,sr_kwargs=sr_kwargs,**synthesis_dict)
    G.load_state_dict(G_load.state_dict())
    G.neural_rendering_resolution = 128
    G = G.to(device)

    cam_pivot = torch.tensor([0, 0.05, 0.2], device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    fov_deg = 18.837
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    
    projector = Projector(G, conditioning_params, device)

    root_dir = './results/edit/'
    sketch_path = os.path.join(root_dir, 'sket.png')
    img_path = os.path.join(root_dir, 'appear.png')
    input_mask_path = os.path.join(root_dir, 'mask.jpg')
    opt_res_path  = os.path.join(root_dir, 'results_opt.jpg')
    opt_code_path  = os.path.join(root_dir, 'results_opt_latent.npy')

    sketch = read_img1(sketch_path).cuda()
    image = read_img1(img_path).cuda()
    mask = read_img1(input_mask_path).cuda()
    mask = mask.mean(1, keepdims=True)

    optimize_path = os.path.join(root_dir, 'optimize_dict.npy')
    optimize_dict = np.load(optimize_path, allow_pickle=True)
    optimize_dict = optimize_dict.item()
    # for k,v in optimize_dict.items():
    #     print(k)
    original_latent = optimize_dict['initial_latent']
    fusion_latent = optimize_dict['fusion_latent']

    cam_pivot = torch.tensor([0, 0.05, 0.2], device=device)
    cam_radius = 2.7
    fov_deg = 18.837
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    angle_y = optimize_dict['angle_y']
    angle_p = optimize_dict['angle_p']
    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    
    optimize_dict={
        'sketch_weight': args.sketch_weight,
        'rgb_weight': args.rgb_weight,
        'sample_weight': args.sample_weight,
        'num_steps': args.num_steps, 
    }
    current_step = 0
    opt_code = projector.project(target_image=image, 
                                edit_sketch=sketch, 
                                edit_mask=mask, 
                                ws=fusion_latent, 
                                w_ori=original_latent, 
                                camera_params=camera_params, 
                                sketch_weight=optimize_dict['sketch_weight'], 
                                RGB_weight=optimize_dict['rgb_weight'], 
                                sample_weight=optimize_dict['sample_weight'], 
                                num_steps=optimize_dict['num_steps'], 
                                current_step=current_step)
    
    sampling_multiplier = 2.0
    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    
    with torch.no_grad():
        #-------------save opt img----------------
        img = G.synthesis(opt_code, camera_params, noise_mode='const')['image']
        img_save = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_save = img_save[0].cpu().numpy()
        Image.fromarray(img_save, 'RGB').save(opt_res_path)
    
        opt_code_np = opt_code.cpu().numpy()
        np.save(opt_code_path, opt_code_np)
