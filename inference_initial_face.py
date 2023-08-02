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

# SketchTriplaneNet modules
sys.path.append('./SketchTriplaneNet')
sys.path.append('./SketchTriplaneNet/src')
import src.util.util as util
from src.render.nerf_planes_eg3d import NeRFRenderer
from src.model import make_model
from pyhocon import ConfigFactory
import warnings
from scipy.interpolate import CubicSpline

# pSp model modules
sys.path.append('./pSp')
from pSp.models.psp_infer import psp_feature

# parsing model modules
from parsing_model.model import BiSeNet
from parsing_model.norm import SpecificNorm

class Coach:
    def __init__(self, ckpt_path):
        self.device = 'cuda'

        # Initialize SketchTriplaneNet
        conf_path = './SketchTriplaneNet/conf/exp/face_planes_eg3d.conf'
        conf = ConfigFactory.parse_file(conf_path)
        self.pixelNeRF = make_model(conf["model"]).to(device=self.device)
        self.pixelNeRF.load_weights_ckpt(ckpt_path)
        # Initialize render
        self.ray_batch_size = 200000
        renderer = NeRFRenderer.from_conf(
            conf["renderer"], lindisp=False, eval_batch_size=self.ray_batch_size,
        ).to(device=self.device)
        self.render_par = renderer.bind_parallel(self.pixelNeRF, "0", simple_output=False).eval()
        self.focal = torch.tensor(np.array([545.9138]), dtype=torch.float32).to(self.device)
        self.c = torch.tensor(np.array([[64, 64]]), dtype=torch.float32).to(self.device)

        self.coord_trans = torch.diag(
        	torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        ).unsqueeze(0).unsqueeze(0).to(device=self.device)

    def pixel_nerf_ortho(self, sketch, image, camera_params):
        # initialize parameters
        H=W=128
        K = 128
        B,_,_,_ = sketch.shape
        coord_trans = self.coord_trans.repeat(B,1,1,1)
        pose = camera_params[:,0:16].reshape(-1,4,4)
        pose = pose.unsqueeze(1) # pose B, 1, 4, 4
        pose = pose @ coord_trans
        plane_features = self.pixelNeRF.encode(
            sketch.unsqueeze(1),
            image.unsqueeze(1),
            pose,
            self.focal.repeat(B),
            c = self.c.repeat(B,1),
        )
        return plane_features

    def render_pixel_nerf(self, camera_params):
        H=W=128
        coord_trans = self.coord_trans
        pose = camera_params[:,0:16].reshape(-1,4,4)
        pose = pose.unsqueeze(1)
        pose = pose @ coord_trans
        pose = pose.squeeze(1)
        ray_batch_size = 200000
        render_rays = util.gen_rays(
            pose,
            W,
            H,
            self.focal,
            2.25,
            3.3,
            c=self.c,
        ).to(device=self.device)
        all_rgb_fine = []
        for rays in tqdm.tqdm(
            torch.split(render_rays.view(-1, 8), ray_batch_size, dim=0)
        ):
            outputs = self.render_par(rays[None])
            rgb = outputs['fine']['rgb']
            all_rgb_fine.append(rgb[0])
        _depth = None
        rgb_fine = torch.cat(all_rgb_fine)
        # rgb_fine (V*H*W, 3)
        frames = rgb_fine.view(-1, H, W, 3)
        frames = frames.permute(0,3,1,2)
        frames = frames * 2.0 - 1.0
        return frames

def read_img(path, img_size = 256):
    transform = transforms.Compose(
        [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    img = Image.open(path).convert('RGB')
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    img = img.cuda()
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default = "./results/edit/", help = "Directory for generating and editing samples")
    parser.add_argument("--img", type=str, default = "appear.png", help = "file name of render image")
    parser.add_argument("--sketch", type=str, default = "sket.png", help = "file name of edit sketch")
    parser.add_argument("--mask", type=str, default = "mask.jpg", help = "file name of mask")
    parser.add_argument("--sketchTriplane_ckpt", type=str, default="./checkpoints/face_planes_eg3d_sketch_new/pixel_nerf_latest", help = "ckpt path for sketchTriplane")
    parser.add_argument("--pSp_ckpt", type=str, default="./checkpoints/pSp/iteration_400000.pt", help = "ckpt path for pSp skpt")
    parser.add_argument("--eg3d_ckpt", type=str, default='./checkpoints/eg3d/network-snapshot-001000.pkl', help = "The checkpoint for eg3d")
    args = parser.parse_args()

    device = torch.device('cuda')
    
    # Initialize SketchTriplane Net
    sketchTriplane_ckpt_path = args.sketchTriplane_ckpt
    coach = Coach(sketchTriplane_ckpt_path)

    pSp_ckpt_path = args.pSp_ckpt
    psp_feature = psp_feature(pSp_ckpt_path)
    psp_feature.to(device)
        
    # Load EG3D model
    eg3d_path = args.eg3d_ckpt
    with open(eg3d_path, 'rb') as f:
        G_load = pickle.load(f)['G_ema'].to(device).eval()
    # ReLoad Generator
    mapping_kwargs = dnnlib.EasyDict()
    mapping_kwargs['num_layers'] = 2
    sr_kwargs = dnnlib.EasyDict(channel_base=32768, channel_max=512, fused_modconv_default='inference_only')
    synthesis_dict = {'conv_clamp':256, 'fused_modconv_default':'inference_only','channel_base':32768, 'channel_max':512}
    G = TriPlaneGenerator(512,25,512,512,3,sr_num_fp16_res=4,mapping_kwargs=mapping_kwargs,rendering_kwargs=G_load.rendering_kwargs,sr_kwargs=sr_kwargs,**synthesis_dict)
    G.load_state_dict(G_load.state_dict())
    G.neural_rendering_resolution = 128
    G = G.to(device)

    sampling_multiplier = 2.0
    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)

    # Initialize face parsing net
    n_classes = 19
    bisenet = BiSeNet(n_classes=n_classes)
    bisenet.cuda()
    save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
    bisenet.load_state_dict(torch.load(save_pth))
    bisenet = bisenet.eval()
    spNorm = SpecificNorm()

    root_dir = './results/edit'
    sketch_path = os.path.join(root_dir, args.sketch)
    img_path = os.path.join(root_dir, args.img)
    input_mask_path = os.path.join(root_dir, args.mask)
    img_mask_path = os.path.join(root_dir, 'appear_mask.jpg')
    result_fusion_path = os.path.join(root_dir, 'projection_fusion.jpg')

    sketch = read_img(sketch_path, 256).cuda()
    mask = read_img(input_mask_path, 256).cuda()
    mask = mask.mean(1, keepdims=True)
    input_mask_01 = torch.ones([1,1,256,256]).cuda() * (mask > 0.0)

    optimize_path = os.path.join(root_dir, 'optimize_dict.npy')
    optimize_dict = np.load(optimize_path, allow_pickle=True)
    optimize_dict = optimize_dict.item()
    #for k,v in optimize_dict.items():
    #    print(k)
    original_latent = torch.tensor(optimize_dict['initial_latent']).to(device)

    cam_pivot = torch.tensor([0, 0.05, 0.2], device=device)
    cam_radius = 2.7
    fov_deg = 18.837
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    angle_y = optimize_dict['angle_y']
    angle_p = optimize_dict['angle_p']
    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    with torch.no_grad():
        # Mask background
        image = read_img(img_path, 512).cuda()
        img_bk = ((image + 1) / 2)
        img_bk = spNorm(img_bk)
        with torch.no_grad():
            out = bisenet(img_bk)[0]
        parsing = out.squeeze(0).detach().cpu().numpy().argmax(0)
        vis_parsing_anno = parsing.astype(np.uint8)  # (512, 512)
        valid_index = np.where(vis_parsing_anno==0)
        img_save = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_save = img_save[0].cpu().numpy()
        img_save[valid_index] = 0
        Image.fromarray(img_save, 'RGB').save(img_mask_path)

        image = read_img(img_mask_path, 256).cuda()

        #--------------Sketch Triplane ----------------
        sketch_256 = F.interpolate(sketch, (256,256), mode='bilinear')
        image_128 = F.interpolate(image, (128,128), mode='bilinear')
        plane_features = coach.pixel_nerf_ortho(sketch_256, image_128, camera_params)
        pSp_code = psp_feature.forward_planes(plane_features)
    
        fusion_code = psp_feature.forward_code(G, input_mask_01, original_latent, pSp_code, camera_params)
        #-------------save fusion img----------------
        fusion_img = G.synthesis(fusion_code, camera_params, noise_mode = "const")['image']
        img_save = (fusion_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_save = img_save[0].cpu().numpy()
        Image.fromarray(img_save, 'RGB').save(result_fusion_path)
    
    optimize_dict['fusion_latent'] = fusion_code.cpu().numpy()
    np.save(optimize_path, optimize_dict)

