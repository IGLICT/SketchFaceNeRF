import os
from PIL import Image
import torch
import numpy as np

# Deep3DFaceRecon modules
from Deep3DFaceRecon.pose import Poser
from Deep3DFaceRecon.Recon_networks import define_net_recon

def split_coeff(coeffs):
    """
    Return:
        coeffs_dict     -- a dict of torch.tensors

    Parameters:
        coeffs          -- torch.tensor, size (B, 256)
    """
    id_coeffs = coeffs[:, :80]
    exp_coeffs = coeffs[:, 80: 144]
    tex_coeffs = coeffs[:, 144: 224]
    angles = coeffs[:, 224: 227]
    gammas = coeffs[:, 227: 254]
    translations = coeffs[:, 254:]
    return {
        'id': id_coeffs,
        'exp': exp_coeffs,
        'tex': tex_coeffs,
        'angle': angles,
        'gamma': gammas,
        'trans': translations
    }

## Initialize 3D recon network
pose_estimator = Poser("cuda")
net_recon = define_net_recon(net_recon='resnet50', use_last_fc=False, init_path='checkpoints/init_model/resnet50-0676ba61.pth')
load_path = os.path.join('./Deep3DFaceRecon/checkpoints/pretrained/epoch_20.pth')
state_dict = torch.load(load_path)
keys = list(state_dict.keys())
net_recon.load_state_dict(state_dict['net_recon'])
net_recon.eval()
net_recon.cuda()

device = 'cuda'
DFD_img_path = ''
cam_path = ''

#---------------predict camera params----------------------
p2p_pil = Image.open(DFD_img_path)
p2p_pil = p2p_pil.resize((224,224))
im = torch.tensor(np.array(p2p_pil)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
im = im.cuda()
output_coeff = net_recon(im)
output_coeff = output_coeff.detach().cpu().numpy()
pred_coeffs = split_coeff(output_coeff)
pose_dict = pose_estimator.get_pose(pred_coeffs)
pose = pose_dict["pose"]
intrinsics = torch.from_numpy(np.array(pose_dict["intrinsics"], dtype=np.float32)).to(device)
camera_params = torch.cat([torch.from_numpy(np.array(pose, dtype=np.float32)).to(device).reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
camera_params_np = camera_params.cpu().numpy()
np.save(cam_path, camera_params_np)

