import torch
import numpy as np

def compute_rotation(angles):
    """
    Return:
        rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

    Parameters:
        angles           -- torch.tensor, size (B, 3), radian
    """

    batch_size = angles.shape[0]
    ones = torch.ones([batch_size, 1])
    zeros = torch.zeros([batch_size, 1])
    x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],

    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x), 
        zeros, torch.sin(x), torch.cos(x)
    ], dim=1).reshape([batch_size, 3, 3])

    rot_y = torch.cat([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y)
    ], dim=1).reshape([batch_size, 3, 3])

    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones
    ], dim=1).reshape([batch_size, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    return rot.permute(0, 2, 1)[0]

def fix_pose_orig(pose):
    pose = np.array(pose).copy()
    location = pose[:3, 3]
    radius = np.linalg.norm(location)
    pose[:3, 3] = pose[:3, 3]/radius * 2.7
    return pose

def fix_intrinsics(intrinsics):
    intrinsics = np.array(intrinsics).copy()
    assert intrinsics.shape == (3, 3), intrinsics
    intrinsics[0,0] = 2985.29/700
    intrinsics[1,1] = 2985.29/700
    intrinsics[0,2] = 1/2
    intrinsics[1,2] = 1/2
    assert intrinsics[0,1] == 0
    assert intrinsics[2,2] == 1
    assert intrinsics[1,0] == 0
    assert intrinsics[2,0] == 0
    assert intrinsics[2,1] == 0
    #print(intrinsics.shape)
    return intrinsics

class Poser(object):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
    def convert(self, angle, trans):
        R = compute_rotation(torch.from_numpy(np.array(angle)[None, ...])).numpy()
        c = -np.dot(R, trans)
        pose = np.eye(4)
        pose[:3, :3] = R

        c *= 0.27 # factor to match tripleganger
        c[1] += 0.006 # offset to align to tripleganger
        c[2] += 0.161 # offset to align to tripleganger
        pose[0,3] = c[0]
        pose[1,3] = c[1]
        pose[2,3] = c[2]

        focal = 2985.29 # = 1015*1024/224*(300/466.285)#
        pp = 512#112
        w = 1024#224
        h = 1024#224
        
        K = np.eye(3)
        K[0][0] = focal
        K[1][1] = focal
        K[0][2] = w/2.0
        K[1][2] = h/2.0
        K = K.tolist()

        Rot = np.eye(3)
        Rot[0, 0] = 1
        Rot[1, 1] = -1
        Rot[2, 2] = -1        
        pose[:3, :3] = np.dot(pose[:3, :3], Rot)

        pose = pose.tolist()
        return fix_pose_orig(pose)
    
    def get_pose(self, pred_coeffs):
        angle = pred_coeffs['angle']
        trans = pred_coeffs['trans'][0]
        R = compute_rotation(torch.from_numpy(angle)).numpy()

        trans[2] += -10
        c = -np.dot(R, trans)
        pose = np.eye(4)
        pose[:3, :3] = R

        c *= 0.27 # factor to match tripleganger
        c[1] += 0.006 # offset to align to tripleganger
        c[2] += 0.161 # offset to align to tripleganger
        pose[0,3] = c[0]
        pose[1,3] = c[1]
        pose[2,3] = c[2]

        focal = 2985.29 # = 1015*1024/224*(300/466.285)#
        pp = 512#112
        w = 1024#224
        h = 1024#224
        
        K = np.eye(3)
        K[0][0] = focal
        K[1][1] = focal
        K[0][2] = w/2.0
        K[1][2] = h/2.0
        K = K.tolist()

        Rot = np.eye(3)
        Rot[0, 0] = 1
        Rot[1, 1] = -1
        Rot[2, 2] = -1        
        pose[:3, :3] = np.dot(pose[:3, :3], Rot)

        pose = pose.tolist()
        out = {}
        #out["intrinsics"] = K
        out["intrinsics"] = fix_intrinsics(K)
        out["trans"] = trans
        out["pose"] = fix_pose_orig(pose)
        out["angle"] = angle.flatten().tolist()
        return out

