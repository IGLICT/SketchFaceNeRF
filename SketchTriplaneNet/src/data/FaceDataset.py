import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
from util import get_image_to_tensor_balanced, get_mask_to_tensor
from torchvision import transforms

class FaceDataset(torch.utils.data.Dataset):
    """
    Dataset from SRN (V. Sitzmann et al. 2020)
    """

    def __init__(
        self, path, stage="train", image_size=(128, 128), world_scale=1.0,
    ):
        """
        :param stage train | val | test
        :param image_size result image size (resizes if different)
        :param world_scale amount to scale entire world by
        """
        super().__init__()
        self.path = path
        self.base_path = os.path.join(path, stage)
        self.dataset_name = os.path.basename(path)

        print("Loading SRN dataset", self.base_path, "name:", self.dataset_name)
        self.stage = stage
        assert os.path.exists(self.base_path)

        self.root_lists = sorted(os.listdir(self.base_path))

        #self.intrins = sorted(
        #    glob.glob(os.path.join(self.base_path, "*", "intrinsics.npy"))
        #)

        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()

        self.image_size = image_size
        self.world_scale = world_scale

        self.z_near = 2.25
        self.z_far = 3.3
        self.lindisp = False

        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )

    def __len__(self):
        return len(self.root_lists)

    def __getitem__(self, index):
        #intrin_path = self.intrins[index]
        #dir_path = os.path.dirname(intrin_path)
        #rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*")))
        #pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*")))
        #assert len(rgb_paths) == len(pose_paths)
        root_dir = self.root_lists[index]
        sketch_paths = os.path.join(self.base_path, root_dir, 'sketch_128.npy')
        rgb_paths = os.path.join(self.base_path, root_dir, 'imgs_seg.npy')
        pose_paths = os.path.join(self.base_path, root_dir, 'poses.npy')
        sketch_list = list(np.load(sketch_paths))
        rgb_list = list(np.load(rgb_paths))
        pose_list = list(np.load(pose_paths))

        intrin_path = os.path.join(self.path, 'intrinsics.npy')
        intrin = np.load(intrin_path)
        focal = intrin[0,0] * 127.9984
        cx = intrin[0,2] * 128
        cy = intrin[1,2] * 128

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        all_sketches = []

        #for rgb_path, pose_path in zip(rgb_paths, pose_paths):
        #    img = imageio.imread(rgb_path)[..., :3]
        for i in range(len(rgb_list)):
            img = transforms.ToPILImage()(rgb_list[i])
            sketch = transforms.ToPILImage()(sketch_list[i])
            #print(type(img))
            img_tensor = self.image_to_tensor(img)
            sketch_tensor = self.image_to_tensor(sketch)
            img = rgb_list[i]
            mask = (img != 255).all(axis=-1)[..., None].astype(np.uint8) * 255
            mask_tensor = self.mask_to_tensor(mask)

            pose = pose_list[i]
            pose = torch.from_numpy(
                pose.reshape(4, 4)
            )
            pose = pose @ self._coord_trans
            
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rnz = np.where(rows)[0]
            cnz = np.where(cols)[0]
            if len(rnz) == 0:
                raise RuntimeError(
                    "ERROR: Bad image at", rgb_path, "please investigate!"
                )
            rmin, rmax = rnz[[0, -1]]
            cmin, cmax = cnz[[0, -1]]
            bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)

            all_imgs.append(img_tensor)
            all_masks.append(mask_tensor)
            all_poses.append(pose)
            all_bboxes.append(bbox)
            all_sketches.append(sketch_tensor)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        all_masks = torch.stack(all_masks)
        all_bboxes = torch.stack(all_bboxes)
        all_sketches = torch.stack(all_sketches)

        if all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale
            cx *= scale
            cy *= scale
            all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")
            all_sketches = F.interpolate(all_sketches, size=self.image_size, mode="area")

        if self.world_scale != 1.0:
            focal *= self.world_scale
            all_poses[:, :3, 3] *= self.world_scale
        #focal = torch.tensor(focal, dtype=torch.float32)

        result = {
            "path": root_dir,
            "img_id": index,
            "focal": focal,
            "c": torch.tensor([cx, cy], dtype=torch.float32),
            "images": all_imgs,
            "masks": all_masks,
            "bbox": all_bboxes,
            "poses": all_poses,
            "sketches": all_sketches,
        }
        return result
