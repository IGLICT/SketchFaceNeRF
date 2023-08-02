import os
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

import cv2
from torchvision import transforms

from argparse import ArgumentParser

from model import BiSeNet
from norm import SpecificNorm

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dir", type=str, default="./data/EG3D_sample", help = "the root path of dataset")
    args = parser.parse_args()

    ## face parsing net
    n_classes = 19
    bisenet = BiSeNet(n_classes=n_classes)
    bisenet.cuda()
    save_pth = os.path.join('./checkpoint', '79999_iter.pth')
    bisenet.load_state_dict(torch.load(save_pth))
    bisenet = bisenet.eval()
    spNorm = SpecificNorm()

    target_transform = transforms.Compose(
        [
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    root_path = args.dir
    sketch_dir = os.path.join(root_path, 'sketch')
    image_dir = os.path.join(root_path, 'image')
    sketch_contour_dir = os.path.join(root_path, 'sketch_contour')
    if not os.path.exists(sketch_contour_dir):
        os.mkdir(sketch_contour_dir)

    for seed in tqdm(range(0, 30000)):
    #for seed in tqdm(range(0, 200)):
        from_path = os.path.join(image_dir, '%05d.png' % (seed))
        img = PIL.Image.open(from_path)
        img = target_transform(img).unsqueeze(0).cuda()

        source_img = ((img + 1) / 2)
        source_img_norm = spNorm(source_img)
        source_img_512  = F.interpolate(source_img_norm,size=(512,512))
        out = bisenet(source_img_512)[0]
        out = torch.argmax(out, dim=1, keepdim=True)

        parsing = out.cpu().detach().numpy().squeeze()
        vis_parsing_anno = parsing.astype(np.uint8)  # (512, 512)

        sketch_path = os.path.join(sketch_dir, '%05d.png' % (seed))
        new_sketch = cv2.imread(sketch_path, 0)
        
        # Mask background
        valid_index = np.where(vis_parsing_anno==0)
        new_sketch[valid_index] = 255.0

        # Mask Lips
        valid_index = np.where(vis_parsing_anno==12)
        new_sketch[valid_index] = 255.0

        valid_index = np.where(vis_parsing_anno==13)
        new_sketch[valid_index] = 255.0
        

        ###     1        2         3        4        5        6        7        8        9       10      11       12       13       14       15       16      17      18
        ### ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
        face_part_ids = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
        face_map = np.ones([512,512]) * 255.0

        for valid_id in face_part_ids:
            valid_index = np.where(vis_parsing_anno==valid_id)
            face_map[valid_index] = 0.0
        
        face_map = face_map.astype(np.uint8)
        ret, binary =cv2.threshold(face_map,127,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(new_sketch,contours,-1,(0,0,0),2)

        ### Neck Mask Drawing
        neck_mask = np.ones((512,512)) * (vis_parsing_anno == 14) * 255.0
        neck_mask = neck_mask.astype(np.uint8)
        
        ret, binary =cv2.threshold(neck_mask,127,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(new_sketch,contours,-1,(0,0,0),2)

        # Hair Mask Drawing
        hair_mask = np.ones((512,512)) * (vis_parsing_anno == 17) * 255.0
        hair_mask = hair_mask.astype(np.uint8)
       
        ret, binary =cv2.threshold(hair_mask,127,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(new_sketch,contours,-1,(0,0,0),2)

        # Lip Mask Drawing
        Lip_mask = np.ones((512,512)) * (vis_parsing_anno == 12) * 255.0
        Lip_mask = Lip_mask.astype(np.uint8)
       
        ret, binary =cv2.threshold(Lip_mask,127,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(new_sketch,contours,-1,(0,0,0),2)

        Lip_mask = np.ones((512,512)) * (vis_parsing_anno == 13) * 255.0
        Lip_mask = Lip_mask.astype(np.uint8)
        
        ret, binary =cv2.threshold(Lip_mask,127,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(new_sketch,contours,-1,(0,0,0),2)
        
        new_sketch[:,0:2] = 255
        new_sketch[:,510:512] = 255
        new_sketch[0:2,:] = 255
        new_sketch[510:512,:] = 255
        sketch_contour_128_path = os.path.join(sketch_contour_dir, '%05d.png' % (seed))
        cv2.imwrite(sketch_contour_128_path, new_sketch)

#python sketch_aug.py --dir /mnt/16T/liufenglin/dataset/anime/EG3D_sample
