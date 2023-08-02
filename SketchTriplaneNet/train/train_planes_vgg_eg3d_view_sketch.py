# Training to a set of multiple objects (e.g. ShapeNet or DTU)
# tensorboard logs available in logs/<expname>

import sys
import os

import numpy as np
import torch.nn.functional as F
import torch
from dotmap import DotMap
import time
import PIL.Image
import dnnlib

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import warnings
from trainlib.trainer_eg3d import Trainer
from model import make_model, loss
from render.nerf_planes_eg3d import NeRFRenderer
from data import get_split_dataset
import util
from criteria.lpips.lpips import LPIPS

from model.networks_stylegan2 import Generator as StyleGAN2Backbone

def extra_args(parser):
    parser.add_argument(
        "--batch_size", "-B", type=int, default=4, help="Object batch size ('SB')"
    )
    parser.add_argument(
        "--nviews",
        "-V",
        type=str,
        default="1",
        help="Number of source views (multiview); put multiple (space delim) to pick randomly per batch ('NV')",
    )
    parser.add_argument(
        "--freeze_enc",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )

    parser.add_argument(
        "--no_bbox_step",
        type=int,
        default=100000,
        help="Step to stop using bbox sampling",
    )
    parser.add_argument(
        "--fixed_test",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )
    return parser


args, conf = util.args.parse_args(extra_args, training=True, default_ray_batch_size=1024)
device = util.get_cuda(args.gpu_id[0])

dset, val_dset, _ = get_split_dataset(args.dataset_format, args.datadir)
print(
    "dset z_near {}, z_far {}, lindisp {}".format(dset.z_near, dset.z_far, dset.lindisp)
)

net = make_model(conf["model"]).to(device=device)
net.stop_encoder_grad = args.freeze_enc
if args.freeze_enc:
    print("Encoder frozen")
    net.encoder.eval()

renderer = NeRFRenderer.from_conf(conf["renderer"], lindisp=dset.lindisp,).to(
    device=device
)

# Parallize
render_par = renderer.bind_parallel(net, args.gpu_id).eval()

nviews = list(map(int, args.nviews.split()))

class PixelNeRFTrainer(Trainer):
    def __init__(self):
        super().__init__(net, dset, val_dset, args, conf["train"], device=device)
        self.renderer_state_path = "%s/%s/_renderer" % (
            self.args.checkpoints_path,
            self.args.name,
        )

        self.lambda_coarse = conf.get_float("loss.lambda_coarse")
        self.lambda_fine = conf.get_float("loss.lambda_fine", 1.0)
        print(
            "lambda coarse {} and fine {}".format(self.lambda_coarse, self.lambda_fine)
        )
        self.rgb_coarse_crit = loss.get_rgb_loss(conf["loss.rgb"], True)
        fine_loss_conf = conf["loss.rgb"]
        if "rgb_fine" in conf["loss"]:
            print("using fine loss")
            fine_loss_conf = conf["loss.rgb_fine"]
        self.rgb_fine_crit = loss.get_rgb_loss(fine_loss_conf, False)

        if args.resume:
            if os.path.exists(self.renderer_state_path):
                renderer.load_state_dict(
                    torch.load(self.renderer_state_path, map_location=device)
                )

        self.z_near = dset.z_near
        self.z_far = dset.z_far

        self.use_bbox = args.no_bbox_step > 0

        self.lpips = False
        if self.lpips:
            self.lpips_loss = LPIPS(net_type='alex').to(device).eval()
        self.vgg = True
        if self.vgg:
            print("Load VGG16 ...")
            home = './checkpoints/'
            vgg16_path = os.path.abspath(os.path.join(home, 'vgg16.pt'))
            with dnnlib.util.open_url(vgg16_path) as f:
                self.vgg16 = torch.jit.load(f).eval().to(device)

        mapping_kwargs = dnnlib.EasyDict()
        mapping_kwargs['num_layers'] = 2
        synthesis_dict = {'conv_clamp':256, 'fused_modconv_default':'inference_only','channel_base':32768, 'channel_max':512}
        self.backbone = StyleGAN2Backbone(z_dim=512, c_dim=25, w_dim=512, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_dict)
        self.backbone.to(device)
        eg3d_dict = torch.load('./checkpoints/EG3D_weights_balanced_anime.pkl')['weights']
        def get_keys(d, name):
            d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
            return d_filt
        backbone_dict = get_keys(eg3d_dict, 'backbone')
        self.backbone.load_state_dict(backbone_dict, strict=True)
        self.plane_loss = torch.nn.L1Loss()

    def post_batch(self, epoch, batch):
        renderer.sched_step(args.batch_size)

    def extra_save_state(self):
        torch.save(renderer.state_dict(), self.renderer_state_path)

    def calc_losses(self, data, is_train=True, global_step=0):
        if "images" not in data:
            return {}
        ws = data["ws"].to(device=device)
        all_sketches = data["sketches"].to(device=device)
        all_images = data["images"].to(device=device)  # (SB, NV, 3, H, W)

        SB, NV, _, H, W = all_images.shape
        all_poses = data["poses"].to(device=device)  # (SB, NV, 4, 4)
        all_bboxes = data.get("bbox")  # (SB, NV, 4)  cmin rmin cmax rmax
        all_focals = data["focal"]  # (SB)
        all_c = data.get("c")  # (SB)

        if self.use_bbox and global_step >= args.no_bbox_step:
            self.use_bbox = False
            print(">>> Stopped using bbox sampling @ iter", global_step)

        if not is_train or not self.use_bbox:
            all_bboxes = None

        all_rgb_gt = []
        all_rays = []
        
        render_img = True
        #render_img = False
        if render_img:
            image_ord = torch.randint(0, NV, (SB, 1))
            target_ord = torch.randint(0, NV, (SB, 1))

            target_ord = target_ord.to(device)
            target_images = util.batched_index_select_nd(
                all_images, target_ord
            )  # (SB, NS, 3, H, W)
            target_images_0to1 = target_images * 0.5 + 0.5
            target_poses = util.batched_index_select_nd(all_poses, target_ord)  # (SB, NS, 4, 4)
            target_poses = target_poses.squeeze(1)

            cam_rays = util.gen_rays(
                    target_poses, W, H, all_focals[0].to(device=device),
                    self.z_near, self.z_far, c=all_c[0].to(device=device)
                )  # (NV, H, W, 8)
            all_rays = cam_rays.reshape(SB, H*W, 8)
            all_rgb_gt = target_images_0to1.reshape(SB, 3, H*W).permute(0,2,1)
        else:
            curr_nviews = nviews[torch.randint(0, len(nviews), ()).item()]
            if curr_nviews == 1:
                image_ord = torch.randint(0, NV, (SB, 1))
            else:
                image_ord = torch.empty((SB, curr_nviews), dtype=torch.long)
            for obj_idx in range(SB):
                if all_bboxes is not None:
                    bboxes = all_bboxes[obj_idx]
                sketches = all_sketches[obj_idx]
                images = all_images[obj_idx]  # (NV, 3, H, W)
                poses = all_poses[obj_idx]  # (NV, 4, 4)
                focal = all_focals[obj_idx]
                c = None
                if "c" in data:
                    c = data["c"][obj_idx]
                if curr_nviews > 1:
                    # Somewhat inefficient, don't know better way
                    image_ord[obj_idx] = torch.from_numpy(
                        np.random.choice(NV, curr_nviews, replace=False)
                    )
                images_0to1 = images * 0.5 + 0.5

                cam_rays = util.gen_rays(
                    poses, W, H, focal, self.z_near, self.z_far, c=c
                )  # (NV, H, W, 8)
                #print("cam_rays: ", cam_rays.shape)
                rgb_gt_all = images_0to1
                rgb_gt_all = (
                    rgb_gt_all.permute(0, 2, 3, 1).contiguous().reshape(-1, 3)
                )  # (NV, H, W, 3)

                if all_bboxes is not None:
                    pix = util.bbox_sample(bboxes, args.ray_batch_size)
                    pix_inds = pix[..., 0] * H * W + pix[..., 1] * W + pix[..., 2]
                else:
                    pix_inds = torch.randint(0, NV * H * W, (args.ray_batch_size,))

                rgb_gt = rgb_gt_all[pix_inds]  # (ray_batch_size, 3)
                rays = cam_rays.view(-1, cam_rays.shape[-1])[pix_inds].to(
                    device=device
                )  # (ray_batch_size, 8)
                #print("rays: ", rays.shape)

                all_rgb_gt.append(rgb_gt)
                all_rays.append(rays)

            all_rgb_gt = torch.stack(all_rgb_gt)  # (SB, ray_batch_size, 3)
            all_rays = torch.stack(all_rays)  # (SB, ray_batch_size, 8)

        image_ord = image_ord.to(device)
        src_images = util.batched_index_select_nd(
            all_images, image_ord
        )  # (SB, NS, 3, H, W)
        src_sketches = util.batched_index_select_nd(
            all_sketches, image_ord
        )  # (SB, NS, 3, H, W)
        src_poses = util.batched_index_select_nd(all_poses, image_ord)  # (SB, NS, 4, 4)

        all_bboxes = all_poses = all_images = None

        planes_syn = net.encode(
            src_sketches, 
            src_images,
            src_poses,
            all_focals.to(device=device),
            c=all_c.to(device=device) if all_c is not None else None,
        )

        render_dict = DotMap(render_par(all_rays, want_weights=True,))
        coarse = render_dict.coarse
        fine = render_dict.fine
        using_fine = len(fine) > 0
        
        loss_dict = {}
        rgb_loss = self.rgb_coarse_crit(coarse.rgb, all_rgb_gt)
        loss_dict["rc"] = rgb_loss.item() * self.lambda_coarse
        if using_fine:
            fine_loss = self.rgb_fine_crit(fine.rgb, all_rgb_gt)
            rgb_loss = rgb_loss * self.lambda_coarse + fine_loss * self.lambda_fine
            loss_dict["rf"] = fine_loss.item() * self.lambda_fine
        loss = rgb_loss

        self.lambda_planes = 0.01
        with torch.no_grad():
            planes_gt = self.backbone.synthesis(ws, update_emas=False, noise_mode = 'const')
        planes_gt = planes_gt.view(len(planes_gt), 3, 32, planes_gt.shape[-2], planes_gt.shape[-1])
        planes_loss = self.plane_loss(planes_syn, planes_gt) * self.lambda_planes
        loss = loss + planes_loss
        loss_dict["planes"] = planes_loss.item()
        
        if self.lpips:
            self.lambda_lpips = 0.1
            fine_img = fine.rgb.reshape(SB, H, W, 3)
            fine_img = fine_img.permute(0,3,1,2)
            fine_img = fine_img * 2.0 - 1.0

            loss_lpips = self.lpips_loss(target_images[:,0,:,:,:], fine_img) * self.lambda_lpips
            loss_dict["lpips"] = loss_lpips.item()
            loss = loss + loss_lpips

        if self.vgg:
            self.lambda_vgg = 0.1
            fine_img = fine.rgb.reshape(SB, H, W, 3)
            fine_img = fine_img.permute(0,3,1,2)
            target_images_0to1 = target_images_0to1 * 255.0
            target_features = self.vgg16(target_images_0to1[:,0,:,:,:], resize_images=False, return_lpips=True)
            fine_img = fine_img * 255.0
            synth_features = self.vgg16(fine_img, resize_images=False, return_lpips=True)
            loss_vgg = (target_features - synth_features).square().sum() * self.lambda_vgg
            loss_dict["vgg"] = loss_vgg.item()
            loss = loss + loss_vgg

        if is_train:
            loss.backward()
        loss_dict["t"] = loss.item()

        return loss_dict

    def train_step(self, data, global_step):
        start = time.time()
        torch.cuda.synchronize()
        loss_dict = self.calc_losses(data, is_train=True, global_step=global_step)
        torch.cuda.synchronize()
        end = time.time()
        #print("one step: ", end-start)
        return loss_dict

    def eval_step(self, data, global_step):
        renderer.eval()
        losses = self.calc_losses(data, is_train=False, global_step=global_step)
        renderer.train()
        return losses

    def vis_step(self, data, global_step, idx=None):
        if "images" not in data:
            return {}
        if idx is None:
            batch_idx = np.random.randint(0, data["images"].shape[0])
        else:
            print(idx)
            batch_idx = idx
        sketches = data["sketches"][batch_idx].to(device=device) 
        images = data["images"][batch_idx].to(device=device)  # (NV, 3, H, W)
        poses = data["poses"][batch_idx].to(device=device)  # (NV, 4, 4)
        focal = data["focal"][batch_idx : batch_idx + 1]  # (1)
        c = data.get("c")
        if c is not None:
            c = c[batch_idx : batch_idx + 1]  # (1)
        NV, _, H, W = images.shape
        cam_rays = util.gen_rays(
            poses, W, H, focal, self.z_near, self.z_far, c=c
        )  # (NV, H, W, 8)
        images_0to1 = images * 0.5 + 0.5  # (NV, 3, H, W)

        curr_nviews = nviews[torch.randint(0, len(nviews), (1,)).item()]
        views_src = np.sort(np.random.choice(NV, curr_nviews, replace=False))
        view_dest = np.random.randint(0, NV - curr_nviews)
        for vs in range(curr_nviews):
            view_dest += view_dest >= views_src[vs]
        views_src = torch.from_numpy(views_src)

        # set renderer net to eval mode
        renderer.eval()
        source_views = (
            images_0to1[views_src]
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
            .reshape(-1, H, W, 3)
        )

        gt = images_0to1[view_dest].permute(1, 2, 0).cpu().numpy().reshape(H, W, 3)
        with torch.no_grad():
            test_rays = cam_rays[view_dest]  # (H, W, 8)
            test_images = images[views_src]  # (NS, 3, H, W)
            test_sketches = sketches[views_src]  # (NS, 3, H, W)
            net.encode(
                test_sketches.unsqueeze(0), 
                test_images.unsqueeze(0),
                poses[views_src].unsqueeze(0),
                focal.to(device=device),
                c=c.to(device=device) if c is not None else None,
            )
            test_rays = test_rays.reshape(1, H * W, -1)
            render_dict = DotMap(render_par(test_rays, want_weights=True))
            coarse = render_dict.coarse
            fine = render_dict.fine

            using_fine = len(fine) > 0

            alpha_coarse_np = coarse.weights[0].sum(dim=-1).cpu().numpy().reshape(H, W)
            rgb_coarse_np = coarse.rgb[0].cpu().numpy().reshape(H, W, 3)
            depth_coarse_np = coarse.depth[0].cpu().numpy().reshape(H, W)

            if using_fine:
                alpha_fine_np = fine.weights[0].sum(dim=1).cpu().numpy().reshape(H, W)
                depth_fine_np = fine.depth[0].cpu().numpy().reshape(H, W)
                rgb_fine_np = fine.rgb[0].cpu().numpy().reshape(H, W, 3)

        print("c rgb min {} max {}".format(rgb_coarse_np.min(), rgb_coarse_np.max()))
        print(
            "c alpha min {}, max {}".format(
                alpha_coarse_np.min(), alpha_coarse_np.max()
            )
        )
        alpha_coarse_cmap = util.cmap(alpha_coarse_np) / 255
        depth_coarse_cmap = util.cmap(depth_coarse_np) / 255
        vis_list = [
            *source_views,
            gt,
            depth_coarse_cmap,
            rgb_coarse_np,
            alpha_coarse_cmap,
        ]

        vis_coarse = np.hstack(vis_list)
        vis = vis_coarse

        if using_fine:
            print("f rgb min {} max {}".format(rgb_fine_np.min(), rgb_fine_np.max()))
            print(
                "f alpha min {}, max {}".format(
                    alpha_fine_np.min(), alpha_fine_np.max()
                )
            )
            depth_fine_cmap = util.cmap(depth_fine_np) / 255
            alpha_fine_cmap = util.cmap(alpha_fine_np) / 255
            vis_list = [
                *source_views,
                gt,
                depth_fine_cmap,
                rgb_fine_np,
                alpha_fine_cmap,
            ]

            vis_fine = np.hstack(vis_list)
            vis = np.vstack((vis_coarse, vis_fine))
            rgb_psnr = rgb_fine_np
        else:
            rgb_psnr = rgb_coarse_np

        psnr = util.psnr(rgb_psnr, gt)
        vals = {"psnr": psnr}
        print("psnr", psnr)

        # set the renderer network back to train mode
        renderer.train()
        return vis, vals


trainer = PixelNeRFTrainer()
trainer.start()


