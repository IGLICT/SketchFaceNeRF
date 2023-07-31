import torch
import torch.nn.functional as F
import torch.autograd.profiler as profiler
from torch.nn import DataParallel
from dotmap import DotMap
import time
import numpy as np
from .ray_marcher import MipRayMarcher2

class _RenderWrapper(torch.nn.Module):
    def __init__(self, net, renderer, simple_output):
        super().__init__()
        self.net = net
        self.renderer = renderer
        self.simple_output = simple_output

    def forward(self, rays, want_weights=False):
        if rays.shape[0] == 0:
            return (
                torch.zeros(0, 3, device=rays.device),
                torch.zeros(0, device=rays.device),
            )

        outputs = self.renderer(
            self.net, rays, want_weights=want_weights and not self.simple_output
        )
        if self.simple_output:
            if self.renderer.using_fine:
                rgb = outputs.fine.rgb
                depth = outputs.fine.depth
            else:
                rgb = outputs.coarse.rgb
                depth = outputs.coarse.depth
            return rgb, depth
        else:
            # Make DotMap to dict to support DataParallel
            return outputs.toDict()
    
    def query_voxel(self, sb, device, render=True):
        outputs = self.renderer.query_voxel(self.net, sb=sb, device=device, render=render)
        if self.simple_output:
            rgb = outputs.coarse.rgb
            depth = outputs.coarse.depth
            return rgb, depth
        else:
            # Make DotMap to dict to support DataParallel
            return outputs.toDict()


class NeRFRenderer(torch.nn.Module):
    """
    NeRF differentiable renderer
    :param n_coarse number of coarse (binned uniform) samples
    :param n_fine number of fine (importance) samples
    :param n_fine_depth number of expected depth samples
    :param noise_std noise to add to sigma. We do not use it
    :param depth_std noise for depth samples
    :param eval_batch_size ray batch size for evaluation
    :param white_bkgd if true, background color is white; else black
    :param lindisp if to use samples linear in disparity instead of distance
    :param sched ray sampling schedule. list containing 3 lists of equal length.
    sched[0] is list of iteration numbers,
    sched[1] is list of coarse sample numbers,
    sched[2] is list of fine sample numbers
    """

    def __init__(
        self,
        n_coarse=128,
        n_fine=0,
        n_fine_depth=0,
        noise_std=0.0,
        depth_std=0.01,
        eval_batch_size=100000,
        white_bkgd=False,
        lindisp=False,
        sched=None,  # ray sampling schedule for coarse and fine rays
    ):
        super().__init__()
        self.n_coarse = n_coarse
        self.n_fine = n_fine
        self.n_fine_depth = n_fine_depth

        self.noise_std = noise_std
        self.depth_std = depth_std

        self.eval_batch_size = eval_batch_size
        self.white_bkgd = white_bkgd
        self.lindisp = lindisp
        if lindisp:
            print("Using linear displacement rays")
        self.using_fine = n_fine > 0
        self.sched = sched
        if sched is not None and len(sched) == 0:
            self.sched = None
        self.register_buffer(
            "iter_idx", torch.tensor(0, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "last_sched", torch.tensor(0, dtype=torch.long), persistent=True
        )
        self.ray_marcher = MipRayMarcher2()

    def sample_coarse(self, rays):
        """
        Stratified sampling. Note this is different from original NeRF slightly.
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :return (B, Kc)
        """
        device = rays.device
        near, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)

        step = 1.0 / self.n_coarse
        B = rays.shape[0]
        z_steps = torch.linspace(0, 1 - step, self.n_coarse, device=device)  # (Kc)
        z_steps = z_steps.unsqueeze(0).repeat(B, 1)  # (B, Kc)
        z_steps += torch.rand_like(z_steps) * step
        if not self.lindisp:  # Use linear sampling in depth space
            return near * (1 - z_steps) + far * z_steps  # (B, Kf)
        else:  # Use linear sampling in disparity space
            return 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)  # (B, Kf)

        # Use linear sampling in depth space
        return near * (1 - z_steps) + far * z_steps  # (B, Kc)

    def sample_fine(self, rays, weights):
        """
        Weighted stratified (importance) sample
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param weights (B, Kc)
        :return (B, Kf-Kfd)
        """
        device = rays.device
        B = rays.shape[0]

        weights = weights.detach() + 1e-5  # Prevent division by zero
        pdf = weights / torch.sum(weights, -1, keepdim=True)  # (B, Kc)
        cdf = torch.cumsum(pdf, -1)  # (B, Kc)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (B, Kc+1)
        #print("self.n_fine - self.n_fine_depth:", self.n_fine - self.n_fine_depth)

        u = torch.rand(
            B, self.n_fine - self.n_fine_depth, dtype=torch.float32, device=device
        )  # (B, Kf)
        inds = torch.searchsorted(cdf, u, right=True).float() - 1.0  # (B, Kf)
        inds = torch.clamp_min(inds, 0.0)

        z_steps = (inds + torch.rand_like(inds)) / self.n_coarse  # (B, Kf)

        near, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)
        if not self.lindisp:  # Use linear sampling in depth space
            z_samp = near * (1 - z_steps) + far * z_steps  # (B, Kf)
        else:  # Use linear sampling in disparity space
            z_samp = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)  # (B, Kf)
        return z_samp

    def sample_fine_depth(self, rays, depth):
        """
        Sample around specified depth
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param depth (B)
        :return (B, Kfd)
        """
        #print("self.n_fine_depth:", self.n_fine_depth)
        z_samp = depth.unsqueeze(1).repeat((1, self.n_fine_depth))
        z_samp += torch.randn_like(z_samp) * self.depth_std
        # Clamp does not support tensor bounds
        z_samp = torch.max(torch.min(z_samp, rays[:, -1:]), rays[:, -2:-1])
        return z_samp

    def query_voxel(self, model, sb, device, render=True):
        resolution = 128
        #uv1, uv2, uv3 = torch.meshgrid(torch.arange(resolution, dtype=torch.float32,), torch.arange(resolution, dtype=torch.float32), torch.arange(resolution, dtype=torch.float32))
        uv1, uv2, uv3 = torch.meshgrid(torch.arange(resolution, dtype=torch.float32,), torch.arange(resolution, dtype=torch.float32), torch.arange(resolution, dtype=torch.float32), indexing='ij')
        voxel_space = torch.stack([uv1, uv2, uv3], dim=3) * (2.0/resolution) - 1.0 + (1./resolution)
        voxel_space[:,:,:,0:2] = voxel_space[:,:,:,0:2]  / 2.0
        voxel_space[:,:,:,2] = -voxel_space[:,:,:,2] / 2.0
        points = voxel_space.reshape(128*128, 128, 3).to(device)
        z_samp = -points[:,:,2] + 2.7
        #z_samp = -points[:,:,2]
        B, K = z_samp.shape

        #  if far:
        #      delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # infty (B, 1)

        val_all = []
        if sb > 0:
            points = points.reshape(
                sb, -1, 3
            )  # (SB, B'*K, 3) B' is real ray batch size
            eval_batch_size = (self.eval_batch_size - 1) // sb + 1
            eval_batch_dim = 1
        else:
            eval_batch_size = self.eval_batch_size
            eval_batch_dim = 0

        split_points = torch.split(points, eval_batch_size, dim=eval_batch_dim)

        for pnts in split_points:
            val_all.append(model(pnts*2))
            
        points = None
        viewdirs = None
        # (B*K, 4) OR (SB, B'*K, 4)
        out = torch.cat(val_all, dim=eval_batch_dim)
        out = out.reshape(B, K, -1)  # (B, K, 4 or 5)

        sigmas = out[..., 0]
        rgbs = out[..., 1:4]

        sigmas = sigmas.unsqueeze(-1)
        z_samp = z_samp.unsqueeze(-1)
        rgb_final, depth_final, weights = self.ray_marcher(colors=rgbs.unsqueeze(0), densities=sigmas.unsqueeze(0), depths=z_samp.unsqueeze(0), rendering_options=None)
        rgb_final = rgb_final.squeeze(0)
        depth_final = depth_final.squeeze(0)
        
        coarse_composite = (weights, rgb_final, depth_final,)
        outputs = DotMap(
            coarse=self._format_outputs(
                coarse_composite, sb, want_weights=False,
            ),
        )
        return outputs

    
    def composite(self, model, rays, z_samp, coarse=True, sb=0, want_gt_out=False):
        """
        Render RGB and depth for each ray using NeRF alpha-compositing formula,
        given sampled positions along each ray (see sample_*)
        :param model should return (B, (r, g, b, sigma)) when called with (B, (x, y, z))
        should also support 'coarse' boolean argument
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param z_samp z positions sampled for each ray (B, K)
        :param coarse whether to evaluate using coarse NeRF
        :param sb super-batch dimension; 0 = disable
        :return weights (B, K), rgb (B, 3), depth (B)
        """
        with profiler.record_function("renderer_composite"):
            B, K = z_samp.shape

            # (B, K, 3)
            # [16384, num_sample, 3]
            points = rays[:, None, :3] + z_samp.unsqueeze(2) * rays[:, None, 3:6]
            points = points.reshape(-1, 3)  # (B*K, 3)
            #print("points shape: ", points.shape)

            val_all = []
            if want_gt_out:
                val_gt_all = []
            if sb > 0:
                #print("sb:",sb)
                points = points.reshape(
                    sb, -1, 3
                )  # (SB, B'*K, 3) B' is real ray batch size
                eval_batch_size = (self.eval_batch_size - 1) // sb + 1
                eval_batch_dim = 1
            else:
                eval_batch_size = self.eval_batch_size
                eval_batch_dim = 0

            split_points = torch.split(points, eval_batch_size, dim=eval_batch_dim)
            if want_gt_out:
                for pnts in split_points:
                    val_all.append(model(pnts*2))
                    with torch.no_grad():
                        val_gt_all.append(model.forward_gt(pnts*2))
            else:
                for pnts in split_points:
                    val_all.append(model(pnts*2))
            
            points = None
            viewdirs = None
            # (B*K, 4) OR (SB, B'*K, 4)
            out = torch.cat(val_all, dim=eval_batch_dim)
            out = out.reshape(B, K, -1)  # (B, K, 4 or 5)

            if want_gt_out:
                out_gt = torch.cat(val_gt_all, dim=eval_batch_dim)
                out_gt = out_gt.reshape(B, K, -1)  # (B, K, 4 or 5)

            sigmas = out[..., 0]
            rgbs = out[..., 1:4]

            sigmas = sigmas.unsqueeze(-1)
            z_samp = z_samp.unsqueeze(-1)
            rgb_final, depth_final, weights = self.ray_marcher(colors=rgbs.unsqueeze(0), densities=sigmas.unsqueeze(0), depths=z_samp.unsqueeze(0), rendering_options=None)
            rgb_final = rgb_final.squeeze(0)
            depth_final = depth_final.squeeze(0).squeeze(1)
            weights = weights.squeeze(0).squeeze(2)
            if self.white_bkgd:
                # White background
                pix_alpha = weights.sum(dim=1)  # (B), pixel alpha
                rgb_final = rgb_final + 1 - pix_alpha.unsqueeze(-1)  # (B, 3)
            if want_gt_out:
                return (
                weights,
                rgb_final,
                depth_final,
                out,
                ), out_gt
            return (
                weights,
                rgb_final,
                depth_final,
                out, 
            )
    
    def forward(
        self, model, rays, want_weights=False, want_gt_out=False
    ):
        ## LFL test
        #mlp_eg3d = torch.tensor(np.load('./visuals/face_30000_sketch/face_30000_1/eg3d/sample_features_save.npy')).cuda(1)
        #mlp_eg3d = torch.mean(mlp_eg3d, dim=1)
        ##print("mlp_eg3d:", mlp_eg3d.shape)
        #mlp_eg3d_out = model.decoder(mlp_eg3d)
        #rgb = torch.sigmoid(mlp_eg3d_out[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        #sigma = mlp_eg3d_out[..., 0:1]
        #print("rgb:", rgb.shape)
        #print("sigma:", sigma.shape)
        #eg3d_ori_path = './visuals/face_30000_sketch/face_30000_1/eg3d/sample_features_save_out.npy'
        #eg3d_ori_output = np.load(eg3d_ori_path, allow_pickle=True).item()
        #print(rgb.cpu().detach() - eg3d_ori_output['rgb'])
        #print(sigma.cpu().detach() - eg3d_ori_output['sigma'])
        """
        :model nerf model, should return (SB, B, (r, g, b, sigma))
        when called with (SB, B, (x, y, z)), for multi-object:
        SB = 'super-batch' = size of object batch,
        B  = size of per-object ray batch.
        Should also support 'coarse' boolean argument for coarse NeRF.
        :param rays ray spec [origins (3), directions (3), near (1), far (1)] (SB, B, 8)
        :param want_weights if true, returns compositing weights (SB, B, K)
        :return render dict
        """
        with profiler.record_function("renderer_forward"):
            if self.sched is not None and self.last_sched.item() > 0:
                self.n_coarse = self.sched[1][self.last_sched.item() - 1]
                self.n_fine = self.sched[2][self.last_sched.item() - 1]

            assert len(rays.shape) == 3
            superbatch_size = rays.shape[0]
            #print("rays: ", rays.shape)
            rays = rays.reshape(-1, 8)  # (SB * B, 8)

            z_coarse = self.sample_coarse(rays)  # (B, Kc)
            if want_gt_out:
                coarse_composite, out_gt_coarse = self.composite(
                model, rays, z_coarse, coarse=True, sb=superbatch_size, want_gt_out=want_gt_out
                )
            else:
                coarse_composite = self.composite(
                    model, rays, z_coarse, coarse=True, sb=superbatch_size,
                )

            outputs = DotMap(
                coarse=self._format_outputs(
                    coarse_composite, superbatch_size, want_weights=want_weights,
                ),
            )

            if want_gt_out:
                outputs.out_gt_coarse = out_gt_coarse

            if self.using_fine:
                all_samps = [z_coarse]
                if self.n_fine - self.n_fine_depth > 0:
                    all_samps.append(
                        self.sample_fine(rays, coarse_composite[0].detach())
                    )  # (B, Kf - Kfd)
                if self.n_fine_depth > 0:
                    all_samps.append(
                        self.sample_fine_depth(rays, coarse_composite[2])
                    )  # (B, Kfd)
                z_combine = torch.cat(all_samps, dim=-1)  # (B, Kc + Kf)
                z_combine_sorted, argsort = torch.sort(z_combine, dim=-1)
                if want_gt_out:
                    fine_composite, out_gt_fine = self.composite(
                        model, rays, z_combine_sorted, coarse=False, sb=superbatch_size,
                    )
                    outputs.out_gt_fine = out_gt_fine
                else:
                    fine_composite = self.composite(
                        model, rays, z_combine_sorted, coarse=False, sb=superbatch_size,
                    )
                outputs.fine = self._format_outputs(
                    fine_composite, superbatch_size, want_weights=want_weights,
                )

            return outputs

    def _format_outputs(
        self, rendered_outputs, superbatch_size, want_weights=False,
    ):
        weights, rgb, depth, out = rendered_outputs
        if superbatch_size > 0:
            #out = out.reshape(superbatch_size, -1, out.shape[-2], out.shape[-1])
            rgb = rgb.reshape(superbatch_size, -1, 3)
            depth = depth.reshape(superbatch_size, -1)
            weights = weights.reshape(superbatch_size, -1, weights.shape[-1])
        ret_dict = DotMap(rgb=rgb, depth=depth, out=out)
        if want_weights:
            ret_dict.weights = weights
        return ret_dict

    def sched_step(self, steps=1):
        """
        Called each training iteration to update sample numbers
        according to schedule
        """
        if self.sched is None:
            return
        self.iter_idx += steps
        while (
            self.last_sched.item() < len(self.sched[0])
            and self.iter_idx.item() >= self.sched[0][self.last_sched.item()]
        ):
            self.n_coarse = self.sched[1][self.last_sched.item()]
            self.n_fine = self.sched[2][self.last_sched.item()]
            print(
                "INFO: NeRF sampling resolution changed on schedule ==> c",
                self.n_coarse,
                "f",
                self.n_fine,
            )
            self.last_sched += 1

    @classmethod
    def from_conf(cls, conf, white_bkgd=False, lindisp=False, eval_batch_size=100000):
        return cls(
            conf.get_int("n_coarse", 128),
            conf.get_int("n_fine", 0),
            n_fine_depth=conf.get_int("n_fine_depth", 0),
            noise_std=conf.get_float("noise_std", 0.0),
            depth_std=conf.get_float("depth_std", 0.01),
            white_bkgd=conf.get_float("white_bkgd", white_bkgd),
            lindisp=lindisp,
            eval_batch_size=conf.get_int("eval_batch_size", eval_batch_size),
            sched=conf.get_list("sched", None),
        )

    def bind_parallel(self, net, gpus=None, simple_output=False):
        """
        Returns a wrapper module compatible with DataParallel.
        Specifically, it renders rays with this renderer
        but always using the given network instance.
        Specify a list of GPU ids in 'gpus' to apply DataParallel automatically.
        :param net A PixelNeRF network
        :param gpus list of GPU ids to parallize to. If length is 1,
        does not parallelize
        :param simple_output only returns rendered (rgb, depth) instead of the 
        full render output map. Saves data tranfer cost.
        :return torch module
        """
        wrapped = _RenderWrapper(net, self, simple_output=simple_output)
        if gpus is not None and len(gpus) > 1:
            print("Using multi-GPU", gpus)
            wrapped = torch.nn.DataParallel(wrapped, gpus, dim=1)
        return wrapped
