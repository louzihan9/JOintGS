#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import os
import glob
import shutil
import time

import torch
import itertools
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
from lpips import LPIPS
from loguru import logger
import open3d as o3d
from torch.optim.lr_scheduler import StepLR
from jointgs.datasets.utils import (
    get_rotating_camera,
    get_smpl_canon_params,
    get_smpl_static_params,
    get_static_camera
)
from jointgs.losses.utils import ssim
from jointgs.datasets.neuman import NeumanDataset
from jointgs.datasets.emdb import EmdbDataset
from jointgs.datasets.EMDB_refine import EMDBrefineDataset

from jointgs.losses.loss import HumanSceneLoss
from jointgs.models.hugs_trimlp import HUGS_TRIMLP
from jointgs.models import SceneGS
from jointgs.models.train_data import train_data
from jointgs.utils.init_opt import optimize_init
from jointgs.renderer.gs_renderer import render_human_scene, render
from jointgs.utils.vis import save_ply
from jointgs.utils.image import psnr, lpips
from jointgs.utils.general import RandomIndexIterator, load_human_ckpt, save_log_images, create_video, SplitRandomIndexIterator


def get_dataset(cfg):
    if cfg.dataset.name == 'neuman':
        logger.info(f'Loading NeuMan dataset {cfg.dataset.seq}-train')
        dataset = NeumanDataset(cfg)
    elif cfg.dataset.name == 'emdb':
        logger.info(f'Loading emdb dataset {cfg.dataset.seq}-train')
        dataset = EmdbDataset(cfg)
    elif cfg.dataset.name == 'emdb_refine':
        logger.info(f'Loading emdb dataset {cfg.dataset.seq}-train')
        dataset = EMDBrefineDataset(cfg)
    else:
        raise "暂不支持该数据集"
    return dataset


class GaussianOptimTrainer():
    def __init__(self, cfg) -> None:
        self.cfg = cfg

        dataset = get_dataset(cfg)
        self.train_data = train_data(dataset).to(cfg.torch_device)
        self.RandomIndexIterator = SplitRandomIndexIterator(len(self.train_data), 1.0)

        self.eval_metrics = {}
        self.lpips = LPIPS(net="alex", pretrained=True).to('cuda')

        cfg_stage = cfg.train
        # get models
        self.scene_gs = SceneGS()
        pcd = self.train_data.points3D
        spatial_lr_scale = self.train_data.radius
        self.scene_gs.create_from_pcd(pcd, spatial_lr_scale)

        init_betas = self.train_data.betas
        self.human_gs = HUGS_TRIMLP(
            n_subdivision=cfg_stage.human.n_subdivision,
            use_surface=cfg_stage.human.use_surface,
            init_2d=cfg_stage.human.init_2d,
            rotate_sh=cfg_stage.human.rotate_sh,
            isotropic=cfg_stage.human.isotropic,
            init_scale_multiplier=cfg_stage.human.init_scale_multiplier,
            n_features=32,
            use_deformer=cfg_stage.human.use_deformer,
            disable_posedirs=cfg_stage.human.disable_posedirs,
            triplane_res=cfg_stage.human.triplane_res,
            betas=init_betas[0]
        )
        self.human_gs.create_betas(init_betas[0], cfg_stage.human.optim_betas)
        # self.human_gs.initialize()
        self.human_gs = optimize_init(cfg, self.human_gs, num_steps=5000)

        init_smpl_global_orient = self.train_data.global_orients
        init_smpl_body_pose = self.train_data.poses
        init_smpl_trans = self.train_data.transls
        init_betas = self.train_data.betas
        init_scales = self.train_data.scales


        self.human_gs.create_betas(init_betas[0], True)
        self.human_gs.create_body_pose(init_smpl_body_pose, True)
        self.human_gs.create_global_orient(init_smpl_global_orient, True)
        self.human_gs.create_transl(init_smpl_trans, True)
        self.human_gs.create_smpl_scale(init_scales, True)

        bg_color = cfg.bg_color
        if bg_color == 'white':
            self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        elif bg_color == 'black':
            self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        else:
            raise ValueError(f"Unknown background color {bg_color}")

        self.cam_pose_optimizer, self.cam_scheduler = self.create_cam_pose_optimizer(self.train_data.cam_rots, self.train_data.cam_transls)



    def train(self):

        render_mode = 'human_scene'
        cfg_stage = self.cfg.train
        pbar = tqdm(range(cfg_stage.num_steps), desc="Training")
        loss_fn = HumanSceneLoss(
            self.cfg
        )
        self.scene_gs.setup_optimizer(cfg=cfg_stage.scene.lr)
        self.human_gs.setup_optimizer(cfg_stage=cfg_stage)
        random_iterator = self.RandomIndexIterator.get_train_iterator()
        for t_iter in range(1, cfg_stage.num_steps + 1):

            rnd_idx = next(random_iterator)
            data = self.train_data[rnd_idx]
            scene_gs_out, human_gs_out = None, None
            if render_mode in ['scene', 'human_scene']:
                self.scene_gs.update_learning_rate(t_iter)
                scene_gs_out = self.scene_gs.forward()

            if render_mode in ['human', 'human_scene']:
                self.human_gs.update_learning_rate(t_iter)
                human_gs_out = self.human_gs.forward(
                    dataset_idx=rnd_idx,
                )

            bg_color = torch.rand(3, dtype=torch.float32, device="cuda")

            render_pkg, render_human_pkg, render_scene_pkg = None, None, None

            render_human_pkg = render_human_scene(
                data=data,
                human_gs_out=human_gs_out,
                scene_gs_out=scene_gs_out,
                bg_color=bg_color,
                render_mode='human',
            )

            render_scene_pkg = render_human_scene(
                data=data,
                human_gs_out=human_gs_out,
                scene_gs_out=scene_gs_out,
                bg_color=bg_color,
                render_mode='scene',
            )

            loss, loss_dict, loss_extras = loss_fn(
                data,
                render_pkg,
                render_human_pkg,
                render_scene_pkg,
                human_gs_out,
                render_mode=render_mode,
                bg_color=bg_color,
            )

            loss.backward()
            loss_dict['loss'] = loss

            if t_iter % 10 == 0:
                postfix_dict = {
                    "#hp": f"{self.human_gs.n_gs / 1000 if self.human_gs else 0:.1f}K",
                    "#sp": f"{self.scene_gs.get_xyz.shape[0] / 1000 if self.scene_gs else 0:.1f}K",
                    'h_sh_d': self.human_gs.active_sh_degree,
                    's_sh_d': self.scene_gs.active_sh_degree if self.scene_gs else 0,
                }
                for k, v in loss_dict.items():
                    postfix_dict["l_" + k] = f"{v.item():.4f}"

                pbar.set_postfix(postfix_dict)
                pbar.update(10)


            if render_scene_pkg is not None:
                render_scene_pkg['scene_viewspace_points'] = render_scene_pkg['viewspace_points']
                render_scene_pkg['scene_viewspace_points'].grad = render_scene_pkg['viewspace_points'].grad
                with torch.no_grad():
                    self.scene_densification(
                        cfg_stage=cfg_stage,
                        visibility_filter=render_scene_pkg['scene_visibility_filter'],
                        radii=render_scene_pkg['scene_radii'],
                        viewspace_point_tensor=render_scene_pkg['scene_viewspace_points'],
                        iteration=t_iter,
                    )
                self.scene_gs.optimizer.step()
                self.scene_gs.optimizer.zero_grad(set_to_none=True)

            if render_human_pkg is not None:
                render_human_pkg['human_viewspace_points'] = render_human_pkg['viewspace_points']
                render_human_pkg['human_viewspace_points'].grad = render_human_pkg['viewspace_points'].grad
                with torch.no_grad():
                    self.human_densification(
                        cfg_stage=cfg_stage,
                        human_gs_out=human_gs_out,
                        visibility_filter=render_human_pkg['human_visibility_filter'],
                        radii=render_human_pkg['human_radii'],
                        viewspace_point_tensor=render_human_pkg['human_viewspace_points'],
                        iteration=t_iter,
                    )
                    self.human_gs.optimizer.step()
                    self.human_gs.optimizer.zero_grad(set_to_none=True)

            if t_iter == cfg_stage.num_steps:
                pbar.close()

            if t_iter % 15000 == 0 or t_iter == 1:
                self.valuation(['human','scene','human_scene'], t_iter)

            if t_iter % 1000 == 0:
                self.human_gs.oneupSHdegree()
                self.scene_gs.oneupSHdegree()

            if t_iter > 5000:
                if self.cam_pose_optimizer is not None:
                    self.cam_pose_optimizer.step()
                    self.cam_pose_optimizer.zero_grad(set_to_none=True)
                    self.cam_scheduler.step()

        self.save_optimized_cam_pose(self.cfg)
        self.human_gs.save_optimized_smpl_param(self.cfg)
        self.save_ckpt(render_mode)


    def save_ckpt(self, render_mode='human_scene', iter=None):
        iter_s = 'final' if iter is None else f'{iter:06d}'
        if render_mode in ['human', 'human_scene']:
            torch.save(self.human_gs.state_dict(), f'{self.cfg.logdir}/ckpt/human_{iter_s}.pth')

        if render_mode in ['scene', 'human_scene']:
            torch.save(self.scene_gs.state_dict(), f'{self.cfg.logdir_ckpt}/scene_{iter_s}.pth')
            self.scene_gs.save_ply(f'{self.cfg.logdir}/meshes/scene_{iter_s}_splat.ply')

        logger.info(f'Saved checkpoint {iter_s}')

    def scene_densification(self, cfg_stage, visibility_filter, radii, viewspace_point_tensor, iteration):
        self.scene_gs.max_radii2D[visibility_filter] = torch.max(
            self.scene_gs.max_radii2D[visibility_filter],
            radii[visibility_filter]
        )
        self.scene_gs.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if ((iteration % cfg_stage.scene.opacity_reset_interval) > cfg_stage.scene.opacity_reset_interval / 4
                and iteration % cfg_stage.scene.densification_interval == 0):
            size_threshold = 20 if iteration > cfg_stage.scene.opacity_reset_interval else None
            self.scene_gs.densify_and_prune(
                cfg_stage.scene.densify_grad_threshold,
                min_opacity=cfg_stage.scene.prune_min_opacity,
                extent=self.train_data.radius,
                max_screen_size=size_threshold,
                max_n_gs=cfg_stage.scene.max_n_gaussians,
            )


        is_white = self.bg_color.sum().item() == 3.

        # if iteration % cfg_stage.scene.opacity_reset_interval == 0 or (is_white and iteration == cfg_stage.scene.densify_from_iter):
        if iteration == cfg_stage.scene.opacity_reset_interval:
            logger.info(f"[{iteration:06d}] Resetting opacity!!!")
            self.scene_gs.reset_opacity()

    def human_densification(self, cfg_stage, human_gs_out, visibility_filter, radii, viewspace_point_tensor, iteration):
        self.human_gs.max_radii2D[visibility_filter] = torch.max(
            self.human_gs.max_radii2D[visibility_filter],
            radii[visibility_filter]
        )

        self.human_gs.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if iteration > cfg_stage.human.densify_from_iter and iteration % cfg_stage.human.densification_interval == 0:
            size_threshold = 20
            self.human_gs.densify_and_prune(
                human_gs_out,
                cfg_stage.human.densify_grad_threshold,
                min_opacity=cfg_stage.human.prune_min_opacity,
                extent=cfg_stage.human.densify_extent,
                max_screen_size=size_threshold,
                max_n_gs=cfg_stage.human.max_n_gaussians,
            )

    def valuation(self, logger_type, iteration):
        if logger_type is None:
            logger_type = ['human','human_scene']

        with torch.no_grad():

            psnr_score_list = []
            ssim_score_list = []
            lpips_score_list = []

            scene_psnr_score_list = []
            scene_ssim_score_list = []
            scene_lpips_score_list = []

            human_psnr_score_list = []
            human_ssim_score_list = []
            human_lpips_score_list = []

            for t_iter in range(len(self.RandomIndexIterator.get_train_indices())):
                data = self.train_data[t_iter]

                human_gs_out = self.human_gs.forward(
                    dataset_idx=t_iter,
                )
                scene_gs_out = self.scene_gs.forward()

                bg_color = torch.rand(3, dtype=torch.float32, device="cuda")

                if 'human' in logger_type:
                    human_pkg = render_human_scene(
                        data=data,
                        human_gs_out=human_gs_out,
                        scene_gs_out=scene_gs_out,
                        bg_color=bg_color,
                        render_mode='human',
                    )
                    mask = data['ground_truth']['mask'].unsqueeze(0)
                    pred_img = human_pkg['render'] * mask
                    gt_img = data['ground_truth']['gt_image'] * mask
                    human_psnr_score_list.append(psnr(pred_img, gt_img).mean().item())
                    human_ssim_score_list.append(ssim(pred_img, gt_img).mean().item())
                    human_lpips_score_list.append(lpips(pred_img.clip(max=1), gt_img).mean().item())
                    save_log_images(gt_img, pred_img, f'{self.cfg.logdir}/optim/human/{iteration}/{t_iter:06d}.png')


                if 'scene' in logger_type:
                    scene_pkg = render_human_scene(
                        data=data,
                        human_gs_out=human_gs_out,
                        scene_gs_out=scene_gs_out,
                        bg_color=bg_color,
                        render_mode='scene'
                    )
                    pred_img = scene_pkg['render'] * (1-mask)
                    gt_img = data['ground_truth']['gt_image'] * (1-mask)
                    scene_psnr_score_list.append(psnr(pred_img, gt_img).mean().item())
                    scene_ssim_score_list.append(ssim(pred_img, gt_img).mean().item())
                    scene_lpips_score_list.append(lpips(pred_img.clip(max=1), gt_img).mean().item())
                    save_log_images(gt_img, pred_img, f'{self.cfg.logdir}/optim/scene/{iteration}/{t_iter:06d}.png')

                if 'human_scene' in logger_type:
                    human_scene_pkg = render_human_scene(
                        data=data,
                        human_gs_out=human_gs_out,
                        scene_gs_out=scene_gs_out,
                        bg_color=bg_color,
                        render_mode='human_scene'
                    )
                    pred_img = human_scene_pkg['render']
                    gt_img = data['ground_truth']['gt_image']
                    psnr_score_list.append(psnr(pred_img, gt_img).mean().item())
                    ssim_score_list.append(ssim(pred_img, gt_img).mean().item())
                    lpips_score_list.append(lpips(pred_img.clip(max=1), gt_img).mean().item())
                    save_log_images(gt_img, pred_img, f'{self.cfg.logdir}/optim/human_scene/{iteration}/{t_iter:06d}.png')

            logger.info(f'\tvaluation-------------------------------------------')
            if 'human' in logger_type:
                logger.info(f'\t\tmode:human')
                logger.info(f"\t\tpsnr:{sum(human_psnr_score_list) / len(human_psnr_score_list)}, "
                            f"ssim:{sum(human_ssim_score_list) / len(human_ssim_score_list)}, "
                            f"lpips:{sum(human_lpips_score_list) / len(human_lpips_score_list)}")

            if 'scene' in logger_type:
                logger.info(f'\t\tmode:scene')
                logger.info(f"\t\tpsnr:{sum(scene_psnr_score_list) / len(scene_psnr_score_list)}, "
                            f"ssim:{sum(scene_ssim_score_list) / len(scene_ssim_score_list)}, "
                            f"lpips:{sum(scene_lpips_score_list) / len(scene_lpips_score_list)}")

            if 'human_scene' in logger_type:
                logger.info(f'\t\tmode:human_scene')
                logger.info(f"\t\tpsnr:{sum(psnr_score_list) / len(psnr_score_list)}, "
                            f"ssim:{sum(ssim_score_list) / len(ssim_score_list)}, "
                            f"lpips:{sum(lpips_score_list) / len(lpips_score_list)}")

    def create_cam_pose_optimizer(self, cam_rots, cam_transls):
        params = [
            {'params': [cam_rots], 'lr': self.cfg.train.lr_cam_rot, "name": "cam_rots"},
            {'params': cam_transls, 'lr': self.cfg.train.lr_cam_transl, 'name': 'cam_transls'},
        ]
        cam_optimizer = torch.optim.SGD(
            params=params,
            lr=0.0,
            momentum=0.0
        )
        cam_scheduler = StepLR(
            optimizer=cam_optimizer,
            step_size=1000,
            gamma=0.5
        )
        return cam_optimizer, cam_scheduler

    def save_optimized_cam_pose(self, cfg):
        with torch.no_grad():
            cam_rot = self.train_data.cam_rots.data.cpu().numpy()
            cam_transl = self.train_data.cam_transls.data.cpu().numpy()

            cam_pose = {
                'cam_rot': cam_rot,
                'cam_transl': cam_transl}

            path1 = f'data/{cfg.dataset.name}/{cfg.dataset.seq}/cam_optimized_by_model.npz'
            np.savez(path1, **cam_pose)

            path2 = f'{cfg.logdir}/cam_optimized_by_model.npz'
            np.savez(path2, **cam_pose)








