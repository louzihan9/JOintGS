#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch
from lpips import LPIPS
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
from jointgs.utils.sampler import PatchSampler
import numpy as np
from .utils import l1_loss, ssim
from jointgs.utils.image import psnr, match_high_fidelity_patches, match_points
import os
import cv2
import glob
import time

class HumanSceneLoss(nn.Module):
    def __init__(
        self,
        cfg,
        bg_color='white',
    ):
        super(HumanSceneLoss, self).__init__()
        loss = cfg.train.loss
        self.l_ssim_w = loss.ssim_w
        self.l_l1_w = loss.l1_w
        self.l_lpips_w = loss.lpips_w
        self.use_patches = loss.use_patches
        num_patches = loss.num_patches
        patch_size = loss.patch_size

        human_regular = cfg.train.human.regular
        self.r_normal = human_regular.r_normal
        self.r_tangent = human_regular.r_tangent
        self.r_projection = human_regular.r_projection

        self.bg_color = bg_color
        self.lpips = LPIPS(net="vgg", pretrained=True).to('cuda')
    
        for param in self.lpips.parameters(): param.requires_grad=False
        
        if self.use_patches:
            self.human_patch_sampler = PatchSampler(num_patch=num_patches, patch_size=patch_size, ratio_mask=1.0, dilate=0)

    def forward(
        self, 
        data, 
        render_pkg,
        render_human_pkg,
        render_scene_pkg,
        human_gs_out,
        render_mode,
        bg_color=None,
    ):
        loss_dict = {}
        extras_dict = {}
        
        if bg_color is not None:
            self.bg_color = bg_color


        if render_pkg is not None:
            gt_image = data['ground_truth']['gt_image']
            pred_img = render_pkg['render']
            Ll1 = l1_loss(pred_img, gt_image)
            loss_dict['l1'] = self.l_l1_w * Ll1

            loss_ssim = 1.0 - ssim(pred_img, gt_image)
            loss_dict['ssim'] = self.l_ssim_w * loss_ssim

        if render_scene_pkg is not None:
            mask = data['ground_truth']['mask'].unsqueeze(0)
            gt_image = data['ground_truth']['gt_image'] * (1-mask)
            pred_scene_img = render_scene_pkg['render'] * (1-mask)

            Ll1 = l1_loss(pred_scene_img, gt_image)
            loss_dict['scene_l1'] = self.l_l1_w * Ll1

            loss_ssim = 1.0 - ssim(pred_scene_img, gt_image)
            loss_dict['scene_ssim'] = self.l_ssim_w * loss_ssim

        if render_human_pkg is not None:
            mask = data['ground_truth']['mask'].unsqueeze(0)
            gt_image = data['ground_truth']['gt_image'] * mask + bg_color[:, None, None] * (1. - mask)
            pred_human_img = render_human_pkg['render']

            Ll1 = l1_loss(pred_human_img, gt_image, mask)
            loss_dict['human_l1'] = self.l_l1_w * Ll1

            loss_ssim = 1.0 - ssim(pred_human_img, gt_image, mask=mask)
            loss_dict['human_ssim'] = self.l_ssim_w * loss_ssim

            _, pred_patches, gt_patches, patch_offsets_np = self.human_patch_sampler.sample(mask, pred_human_img, gt_image)
            loss_lpips = self.lpips(pred_patches.clip(max=1), gt_patches).mean()
            loss_dict['human_lpips'] = self.l_lpips_w * loss_lpips


            init_points_loss = F.mse_loss(
                human_gs_out['init_points'],
                human_gs_out['t_pose_verts']
            )
            loss_dict['r_p1'] = 0.4 * init_points_loss


        loss = 0.0
        for k, v in loss_dict.items():
            loss += v
        
        return loss, loss_dict, extras_dict




