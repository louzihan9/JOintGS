# Code based on 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/image_utils.py
# License from 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md

#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import pathlib
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Union, List, BinaryIO
import random
import numpy as np
import cv2
import glob
def psnr(img1, img2, mask=None):
    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1:
            mask = mask.expand(3, -1, -1)
        mse = ((img1 - img2) ** 2 * mask).sum() / mask.sum()
    else:
        mse = ((img1 - img2) ** 2).mean()
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


@torch.no_grad()
def normalize_depth(depth, min=None, max=None):
    if depth.shape[0] == 1:
        depth = depth[0]
    
    if min is None:
        min = depth.min()

    if max is None:
        max = depth.max()
        
    depth = (depth - min) / (max - min)
    depth = 1.0 - depth
    return depth


@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    text_labels: Optional[List[str]] = None,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    
    grid = make_grid(tensor, **kwargs)
    txt_font = ImageFont.load_default()
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(im)
    draw.text((10, 10), text_labels, fill=(0, 0, 0), font=txt_font)
    im.save(fp, format=format)
    
    
def save_rgba_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
    format: Optional[str] = 'PNG',
    text_labels: Optional[List[str]] = None,
    **kwargs,
) -> None:
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    
    grid = make_grid(tensor, **kwargs)
    txt_font = ImageFont.load_default()
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    if text_labels is not None:
        draw = ImageDraw.Draw(im)
        draw.text((10, 10), text_labels, fill=(0, 0, 0), font=txt_font)
    im.save(fp, format=format)


def calculate_psnr_torch(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0):
    """
    计算两张图片之间的峰值信噪比 (PSNR)。
    输入: PyTorch Tensor (C, H, W)，值域通常是 [0, 1]。
    """
    # 确保张量在同一个设备上进行计算
    img1 = img1.float()
    img2 = img2.float()

    # 计算均方误差 (MSE)
    mse = torch.mean((img1 - img2) ** 2)

    if mse.item() == 0:
        return 100.0  # 图像完全相同

    # PSNR 公式: 20 * log10(MAX_I) - 10 * log10(MSE)
    psnr = 20 * torch.log10(torch.tensor(max_val, device=img1.device)) - 10 * torch.log10(mse)

    return psnr.item()

from lpips import LPIPS
lpips_fn = LPIPS(net="alex", pretrained=True).to('cuda')
def lpips(img1, img2):
    return lpips_fn(img1.clip(max=1), img2).mean().double()

def match_high_fidelity_patches(
    pred_patches,
    gt_patches,
    patch_offsets_np,
    PSNR_THRESHOLD = 20,
    PATCH_SIZE = 128):
    N = pred_patches.shape[0]

    all_matched_pred_points = []
    all_matched_gt_points = []

    # 初始化 ORB 特征检测器和 BFMatcher
    orb = cv2.ORB_create(nfeatures=10)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for i in range(N):
        pred_patch_t = pred_patches[i]
        gt_patch_t = gt_patches[i]
        offset_vector_np = patch_offsets_np[i]  # 当前 Patch 的 (x, y) 偏移量 (NumPy)

        psnr_score = psnr(pred_patch_t, gt_patch_t)

        if psnr_score < PSNR_THRESHOLD:  # 使用传入的阈值进行筛选
            continue

        # 4. 转换到 NumPy 和灰度图（仅在需要匹配时才进行 CPU 传输）

        # 从 (C, H, W) PyTorch Tensor 转换为 (H, W, C) NumPy 数组
        # 注意: 如果 pred_patch_t 在 GPU 上，此操作涉及 CPU 传输
        pred_patch_np = pred_patch_t.permute(1, 2, 0).mul(255).clamp(0, 255).byte().cpu().numpy()
        gt_patch_np = gt_patch_t.permute(1, 2, 0).mul(255).clamp(0, 255).byte().cpu().numpy()


        pred_gray = cv2.cvtColor(pred_patch_np, cv2.COLOR_RGB2GRAY)
        gt_gray = cv2.cvtColor(gt_patch_np, cv2.COLOR_RGB2GRAY)


        # 5. ORB 特征检测和描述符计算
        kp_pred, des_pred = orb.detectAndCompute(pred_gray, None)
        kp_gt, des_gt = orb.detectAndCompute(gt_gray, None)


        pred_patch_np_bgr = cv2.cvtColor(pred_patch_np, cv2.COLOR_RGB2BGR)
        gt_patch_np_bgr = cv2.cvtColor(gt_patch_np, cv2.COLOR_RGB2BGR)

        # 7. 默认设置为无匹配的拼接图
        # 这是没有角点或没有匹配时的默认输出
        patch_vis_np = np.concatenate([pred_patch_np_bgr, gt_patch_np_bgr], axis=1)
        print_msg = ", No Matches Detected"

        # 8. 检查并尝试匹配
        # 确保描述符存在且数量足够
        if des_pred is not None and des_gt is not None and len(kp_pred) >= 2 and len(kp_gt) >= 2:

            # 9. 特征匹配
            matches = bf.match(des_pred, des_gt)

            # 10. 提取匹配的关键点坐标 (Patch 内部坐标)
            src_pts = np.float32([kp_pred[m.queryIdx].pt for m in matches]).reshape(-1, 2)
            dst_pts = np.float32([kp_gt[m.trainIdx].pt for m in matches]).reshape(-1, 2)

            if src_pts.shape[0] > 0:
                # 匹配成功，生成 drawMatches 图像

                # 1. Patch 内部可视化所需的 KeyPoint 和 DMatch
                local_pred_keypoints = [cv2.KeyPoint(x=p[0], y=p[1], size=4.0) for p in src_pts]
                local_gt_keypoints = [cv2.KeyPoint(x=p[0], y=p[1], size=4.0) for p in dst_pts]
                local_matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(src_pts))]

                # 2. 绘制 Patch 匹配图
                # 覆盖默认的 patch_vis_np
                patch_vis_np = cv2.drawMatches(
                    img1=pred_patch_np_bgr,
                    keypoints1=local_pred_keypoints,
                    img2=gt_patch_np_bgr,
                    keypoints2=local_gt_keypoints,
                    matches1to2=local_matches,
                    outImg=None,
                    matchColor=(255, 0, 0),  # 蓝色连线
                    singlePointColor=(0, 255, 255),  # 黄色角点
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS | cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
                )


        # 11. 动态读取文件数并保存
        # !!! 这一步是关键：它位于 PSNR 检查之后，但在所有匹配检查之外 !!!
        # 无论 patch_vis_np 是默认拼接图还是匹配图，都会在这里保存。

        # 重新统计目录中的文件数量 (实现不依赖内部计数器)
        SAVE_DIR = 'vis'
        current_existing_files_png = glob.glob(os.path.join(SAVE_DIR, '*.png'))
        current_existing_files_PNG = glob.glob(os.path.join(SAVE_DIR, '*.PNG'))
        current_file_count = len(current_existing_files_png) + len(current_existing_files_PNG)

        new_filename_index = current_file_count + 1  # 新文件名索引 (n + 1)
        patch_new_filename = os.path.join(SAVE_DIR, f'{new_filename_index}_{psnr_score}.png')
        cv2.imwrite(patch_new_filename, patch_vis_np)



        if des_pred is None or des_gt is None:
            continue

        # 确保描述符数量足够
        if len(kp_pred) < 2 or len(kp_gt) < 2:
            continue

        # 6. 特征匹配
        matches = bf.match(des_pred, des_gt)

        # 7. 提取匹配的关键点坐标 (Patch 内部坐标)
        src_pts = np.float32([kp_pred[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([kp_gt[m.trainIdx].pt for m in matches]).reshape(-1, 2)







        if src_pts.shape[0] == 0:
            continue

        # 8. 坐标转换（加上 Patch 偏移）

        # 偏移向量 (x_col_offset, y_row_offset)
        # NumPy 广播机制：(M, 2) + (2,) -> (M, 2)
        matched_pred_points = src_pts + offset_vector_np
        matched_gt_points = dst_pts + offset_vector_np

        all_matched_pred_points.append(matched_pred_points)
        all_matched_gt_points.append(matched_gt_points)

    # 9. 组合所有 Patch 的匹配结果
    if not all_matched_pred_points:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    final_pred_points = np.concatenate(all_matched_pred_points, axis=0)
    final_gt_points = np.concatenate(all_matched_gt_points, axis=0)

    return final_pred_points, final_gt_points


def match_points(gt_image, pred_img,
                 max_distance_pixels = 20,  # 邻域筛选的距离阈值（像素）
                 lowe_ratio_thresh = 0.9):
    """
    计算两张极相近图像之间的相似点（特征点匹配）。

    Args:
        gt_image (torch.Tensor): 第一张图像 (Ground Truth)，形状为 (3, H, W)。
        pred_img (torch.Tensor): 第二张图像 (Prediction)，形状为 (3, H, W)。
        max_distance_pixels (int): 特征点匹配后的邻域筛选距离（像素），用于排除距离较远的匹配点。
        lowe_ratio_thresh (float): Lowe's Ratio Test 的阈值，用于高置信度匹配筛选（例如 SIFT）。

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - full_gt_points (torch.Tensor): 匹配成功的 gt_image 上的特征点坐标 (N, 2)。
            - full_pred_points (torch.Tensor): 匹配成功的 pred_img 上的特征点坐标 (N, 2)。
            (坐标格式为 [x, y]，即 [列, 行])
    """

    # 1. PyTorch Tensor 转换为 OpenCV 图像 (NumPy 数组)
    # 假设输入的张量是 RGB 格式，且值范围在 [0, 1] 或 [0, 255]

    def tensor_to_cv_img(tensor: torch.Tensor) -> np.ndarray:
        # 移至 CPU，分离计算图，转为 NumPy
        img_np = tensor.detach().cpu().numpy()
        # 形状从 (C, H, W) 转换为 (H, W, C)
        img_np = np.transpose(img_np, (1, 2, 0))

        # 归一化处理：如果值范围在 [0, 1]，则转换为 [0, 255] 并转为 uint8
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        # OpenCV 默认使用 BGR，如果输入是 RGB，需要转换
        # 特征点检测通常在灰度图上进行，但这里先保留 BGR/RGB 转换以确保颜色通道正确
        # 如果您的张量是 BGR，请跳过此步骤或使用 cv2.COLOR_BGR2GRAY
        # 保持一致性，统一转换为灰度图
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # 转换图像

    gt_gray = tensor_to_cv_img(gt_image)
    pred_gray = tensor_to_cv_img(pred_img)


    sift = cv2.SIFT_create()


    # kp 是关键点 (KeyPoint 对象)，des 是描述符 (Descriptor)
    kp_gt, des_gt = sift.detectAndCompute(gt_gray, None)
    kp_pred, des_pred = sift.detectAndCompute(pred_gray, None)

    if des_gt is None or des_pred is None:
        return None, None

    # 3. 特征点匹配
    # 使用 BFMatcher (Brute-Force Matcher) 配合 K-Nearest Neighbors (k=2)
    # NORM_L2 适用于 SIFT 等基于浮点数的描述符
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # 找到 k=2 个最佳匹配
    matches_knn = matcher.knnMatch(des_gt, des_pred, k=2)

    # 4. 高置信度筛选：Lowe's Ratio Test
    # Lowe's Ratio Test (m.distance < lowe_ratio_thresh * n.distance)
    # 只保留第一个最佳匹配与第二个最佳匹配距离比值小于阈值的匹配点，这能排除许多不明确的匹配
    good_matches = []
    for m, n in matches_knn:
        if m.distance < lowe_ratio_thresh * n.distance:
            good_matches.append(m)

    # 5. 邻域筛选 (自定义的距离控制项)
    final_gt_points = []
    final_pred_points = []

    for m in good_matches:
        # 获取匹配点在两张图像中的坐标 (x, y)
        pt_gt = kp_gt[m.queryIdx].pt  # gt_image 中的特征点坐标 (x, y)
        pt_pred = kp_pred[m.trainIdx].pt  # pred_img 中的特征点坐标 (x, y)

        # 计算两点之间的欧式距离 (L2 距离)
        # 假设两张图片极相近，匹配点在空间位置上也应该非常接近
        distance = np.sqrt((pt_gt[0] - pt_pred[0]) ** 2 + (pt_gt[1] - pt_pred[1]) ** 2)

        # 仅保留距离在 max_distance_pixels 内的匹配点
        if distance < max_distance_pixels:
            final_gt_points.append(pt_gt)
            final_pred_points.append(pt_pred)

    # 6. 转换为 torch.Tensor 输出
    full_gt_points = torch.tensor(final_gt_points, dtype=torch.float32)
    full_pred_points = torch.tensor(final_pred_points, dtype=torch.float32)

    return full_pred_points, full_gt_points