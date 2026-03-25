# Code based on COTR: https://github.com/ubc-vision/COTR/blob/master/COTR/datasets/colmap_helper.py
# License from COTR: https://github.com/ubc-vision/COTR/blob/master/LICENSE


import os
import re
from collections import namedtuple

import numpy as np
from tqdm import tqdm

from .geometry.basics import Translation, Rotation
from .cameras.camera_pose import CameraPose
from .cameras.pinhole_camera import PinholeCamera
from .cameras import captures as captures_module
from .scenes import scene as scene_module
from scipy.spatial.transform import Rotation as scipy_Rotation

ImageMeta = namedtuple('ImageMeta', ['image_id', 'camera_pose', 'camera_id', 'image_path'])


class ColmapAsciiReader():
    def __init__(self):
        pass

    @classmethod
    def read_scene(cls, cfg, scene_dir, images_dir, tgt_size=None, order='default'):
        point_cloud_path = os.path.join(scene_dir, 'points3D.txt')
        cameras_path = os.path.join(scene_dir, 'cameras.txt')
        images_path = os.path.join(scene_dir, 'images.txt')
        captures = cls.read_captures(cfg, images_path, cameras_path, images_dir, tgt_size, order)
        point_cloud = cls.read_point_cloud(point_cloud_path)
        scene = scene_module.ImageFileScene(captures, point_cloud)
        return scene

    @staticmethod
    def read_point_cloud(points_txt_path):
        with open(points_txt_path, "r") as fid:
            line = fid.readline()
            assert line == '# 3D point list with one line of data per point:\n'
            line = fid.readline()
            assert line == '#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n'
            line = fid.readline()
            assert re.search('^# Number of points: \d+, mean track length: [-+]?\d*\.\d+|\d+\n$', line)
            num_points, mean_track_length = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            num_points = int(num_points)
            mean_track_length = float(mean_track_length)

            xyz = np.zeros((num_points, 3), dtype=np.float32)
            rgb = np.zeros((num_points, 3), dtype=np.float32)

            for i in tqdm(range(num_points), desc='reading point cloud'):
                elems = fid.readline().split()
                xyz[i] = list(map(float, elems[1:4]))
                rgb[i] = list(map(float, elems[4:7]))
            pcd = np.concatenate([xyz, rgb], axis=1)
        return pcd

    @classmethod
    def read_captures(cls, cfg, images_txt_path, cameras_txt_path, images_dir, tgt_size, order='default'):
        captures = []


        cameras = cls.read_cameras(cameras_txt_path)
        images_meta = cls.read_images_meta(cfg, images_txt_path, images_dir)
        if order == 'default':
            keys = images_meta.keys()
        elif order == 'video':
            keys = []
            frames = []
            for k, v in images_meta.items():
                keys.append(k)
                frames.append(os.path.basename(v.image_path))
            keys = [x for _, x in sorted(zip(frames, keys))]
        else:
            raise ValueError(f'unknown order: {order}')
        for i, key in enumerate(keys):
            cur_cam_id = images_meta[key].camera_id
            cur_cam = cameras[cur_cam_id]
            cur_camera_pose = images_meta[key].camera_pose
            cur_image_path = images_meta[key].image_path
            if tgt_size is None:
                cap = captures_module.RGBPinholeCapture(cur_image_path, cur_cam, cur_camera_pose)
            else:
                cap = captures_module.ResizedRGBPinholeCapture(cur_image_path, cur_cam, cur_camera_pose, tgt_size)
            if order == 'video':
                cap.frame_id = {'frame_id': i, 'total_frames': len(images_meta)}
            captures.append(cap)
        return captures

    @classmethod
    def read_cameras(cls, cameras_txt_path):
        cameras = {}
        with open(cameras_txt_path, "r") as fid:
            line = fid.readline()
            assert line == '# Camera list with one line of data per camera:\n'
            line = fid.readline()
            assert line == '#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n'
            line = fid.readline()
            assert re.search('^# Number of cameras: \d+\n$', line)
            num_cams = int(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])

            for _ in tqdm(range(num_cams), desc='reading cameras'):
                elems = fid.readline().split()
                camera_id = int(elems[0])
                if elems[1] == 'SIMPLE_RADIAL':
                    width, height, focal_length, cx, cy, radial = list(map(float, elems[2:]))
                    cur_cam = PinholeCamera(width, height, focal_length, focal_length, cx, cy)
                elif elems[1] == 'PINHOLE':
                    width, height, fx, fy, cx, cy = list(map(float, elems[2:]))
                    cur_cam = PinholeCamera(width, height, fx, fy, cx, cy)
                elif elems[1] == 'OPENCV':
                    width, height, fx, fy, cx, cy, k1, k2, k3, k4 = list(map(float, elems[2:]))
                    cur_cam = PinholeCamera(width, height, fx, fy, cx, cy)
                else:
                    raise ValueError(f'unsupported camera: {elems[1]}')
                assert camera_id not in cameras
                cameras[camera_id] = cur_cam
        return cameras

    @classmethod
    def read_images_meta(cls, cfg, images_txt_path, images_dir):
        images_meta = {}
        print(images_txt_path)
        with open(images_txt_path, "r") as fid:
            line = fid.readline()
            assert line == '# Image list with two lines of data per image:\n'
            line = fid.readline()
            assert line == '#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n'
            line = fid.readline()
            assert line == '#   POINTS2D[] as (X, Y, POINT3D_ID)\n'
            line = fid.readline()
            assert re.search('^# Number of images: \d+, mean observations per image: [-+]?\d*\.\d+|\d+\n$', line)
            num_images, mean_ob_per_img = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            num_images = int(num_images)
            mean_ob_per_img = float(mean_ob_per_img)

            noise_power = cfg.noise_power

            for _ in tqdm(range(num_images), desc='reading images meta'):
                elems = fid.readline().split()
                assert len(elems) == 10
                line = fid.readline()
                image_path = os.path.join(images_dir, elems[9])

                sort_image_index = int(image_path[-9:-4])

                assert os.path.isfile(image_path), f'missing file: {image_path}'
                image_id = int(elems[0])
                qw, qx, qy, qz, tx, ty, tz = list(map(float, elems[1:8]))


                tx += np.random.normal(loc=0, scale=noise_power, size=[1])[0]
                ty += np.random.normal(loc=0, scale=noise_power, size=[1])[0]
                tz += np.random.normal(loc=0, scale=noise_power, size=[1])[0]

                qw, qx, qy, qz = cls.add_gaussian_noise_to_quaternion(qw, qx, qy, qz, noise_power)

                t = Translation(np.array([tx, ty, tz], dtype=np.float32))
                r = Rotation(np.array([qw, qx, qy, qz], dtype=np.float32))
                camera_pose = CameraPose(t, r)
                camera_id = int(elems[8])
                assert image_id not in images_meta, f'duplicated image, id: {image_id}, path: {image_path}'
                images_meta[image_id] = ImageMeta(image_id, camera_pose, camera_id, image_path)
        return images_meta

    @classmethod
    def add_gaussian_noise_to_quaternion(cls, qw, qx, qy, qz, rotation_noise_power):
        """
        将高斯噪声添加到四元数表示的旋转中。
        步骤：四元数 -> 旋转向量（轴角）-> 添加噪声 -> 新四元数 -> 提取 float 分量。

        参数:
        qw, qx, qy, qz (float): 原始四元数的四个分量 (w, x, y, z)。
        rotation_noise_power (float): 噪声的标准差 (sigma)，用于旋转向量。

        返回:
        tuple: (qw_noisy, qx_noisy, qy_noisy, qz_noisy)，均为 Python 原生浮点数。
        """

        # 1. 组合原始四元数并转换为 Rotation 对象
        # SciPy 使用 (x, y, z, w) 顺序
        original_quaternion = np.array([qx, qy, qz, qw])
        rotation = scipy_Rotation.from_quat(original_quaternion)

        # 2. 转换为旋转向量 (Rotation Vector / Angle-Axis)
        # 结果是一个 3D 向量，其方向是旋转轴，长度是旋转角度
        rotation_vector = rotation.as_rotvec()  # 形状 (3,)

        # 3. 生成高斯噪声并添加到旋转向量中
        # 噪声均值 loc=0，标准差 scale=rotation_noise_power
        # 由于 rotation_vector 是 (3,) 向量，我们生成一个相同形状的噪声向量
        noise_vector = np.random.normal(
            loc=0,
            scale=rotation_noise_power,
            size=rotation_vector.shape
        )

        noisy_rotation_vector = rotation_vector + noise_vector

        # 4. 将带噪的旋转向量转换回四元数
        noisy_rotation = scipy_Rotation.from_rotvec(noisy_rotation_vector)

        # SciPy 返回 (x, y, z, w) 顺序的四元数数组
        noisy_quaternion_array = noisy_rotation.as_quat()

        # 5. 提取并返回 Python 原生 float 形式的 (w, x, y, z)
        qx_noisy = float(noisy_quaternion_array[0])
        qy_noisy = float(noisy_quaternion_array[1])
        qz_noisy = float(noisy_quaternion_array[2])
        qw_noisy = float(noisy_quaternion_array[3])

        # 返回顺序与输入保持一致 (w, x, y, z)
        return qw_noisy, qx_noisy, qy_noisy, qz_noisy
