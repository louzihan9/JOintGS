import torch
from torch.utils.data import DataLoader
import os

from jointgs.datasets.components.frame import FRAME
from jointgs.datasets.components.camera import PINHOLE_CAMERA
from jointgs.datasets.components.smpl_param import SMPL_PARAM
from scipy.spatial.transform import Rotation as R
import numpy as np
from PIL import Image
from loguru import logger
from plyfile import PlyData
from torch.utils.data import Subset
import pickle
from jointgs.utils.rotations import axis_angle_to_matrix, matrix_to_axis_angle


def readColmapCamera(dataset_path, resolution_scale=1.0):
    path = os.path.join(dataset_path, 'sparse', '0', 'cameras.txt')
    camera_list = []
    with open(path, 'r') as f:
        lines = f.readlines()
    num_cameras = int(lines[2].split(':')[1].strip())
    for i in range(num_cameras):
        camera_data_line = lines[3 + i]
        parts = camera_data_line.split()
        camera_id = int(parts[0])-1
        model = parts[1]
        if model == 'PINHOLE':
            width, height, fx, fy, cx, cy  = int(parts[2]), int(parts[3]), float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            camera = PINHOLE_CAMERA(camera_id, width, height, fx, fy, cx, cy, resolution_scale)
        elif model == 'SIMPLE_RADIAL':
            width, height, f, cx, cy = int(parts[2]), int(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
            camera = PINHOLE_CAMERA(camera_id, width, height, f, f, cx, cy, resolution_scale)
        else:
            raise ValueError(f"not support model type {model}")

        camera_list.append(camera)

    return camera_list

def readColmapImage(dataset_path, resolution_scale=1.0):
    path = os.path.join(dataset_path, 'sparse', '0', 'images_rescaled.txt')
    frame_list = []
    with open(path, 'r') as f:
        lines = f.readlines()
    num_frames = int(lines[3].split(':')[1].split(',')[0].strip())
    for i in range(num_frames):
        parts = lines[4+2*i].split()
        image_id = int(parts[9][0:5])
        q_w, q_x, q_y, q_z = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        t_x, t_y, t_z = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])-1
        image_name = parts[9]

        cam_rot = R.from_quat([q_x, q_y, q_z, q_w]).as_matrix()
        cam_translation = np.array([t_x, t_y, t_z])


        image_path = os.path.join(dataset_path, 'images', image_name)
        image = Image.open(image_path)

        frame = FRAME(image_id, image_name, image, camera_id, cam_rot, cam_translation, resolution_scale)

        frame_list.append(frame)
    frame_list.sort(key=lambda frame: frame.image_id)
    return frame_list

def readColmapPoints3D(dataset_path):
    ply_path = os.path.join(dataset_path, 'sparse', '0', 'points3D_rescaled.ply')
    if os.path.exists(ply_path):
        plydata = PlyData.read(ply_path)
        vertices = plydata['vertex'].data
        x = vertices['x']
        y = vertices['y']
        z = vertices['z']
        coordinates = np.vstack((x, y, z)).T

        red = vertices['red'] / 255.0
        green = vertices['green'] / 255.0
        blue = vertices['blue'] / 255.0
        colors = np.vstack((red, green, blue)).T

        points3D = {
            'points': np.array(coordinates),
            'colors': np.array(colors)
        }
    else:
        txt_path = os.path.join(dataset_path, 'sparse', '0', 'points3D_rescaled.txt')
        coordinates = []
        colors = []
        with open(txt_path, 'r') as f:
            line_count = 0
            for line in f:
                line_count += 1
                stripped_line = line.strip()

                if not stripped_line or stripped_line.startswith('#'):
                    continue
                parts = stripped_line.split()
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                coordinates.append([x, y, z])
                r = int(parts[4]) / 255.0
                g = int(parts[5]) / 255.0
                b = int(parts[6]) / 255.0
                colors.append([r, g, b])

        points3D = {
            'points': np.array(coordinates),
            'colors': np.array(colors)
        }
    return points3D



def readEMDBDataset(dataset_path, resolution_scale=1.0):
    cameras = readColmapCamera(dataset_path, resolution_scale)
    frames = readColmapImage(dataset_path, resolution_scale)
    points3D_data = readColmapPoints3D(dataset_path)
    masks = []
    smpl_params = []


    for index, frame in enumerate(frames):

        image_id = frame.image_id

        mask_path = os.path.join(dataset_path, 'masks', f'{image_id:05d}.png')
        if not os.path.exists(mask_path):
            mask_path = os.path.join(dataset_path, 'mask', f'{image_id:05d}.png')

        mask = Image.open(mask_path)
        if resolution_scale != 1.0:
            new_w = int(mask.width * resolution_scale)
            new_h = int(mask.height * resolution_scale)
            mask = mask.resize((new_w, new_h), Image.NEAREST)

        mask_array = np.array(mask)

        if len(mask_array.shape) == 3:
            mask_array = mask_array[:, :, 0]

        mask_array = mask_array / 255.0
        masks.append(mask_array)

        smpl_path = os.path.join(dataset_path, 'sam3d', 'smpl' ,f'smpl_rescaled_{image_id:05d}.npz')
        smpl_npz = dict(np.load(smpl_path), allow_pickle=True)
        poses = smpl_npz['body_pose']
        betas = smpl_npz['betas']
        transl = smpl_npz['transl']
        global_orient = smpl_npz['global_orient']
        scale = smpl_npz['scale'].squeeze()


        smpl_param = SMPL_PARAM(betas, poses, global_orient, transl, scale)
        smpl_params.append(smpl_param)

    return cameras, frames, points3D_data, masks, smpl_params


class EMDBrefineDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):

        seq = cfg.dataset.seq
        dataset_path = os.path.join(f"data/emdb_refine", f'{seq}')
        resolution_scale = cfg.get("resolution_scale", 1.0)
        cameras, frames, points3D, masks, smpl_params = readEMDBDataset(dataset_path, resolution_scale)

        if_refine = cfg.if_refine
        if if_refine:
            smpl_path = f'{cfg.logdir}/smpl_optimized_by_model.npz'
            if os.path.exists(smpl_path):
                logger.info(f'use refine smpl')
                smpl_npz = np.load(smpl_path)
                smpl_params = []
                for index, frame in enumerate(frames):

                    poses = smpl_npz['body_pose'][index]
                    betas = smpl_npz['betas'][index]
                    transl = smpl_npz['transl'][index]
                    global_orient = smpl_npz['global_orient'][index]
                    scale = smpl_npz['scale'][index]

                    smpl_param = SMPL_PARAM(betas, poses, global_orient, transl, scale)
                    smpl_params.append(smpl_param)
            
            cam_path = f'{cfg.logdir}/cam_optimized_by_model.npz'
            if os.path.exists(cam_path):
                logger.info(f'use refine cams')
                cam_npz = np.load(cam_path)
                cam_rot = cam_npz['cam_rot'][index]
                cam_transl = cam_npz['cam_transl'][index]

                frames[index].cam_rot = cam_rot
                frames[index].cam_transl = cam_transl


        self.num_cameras = len(cameras)
        self.cameras = cameras
        self.num_frames = len(frames)
        self.frames = frames
        self.points3D = points3D
        self.masks = masks
        self.smpl_params = smpl_params

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        if idx >= self.num_frames:
            raise IndexError("索引超出范围")

        return {
            "frame": self.frames[idx],
            "camera": self.cameras[self.frames[idx].camera_id],
            "mask": self.masks[idx],
            "smpl_param": self.smpl_params[idx]
        }

    def split_train_val(self, train_ratio=0.8, seed=None):
        all_indices = list(range(self.num_frames))
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(all_indices)
        train_size = int(self.num_frames * train_ratio)
        train_indices = all_indices[:train_size]
        val_indices = all_indices[train_size:]

        train_subset = Subset(self, train_indices)
        val_subset = Subset(self, val_indices)

        return train_subset, val_subset










