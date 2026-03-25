import torch
from torch.utils.data import DataLoader
import os

from jointgs.datasets.components.frame import FRAME
from jointgs.datasets.components.camera import PINHOLE_CAMERA
from jointgs.datasets.components.smpl_param import SMPL_PARAM
from scipy.spatial.transform import Rotation as R
import numpy as np
from PIL import Image
from plyfile import PlyData
from torch.utils.data import Subset



def readColmapCamera(dataset_path):
    path = os.path.join(dataset_path, 'sparse', 'cameras.txt')
    camera_list = []
    with open(path, 'r') as f:
        lines = f.readlines()
    num_cameras = int(lines[2].split(':')[1].strip())
    for i in range(num_cameras):
        camera_data_line = lines[3 + i]
        parts = camera_data_line.split()
        camera_id = int(parts[0])-1
        model = parts[1]
        print(model)
        if model == 'PINHOLE':
            width, height, fx, fy, cx, cy  = int(parts[2]), int(parts[3]), float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            camera = PINHOLE_CAMERA(camera_id, width, height, fx, fy, cx, cy)
        elif model == 'SIMPLE_RADIAL':
            width, height, f, cx, cy = int(parts[2]), int(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
            camera = PINHOLE_CAMERA(camera_id, width, height, f, f, cx, cy)
        else:
            raise ValueError(f"not support model type {model}")

        camera_list.append(camera)

    return camera_list

def readColmapImage(dataset_path):
    path = os.path.join(dataset_path, 'sparse', 'images.txt')
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
        image_path = path.replace('sparse', 'images').replace('images.txt', image_name)
        image = Image.open(image_path)

        frame = FRAME(image_id,  image_name, image, camera_id, cam_rot, cam_translation)

        frame_list.append(frame)
    frame_list.sort(key=lambda frame: frame.image_id)
    return frame_list

def readColmapPoints3D(dataset_path):
    ply_path = os.path.join(dataset_path, 'sparse', 'points3D.ply')
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
        txt_path = os.path.join(dataset_path, 'sparse', 'points3D.txt')
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



def readColmapDataset(dataset_path):
    cameras = readColmapCamera(dataset_path)
    frames = readColmapImage(dataset_path)
    points3D_data = readColmapPoints3D(dataset_path)
    masks = []
    smpl_params = []

    smpl_path = os.path.join(dataset_path, '4d_humans', 'smpl_optimized_aligned_scale.npz')
    smpl_npz = np.load(smpl_path)

    for index, frame in enumerate(frames):
        image_name = frame.image_name
        mask_path = os.path.join(dataset_path, '4d_humans', 'sam_segmentations', 'mask_' + image_name[1:])
        mask = Image.open(mask_path)
        mask = np.array(mask) / 255.0
        masks.append(mask)

        poses = smpl_npz['body_pose'][index]
        betas = smpl_npz['betas'][index]
        transl = smpl_npz['transl'][index]
        global_orient = smpl_npz['global_orient'][index]
        scale = smpl_npz['scale'][index]

        smpl_param = SMPL_PARAM(betas, poses, global_orient, transl, scale)
        smpl_params.append(smpl_param)

    return cameras, frames, points3D_data, masks, smpl_params



class NeumanDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):

        seq = cfg.dataset.seq
        if_refine = cfg.if_refine
        dataset_path = f"data/neuman/dataset/{seq}"

        cameras, frames, points3D, masks, smpl_params = readColmapDataset(dataset_path)

        if if_refine:
            smpl_path = os.path.join(dataset_path, '4d_humans', 'smpl_optimized_by_model.npz')

            smpl_npz = np.load(smpl_path)
            smpl_params = []

            cam_path = os.path.join(dataset_path, '4d_humans', 'cam_optimized_by_model.npz')
            cam_npz = dict(np.load(cam_path))

            for index, frame in enumerate(frames):

                poses = smpl_npz['body_pose'][index]
                betas = smpl_npz['betas'][index]
                transl = smpl_npz['transl'][index]
                global_orient = smpl_npz['global_orient'][index]
                scale = smpl_npz['scale'][index]

                smpl_param = SMPL_PARAM(betas, poses, global_orient, transl, scale)
                smpl_params.append(smpl_param)

                cam_rot = cam_npz['cam_rot'][index]
                cam_transl = cam_npz['cam_transl'][index]


                frames[index].cam_rot = cam_rot
                frames[index].cam_transl = cam_transl
        else:
            mean = 0.0
            std_dev = cfg.noise_power
            for index, frame in enumerate(frames):
                print('add_noise_power:', std_dev)
                smpl_params[index].global_orient += np.random.normal(loc=mean, scale=std_dev, size=smpl_params[index].global_orient.shape)
                smpl_params[index].pose += np.random.normal(loc=mean, scale=std_dev, size=smpl_params[index].pose.shape)
                frames[index].cam_transl += np.random.normal(loc=mean, scale=std_dev, size=frames[index].cam_transl.shape)

                r_original = R.from_matrix(frames[index].cam_rot)
                axis_angle_vec = r_original.as_rotvec()
                axis_angle_vec += np.random.normal(loc=mean, scale=std_dev, size=axis_angle_vec.shape)
                r_noisy = R.from_rotvec(axis_angle_vec)
                noisy_cam_rot = r_noisy.as_matrix()
                frames[index].cam_rot = noisy_cam_rot





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