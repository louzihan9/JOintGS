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
import pickle
from jointgs.utils.rotations import axis_angle_to_matrix, matrix_to_axis_angle



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



class EmdbDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        image_scale = cfg.image_scale

        person = cfg.dataset.person
        seq = cfg.dataset.seq

        dataset_path = os.path.join(f"data/emdb", f'{person}_{seq}')
        colmap_folder_path = os.path.join(dataset_path, f'{person}_{seq}_colmap')

        pkl_path = os.path.join(dataset_path, f"{person}_{seq}_data.pkl")
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)

        image_folder_path = os.path.join(colmap_folder_path, "images")
        mask_folder_path = os.path.join(colmap_folder_path, "masks")

        cameras = []
        frames = []
        masks = []
        smpl_params = []

        for filename in os.listdir(image_folder_path):
            if filename.lower().endswith(".png") or filename.lower().endswith(".jpg"):
                image_path = os.path.join(image_folder_path, filename)
                with Image.open(image_path) as img:
                    width, height = img.size
                    break


        intrinsics = pkl_data['camera']['intrinsics']
        fx, fy, cx, cy =intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]

        width = int(width * image_scale)
        height = int(height * image_scale)
        fx = fx * image_scale
        fy = fy * image_scale
        cx = cx * image_scale
        cy = cy * image_scale

        camera = PINHOLE_CAMERA(0, width, height, fx, fy, cx, cy)
        cameras.append(camera)

        points3D_data = readColmapPoints3D(colmap_folder_path)
        for image_name in sorted(os.listdir(image_folder_path)):
            image_id = int(image_name[:-4])

            image = Image.open(os.path.join(image_folder_path, image_name))
            new_size = (int(image.width * image_scale), int(image.height * image_scale))
            image = image.resize(new_size, Image.LANCZOS)

            cam_rot = pkl_data['camera']['extrinsics'][image_id][:3, :3]
            cam_translation = pkl_data['camera']['extrinsics'][image_id][:3, 3]
            frame = FRAME(image_id, image_name, image, 0, cam_rot, cam_translation)
            frames.append(frame)

            mask_name = image_name.replace('jpg', 'png')
            mask = Image.open(os.path.join(mask_folder_path, mask_name))
            mask = mask.resize(new_size, Image.NEAREST)

            mask = np.array(mask) / 255.0
            masks.append(mask)


            global_orient = pkl_data['smpl']['poses_root'][image_id]
            poses = pkl_data['smpl']['poses_body'][image_id]
            transl = pkl_data['smpl']['trans'][image_id]
            betas = pkl_data['smpl']['betas']

            smpl_param = SMPL_PARAM(betas, poses, global_orient, transl)
            smpl_params.append(smpl_param)

        if_refine = cfg.if_refine
        if if_refine:
            smpl_path = os.path.join(dataset_path, 'smpl_optimized_by_model.npz')
            if os.path.exists(smpl_path):
                smpl_npz = np.load(smpl_path)
                smpl_params = []

                cam_path = os.path.join(dataset_path, 'cam_optimized_by_model.npz')
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



        self.num_cameras = len(cameras)
        self.cameras = cameras
        self.num_frames = len(frames)
        self.frames = frames
        self.points3D = points3D_data
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




