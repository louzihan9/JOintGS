import numpy as np
from PIL import Image


class FRAME:
    def __init__(self, image_id, image_name, image, camera_id, cam_rot, cam_transl, resolution_scale=1.0):
        self.image_id = image_id
        self.image_name = image_name
        self.resolution_scale = resolution_scale

        # 1. 图像缩放处理
        if resolution_scale != 1.0:
            # 假设 image 是 PIL.Image 对象
            # 如果是 numpy 数组，请使用 cv2.resize(image, (new_w, new_h))
            orig_w, orig_h = image.size
            new_w = int(orig_w * resolution_scale)
            new_h = int(orig_h * resolution_scale)
            # 使用 LANCZOS 滤镜保证下采样质量
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 2. 转换为张量格式 (C, H, W) 并归一化
        # 现在这里的 image 已经是缩放后的了
        self.image = np.array(image).transpose(2, 0, 1) / 255.0

        # 3. 记录缩放后的尺寸
        self.image_height, self.image_width = self.image.shape[1], self.image.shape[2]

        # 4. 相机外参 (通常不需要缩放)
        self.camera_id = camera_id
        self.cam_rot = cam_rot
        self.cam_transl = cam_transl