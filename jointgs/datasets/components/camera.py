class PINHOLE_CAMERA:
    def __init__(self, camera_id, width, height, fx, fy, cx, cy, resolution_scale=1.0):
        self.camera_id = camera_id
        self.resolution_scale = resolution_scale

        # 1. 缩放图像尺寸（必须是整数）
        self.width = int(width * resolution_scale)
        self.height = int(height * resolution_scale)

        # 2. 缩放焦距
        self.fx = fx * resolution_scale
        self.fy = fy * resolution_scale

        # 3. 缩放主点坐标
        self.cx = cx * resolution_scale
        self.cy = cy * resolution_scale

    @property
    def intrinsic_matrix(self):
        """方便后续计算，返回 3x3 内参矩阵"""
        import numpy as np
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])