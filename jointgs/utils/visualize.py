import open3d as o3d
import torch
import numpy as np
import math


def to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


import open3d as o3d
import torch
import numpy as np


def visualize_all(data, points3d_obj, model_path="data/smpl/SMPL_NEUTRAL.pkl"):
    from smplx import SMPL
    geometries = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. 场景点云 ---
    pts_dict = points3d_obj.item() if isinstance(points3d_obj, np.ndarray) else points3d_obj
    xyz = to_numpy(pts_dict['points'])
    rgb = to_numpy(pts_dict['colors'])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    geometries.append(pcd)

    # --- 2. 核心修改：同步渲染器的矩阵转置 ---
    # 这里加上 .T，确保可视化看到的相机参数与传给 gsplat 渲染器的一致
    w2c = to_numpy(data["camera_transforms"]["world_to_camera"]).T

    # 重新计算 C2W
    c2w = np.linalg.inv(w2c)
    cam_center = c2w[:3, 3]
    cam_rotation = c2w[:3, :3]

    W, H = data["camera_intrinsic"]["width"], data["camera_intrinsic"]["height"]
    # 理想内参 (如果你的渲染结果不对，可能这里也需要从 data 中提取真实的 cx, cy)
    intrinsic = np.array([[W, 0, W / 2], [0, W, H / 2], [0, 0, 1]])

    # A. 红色视锥体
    frustum = o3d.geometry.LineSet.create_camera_visualization(
        view_width_px=int(W), view_height_px=int(H),
        intrinsic=intrinsic, extrinsic=w2c, scale=0.2
    )
    frustum.paint_uniform_color([1, 0, 0])
    geometries.append(frustum)

    # B. 相机局部坐标轴
    cam_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    cam_axes.rotate(cam_rotation, center=(0, 0, 0))
    cam_axes.translate(cam_center)
    geometries.append(cam_axes)


    smpl_model = SMPL(model_path, gender='neutral').to(device)
    params = data["smpl_param"]


    output = smpl_model(
        betas=params["beta"].to(device).unsqueeze(0),
        body_pose=params["pose"].to(device).unsqueeze(0),
        global_orient=params["global_orient"].to(device).unsqueeze(0),
        transl=params["transl"].to(device).unsqueeze(0),
        return_verts=True
    )
    verts = to_numpy(output.vertices.squeeze(0)) * to_numpy(params["scale"])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(to_numpy(smpl_model.faces))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.2, 0.7, 0.9])
    geometries.append(mesh)

    # --- 4. 相机视角模拟参数 ---
    view_dir = c2w[:3, 2]  # Z 轴
    up_dir = c2w[:3, 1]  # Y 轴
    target_point = cam_center + view_dir
    front_vector = -view_dir
    up_vector = -up_dir

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Camera POV (With Transpose Sync)",
        front=front_vector.astype(np.float64),
        lookat=target_point.astype(np.float64),
        up=up_vector.astype(np.float64),
        zoom=0.01
    )


def visualize_camera_and_points(means3D, w2c_row_major):
    """
    针对行优先 (Row-major) w2c 矩阵的可视化
    """
    # 1. 转换为标准列优先格式 (Standard Column-major)
    # 这样 w2c_standard[:3, 3] 就是平移，且符合 Open3D 预期
    w2c_standard = w2c_row_major.T

    # 2. 点云处理
    if hasattr(means3D, "detach"):
        means3D = means3D.detach().cpu().numpy()
    means3D = np.ascontiguousarray(means3D.reshape(-1, 3).astype(np.float64))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means3D)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])

    # 3. 计算相机在世界系的位置 (c2w 的最后一列)
    c2w = np.linalg.inv(w2c_standard)
    cam_pos = c2w[:3, 3]

    # 4. 可视化
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    cam_frame.transform(c2w)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="World View (Row-major Fixed)", width=1280, height=720)
    vis.add_geometry(pcd)
    vis.add_geometry(cam_frame)

    view_control = vis.get_view_control()
    cam_params = view_control.convert_to_pinhole_camera_parameters()

    # Open3D 必须接收列优先矩阵
    cam_params.extrinsic = w2c_standard

    view_control.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

    print(f"Detected Row-major Matrix. Camera World Position: {cam_pos}")
    vis.run()
    vis.destroy_window()


def visualize_3dpoints(deformed_xyz):
    points_np = deformed_xyz.detach().cpu().numpy()

    # 2. 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    # 3. (可选) 给点云上色，方便在黑色背景下观察
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色

    # 4. 创建一个坐标轴 (Size=0.5) 帮助观察原点位置
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    # 5. 弹出窗口可视化
    print("Visualizing deformed_xyz... Close the window to continue.")
    o3d.visualization.draw_geometries([pcd, coord_frame],
                                      window_name="Deformed XYZ Visualization",
                                      width=1024, height=768)