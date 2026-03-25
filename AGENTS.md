# AGENTS.md

## 项目概述

这个项目基于 HUGS / Human Gaussian Splatting 的思路，目标是从单目视频中联合重建：

- 静态背景场景
- 可随姿态变化的人体 3D 表示

项目最终学习到的是两套高斯表示：

- `SceneGS`：用于表示背景场景的 3D Gaussians
- `HUGS_TRIMLP`：用于表示人体的可动画 3D Gaussians

系统依赖 SMPL 人体模型作为人体先验，并结合 Gaussian Splatting 渲染器进行可微渲染与训练。  
整体目标可以理解为：

`输入单目视频 -> 重建背景 + 重建人体 -> 支持原视角重建与后续可动画渲染`


## 这个项目在做什么

从代码执行逻辑看，项目主要在做一套“人体和场景联合建模”的训练系统，具体包括：

1. 读取数据集中的图像、mask、相机参数、COLMAP 点云、SMPL 参数
2. 用 COLMAP 点云初始化背景高斯
3. 用 SMPL 模板和神经网络初始化人体高斯
4. 通过可微 Gaussian Splatting 渲染人体、背景以及两者合成结果
5. 用图像重建损失反向优化
6. 在训练过程中持续 densify / prune 高斯点
7. 保存优化后的相机参数、SMPL 参数、模型 checkpoint 和可视化结果


## 核心训练流程

项目当前入口主要是：

- `emdb_train.py`
- `neuman_train.py`

其中最重要的流程是“两阶段训练”：

### 第一阶段：参数优化

由 `GaussianOptimTrainer` 执行，目标是优化：

- 相机外参
- SMPL 姿态
- SMPL 平移
- SMPL 尺度
- 人体与场景高斯表示的初始适配

这一步结束后会导出：

- `cam_optimized_by_model.npz`
- `smpl_optimized_by_model.npz`

这些文件会在第二阶段继续使用。

### 第二阶段：正式训练

由 `GaussianTrainer` 执行，使用第一阶段优化后的相机和 SMPL 参数继续训练最终模型。

这一阶段会联合优化：

- 背景场景高斯
- 人体高斯
- 人体动态外观和几何偏移

训练前期更偏向分开学习人体和背景，后期再更多使用人体与场景合成渲染结果进行联合优化。


## 数据集与输入

当前代码支持的数据集主要有：

- `NeuMan`
- `EMDB`
- `EMDB_refine`

数据读取阶段会整合以下信息：

- RGB 图像
- 人体前景 mask
- COLMAP 稀疏点云
- 相机内参和外参
- 每帧 SMPL 参数

对 `EMDB` 来说，项目会读取形如以下内容：

- `data/emdb/...` 或 `data/emdb_refine/...`
- 图像目录
- mask 目录
- COLMAP 稀疏重建结果
- 每帧对应的 SMPL 参数文件或 pickle / npz

如果存在第一阶段导出的优化结果，则 `refine` 版本数据集会优先使用这些优化后的相机和人体参数。


## 主要模块说明

### 1. 背景模型：`SceneGS`

文件：`jointgs/models/scene.py`

职责：

- 从 COLMAP 点云初始化背景高斯
- 学习每个高斯的位置、颜色、尺度、旋转、透明度
- 支持 densify 和 prune
- 用于渲染静态背景场景

它基本属于标准 3D Gaussian Splatting 风格的场景建模。


### 2. 人体模型：`HUGS_TRIMLP`

文件：`jointgs/models/hugs_trimlp.py`

职责：

- 使用 SMPL 模板作为人体结构先验
- 在 canonical 空间中维护人体高斯 anchor
- 借助 tri-plane 特征和多个 decoder 预测：
  - 高斯位置偏移
  - 高斯旋转
  - 高斯尺度
  - 颜色 / SH 特征
  - 透明度
- 使用 SMPL / LBS 将 canonical 人体变形到当前帧姿态
- 支持基于时间的动态几何和动态外观变化

这是项目中最核心的人体表示模块。


### 3. Tri-plane 与解码器

相关文件：

- `jointgs/models/modules/triplane.py`
- `jointgs/models/modules/decoders.py`

职责：

- `TriPlane`：在三张特征平面上查询 3D 点特征
- `GeometryDecoder`：预测几何偏移、旋转、尺度
- `AppearanceDecoder`：预测颜色 / SH 特征与透明度
- `DynamicGeometryDecoder`：预测随时间变化的几何偏移
- `DynamicAppearanceEncoder`：预测随时间变化的外观变化

这部分让人体表示不仅跟着 SMPL 动，还能学习视频中的动态细节。


### 4. 渲染器

文件：`jointgs/renderer/gs_renderer.py`

职责：

- 渲染纯人体
- 渲染纯场景
- 渲染人体与场景合成结果

底层使用 `diff_gaussian_rasterization` 进行 Gaussian Splatting 渲染。


### 5. Loss

文件：`jointgs/losses/loss.py`

训练损失主要包括：

- `L1`
- `SSIM`
- `LPIPS` patch 感知损失
- 人体初始化点与模板点的一致性约束

监督方式上，项目会分别对：

- 整体重建
- 背景区域
- 人体区域

进行约束。


### 6. 训练器

文件：

- `jointgs/trainer/gs_optimtrainer.py`
- `jointgs/trainer/gs_trainer.py`

职责：

- 组织训练循环
- 调用模型前向
- 调用渲染器
- 计算 loss
- 执行优化器更新
- 进行 densification / prune
- 保存日志、评估结果、checkpoint


## `emdb_train.py` 的真实作用

`emdb_train.py` 不是简单地“训练一个模型”，而是串起整个 EMDB 流程：

1. 读取 `cfg/emdb/human_scene.yaml`
2. 创建日志目录
3. 运行第一阶段参数优化
4. 运行第二阶段正式训练

所以它更像是一个 EMDB 数据上的完整训练入口脚本。


## 训练输出会产生什么

训练过程中通常会在 `output/...` 目录下生成：

- 日志文件
- 当前配置备份
- `ckpt/` 模型权重
- `train/` 和 `optim/` 可视化图像
- `meshes/` 导出的场景高斯 ply
- 优化后的相机参数
- 优化后的 SMPL 参数

这些结果既用于继续训练，也用于后处理、可视化和评估。


## 项目最重要的理解方式

如果只保留一句话来理解这个项目，可以记成：

> 这是一个基于 SMPL 先验和 Gaussian Splatting 的单目视频人体-场景联合重建系统，用两阶段优化同时学习背景场景和可动画人体表示。


## 给后续协作者的建议

后续阅读代码时，建议优先按这个顺序理解：

1. `emdb_train.py` / `neuman_train.py`
2. `jointgs/trainer/gs_optimtrainer.py`
3. `jointgs/trainer/gs_trainer.py`
4. `jointgs/datasets/*.py`
5. `jointgs/models/hugs_trimlp.py`
6. `jointgs/models/scene.py`
7. `jointgs/renderer/gs_renderer.py`
8. `jointgs/losses/loss.py`

这样最容易把“数据如何进入系统”以及“模型如何被训练起来”串联起来。

## 潜在创新方向

- 双层高斯人体表示：将人体拆分为强附着的 `body-attached gaussians` 与弱附着或自由运动的 `cloth/detached gaussians`，以更好建模宽松服装和非刚性附件。这个方向的核心价值是把“贴身身体”和“松散服饰”从表示层面明确区分开，减少单层人体表示对衣摆、外套和头发等区域的欠拟合。
- 骨骼局部 canonical 表示：围绕 SMPL 骨骼为不同身体部位建立 `bone-local canonical spaces`，并在关节区域做 `soft fusion`，以减弱全局 canonical 表示的长程耦合。它特别适合解决大幅度 articulated motion 下的局部变形困难，让手臂、腿部和关节附近的建模更稳定。
- 场景人体感知联合建模：让人体分支与场景分支在训练中显式交换几何、遮挡和边界信息，而不是仅在渲染阶段简单拼接。这个方向强调 human branch 和 scene branch 的双向协同，使场景理解能够反向帮助人体边界、遮挡和姿态优化。
- 人景分离表示：将视频解释为可分解的 `human / scene / boundary` 组件，以提升前景纯净度、背景稳定性和结果可解释性。相比简单联合渲染，这种设计更容易控制前景泄漏到背景、背景污染人体等问题。
- 遮挡感知 compositional rendering：在 `human-scene` 合成时显式建模前后遮挡排序与边界竞争关系，以减少人景交界处的混色和漂移。这个方向适合把新视角下的遮挡稳定性和边界质量作为重点改进目标。
- 运动驱动 ownership 建模：利用 `articulated human motion` 与 `static scene consistency` 共同决定高斯归属，从动态证据而非仅靠 mask 完成人景解耦。它的优势在于更贴近单目视频本身的时空结构，能更自然地解释哪些区域属于人、哪些区域属于静态场景。
- 接触感知人景优化：把脚地接触、坐姿支撑和几何穿插约束引入优化过程，使人体姿态和场景几何相互校正。这个方向适合强调 geometry-aware reconstruction，而不只是视觉上把人和景同时拟合出来。
- 不确定性感知自校准：对相机和 SMPL 初始化建立 `uncertainty-aware refinement` 机制，在不可靠帧上自适应调整优化强度和监督权重。它能够把“初始化误差不可避免”这件事显式纳入方法设计，尤其适合单目视频中的噪声 pose 和相机估计。
- 语义感知 densification：根据身体部位、运动幅度和边界复杂度自适应分配高斯密度，把预算集中在高频细节区域。这个方向更偏工程增强，但很适合作为提升渲染细节、边界清晰度和效率的配套创新。
- 外观分解建模：将人体外观拆分为 `canonical identity appearance` 与 `dynamic clothing / lighting residual`，以稳定身份信息并捕获时变细节。它适合和动态人体表示结合，用来解释同一人物在不同帧中的服装褶皱、光照变化和局部外观扰动。
- 边界专门建模：为头发、衣摆、手臂边缘等过渡区域引入 `boundary gaussians` 或 `transition ownership`，以提升细结构和轮廓质量。这个方向特别适合解决联合渲染里最容易翻车的人景交界、细长结构和半透明边缘问题。
- 非 SMPL 依附区域建模：为宽松服装、头发和配饰建立 `topology-relaxed gaussians`，使其不再被严格限制在 SMPL 表面附近。它本质上是在挑战“所有人体相关高斯都应贴附在 SMPL 上”这一假设，适合做成对复杂服装和附件更友好的扩展表示。
