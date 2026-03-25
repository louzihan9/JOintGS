#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import sys
import torch
import random
import os
import open3d as o3d

def optimize_init(cfg, model, num_steps: int = 5000):
    cfg_stage = cfg.train
    model.train()

    model.setup_optimizer(cfg_stage)
    optim = model.optimizer
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1000, factor=0.5)
    fn = torch.nn.MSELoss()
    
    body_pose = torch.zeros((69)).to("cuda").float()
    global_orient = torch.zeros((3)).to("cuda").float()
    betas = torch.zeros((10)).to("cuda").float()
    
    gt_vals = model.initialize()
    
    print("===== Ground truth values: =====")
    for k, v in gt_vals.items():
        print(k, v.shape)
        gt_vals[k] = v.detach().clone().to("cuda").float()
    print("================================")


    # triplane_model_path = f'{cfg.logdir}/ckpt/triplane_model.pth'
    triplane_model_path = f'data/triplane_model/triplane_model.pth'

    if os.path.exists(triplane_model_path):
        loaded_state = torch.load(triplane_model_path)
        model.triplane.load_state_dict(loaded_state['triplane'])
        model.appearance_dec.load_state_dict(loaded_state['appearance_dec'])
        model.dynamic_appearance_dec.load_state_dict(loaded_state['dynamic_appearance_dec'])
        model.geometry_dec.load_state_dict(loaded_state['geometry_dec'])
        model.dynamic_geometry_dec.load_state_dict(loaded_state['dynamic_geometry_dec'])
        model.deformation_dec.load_state_dict(loaded_state['deformation_dec'])
    else:
        losses = []
        for i in range(num_steps):
            if i % 100 ==0:
                print(i)

            model_out = model.forward(global_orient, body_pose, betas, dataset_idx=random.randint(0, 500))
            loss_dict = {}
            for k, v in gt_vals.items():
                if k in ['faces', 'deformed_normals', 'edges']:
                    continue
                if k in model_out:
                    if model_out[k] is not None:
                        loss_dict['loss_' + k] = fn(model_out[k], v)

            loss = sum(loss_dict.values())
            loss.backward()
            loss_str = ", ".join([f"{k}: {v.item():.7f}" for k, v in loss_dict.items()])
            print(f"Step {i:04d}: {loss.item():.7f} ({loss_str})", end='\r')

            optim.step()
            optim.zero_grad(set_to_none=True)
            lr_scheduler.step(loss.item())

            losses.append(loss.item())


        triplane_state = {
            'triplane': model.triplane.state_dict(),
            'appearance_dec': model.appearance_dec.state_dict(),
            'dynamic_appearance_dec': model.dynamic_appearance_dec.state_dict(),
            'geometry_dec': model.geometry_dec.state_dict(),
            'dynamic_geometry_dec': model.dynamic_geometry_dec.state_dict(),
            'deformation_dec': model.deformation_dec.state_dict()
        }
        torch.save(triplane_state, triplane_model_path)
    return model
