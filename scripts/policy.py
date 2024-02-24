from madrona_escape_room_learn import (
    ActorCritic, DiscreteActor, Critic, 
    BackboneShared, BackboneSeparate,
    BackboneEncoder,
    RecurrentBackboneEncoder,
)

from madrona_escape_room_learn.models import (
    CNN, LinearLayerDiscreteActor, LinearLayerCritic,
    DenseLayerDiscreteActor, DenseLayerCritic, MLP
)

from madrona_escape_room_learn.rnn import LSTM

import math
import torch

import time

import cv2

import torch.nn as nn
import torch.nn.functional as F

def setup_obs(sim, raw_pixels=False):
    self_obs_tensor = sim.self_observation_tensor().to_torch()
    partner_obs_tensor = sim.partner_observations_tensor().to_torch()
    room_ent_obs_tensor = sim.room_entity_observations_tensor().to_torch()
    door_obs_tensor = sim.door_observation_tensor().to_torch()
    lidar_tensor = sim.lidar_tensor().to_torch()
    steps_remaining_tensor = sim.steps_remaining_tensor().to_torch()
    room_ent_vis_tensor = sim.room_entity_visibilities_tensor().to_torch()

    if raw_pixels:
        rgb_tensor = sim.rgb_tensor().to_torch()   
        depth_tensor = sim.depth_tensor().to_torch()
    
    N, A = self_obs_tensor.shape[0:2]
    batch_size = N * A

    # Add in an agent ID tensor
    id_tensor = torch.arange(A).float()
    if A > 1:
        id_tensor = id_tensor / (A - 1)

    id_tensor = id_tensor.to(device=self_obs_tensor.device)
    id_tensor = id_tensor.view(1, 2).expand(N, 2).reshape(batch_size, 1)

    if not raw_pixels:
        obs_tensors = [
            self_obs_tensor.view(batch_size, *self_obs_tensor.shape[2:]),
            partner_obs_tensor.view(batch_size, *partner_obs_tensor.shape[2:]),
            room_ent_obs_tensor.view(batch_size, *room_ent_obs_tensor.shape[2:]),
            door_obs_tensor.view(batch_size, *door_obs_tensor.shape[2:]),
            lidar_tensor.view(batch_size, *lidar_tensor.shape[2:]),
            # steps_remaining_tensor.view(batch_size, *steps_remaining_tensor.shape[2:]),
            room_ent_vis_tensor.view(batch_size, *room_ent_vis_tensor.shape[2:]), 
            id_tensor,
        ]

        num_obs_features = 0
        for i in range(len(obs_tensors)):
            tensor = obs_tensors[i]
            shape = list(tensor.size())
            if i == 5:
                continue
            if i in [1, 3]:
                # door obs and partner obs have a redundant isVisible field
                shape[-1] -= 1
            num_obs_features += math.prod(shape[1:])
        return obs_tensors, num_obs_features
        
    else:
        rgb_tensor = rgb_tensor[:, :, :, :, 0:3]
        
        rgb_tensor = rgb_tensor.view(-1, *(rgb_tensor.shape[2:]))
        depth_tensor = depth_tensor.view(-1, *(depth_tensor.shape[2:]))
        # raw pixels

        # print rgb_tensor.shape   # 2048, 64, 64, 3
        # print depth_tensor.shape # 2048, 64, 64, 1
        # print id_tensor.shape    # 2048, 1

        global_pos_tensor = self_obs_tensor.view(batch_size, *self_obs_tensor.shape[2:])
        global_pos_tensor = global_pos_tensor[:, 2:5]
        
        obs_tensors = [
            rgb_tensor,
            depth_tensor,
            id_tensor,
            global_pos_tensor
        ]
        # obs_tensors = [
        #     rgb_tensor.view(batch_size, *rgb_tensor.shape[2:]),
        #     depth_tensor.view(batch_size, *depth_tensor.shape[2:]),
        # ]
        num_channels = rgb_tensor.shape[-1] + depth_tensor.shape[-1]
        
        return obs_tensors, num_channels

def process_obs(self_obs, partner_obs, room_ent_obs,
                door_obs, lidar, visbs, ids):
    assert(not torch.isnan(self_obs).any())
    assert(not torch.isinf(self_obs).any())

    assert(not torch.isnan(partner_obs).any())
    assert(not torch.isinf(partner_obs).any())

    assert(not torch.isnan(room_ent_obs).any())
    assert(not torch.isinf(room_ent_obs).any())

    assert(not torch.isnan(lidar).any())
    assert(not torch.isinf(lidar).any())

    # assert(not torch.isnan(steps_remaining).any())
    # assert(not torch.isinf(steps_remaining).any())


    partner_obs_view = partner_obs.view(partner_obs.shape[0], -1)
    partner_obs_view[..., -1] = 1
    partner_masked = partner_obs_view.masked_fill(~partner_obs_view[..., -1:].bool(), -1.1)
    
    door_obs_view = door_obs.view(door_obs.shape[0], -1)
    door_obs_view[..., -1] = 1
    door_obs_masked = door_obs_view.masked_fill(~door_obs_view[..., -1:].bool(), -1.1);
    
    # eject out the is_visibility fields - their purpose has been served
    partner_masked = partner_masked[..., :-1]
    door_obs_masked = door_obs_masked[..., :-1]
    
    # room_ent_obs_view = room_ent_obs.view(room_ent_obs.shape[0], -1)
    visbs = torch.ones_like(visbs)
    room_ent_obs_masked = \
        room_ent_obs.masked_fill(~visbs.bool(), -1.1) \
        .view(room_ent_obs.shape[0], -1)

    return torch.cat([
        self_obs.view(self_obs.shape[0], -1),
        # partner_obs.view(partner_obs.shape[0], -1),
        partner_masked,
        # room_ent_obs.view(room_ent_obs.shape[0], -1),
        room_ent_obs_masked,
        # door_obs.view(door_obs.shape[0], -1),
        door_obs_masked,
        lidar.view(lidar.shape[0], -1),
        # steps_remaining.float() / 200,
        ids,
    ], dim=1).half()

def process_pixels(rgb, depth, ids=None, global_pos=None):
    assert(not torch.isnan(rgb).any())
    assert(not torch.isinf(rgb).any())

    # convert depth nan or inf to 0
    depth[torch.isnan(depth)] = 0
    depth[torch.isinf(depth)] = 0

    assert(not torch.isnan(depth).any())
    assert(not torch.isinf(depth).any())

    # convert rgb to float
    rgb = rgb.to(torch.float32)
    # normal rgb values (3 channels in last dim of rgb tensor) are 0-255, so divide by 255
    
    rgb = rgb / 255
    depth = depth / 255
    # rgb = rgb.view(rgb.shape[0]//2, 2, rgb.shape[1], rgb.shape[2], rgb.shape[3])
    # depth = depth.view(depth.shape[0]//2, 2, depth.shape[1], depth.shape[2], depth.shape[3])

    # rgb = rgb.permute(0, 2, 3, 1, 4)
    # depth = depth.permute(0, 2, 3, 1, 4)

    # rgb = rgb.reshape(rgb.shape[0], rgb.shape[1], rgb.shape[2], -1)
    # depth = depth.reshape(depth.shape[0], depth.shape[1], depth.shape[2], -1)

    CNN_input = torch.cat([rgb, depth], dim=-1) # shape = B (N * A), W, H, C
    # CNN_input = torch.cat([CNN_input, CNN_input], dim=0)

    # NOTE: UNCOMMENT THIS IN ORDER TO SEE THE CONSTANT IMAGES BEING PASSED TO CNN
    # cv2.imwrite(f"pix_{time.time()}.png", rgb[0].cpu().numpy())
    return CNN_input.to(torch.float16)

def make_policy(dim_info, num_channels, separate_value, raw_pixels=False):
    if raw_pixels:
        # encoder = RecurrentBackboneEncoder(
        #     net = MLP(input_dim = num_channels * dim_info,
        #               num_channels = num_channels,
        #               num_layers = 1),
        #     rnn = LSTM(in_channels = num_channels,
        #                hidden_channels = num_channels,
        #                num_layers = 1),
        # )
        
        # encoder = RecurrentBackboneEncoder(
        #     net = CNN(in_channels = dim_info),
        #     rnn = LSTM(in_channels = num_channels,
        #                hidden_channels = num_channels,
        #                num_layers = 1)
        # )
        
        
        encoder = BackboneEncoder(
            net = CNN(in_channels = dim_info)
        )
        
        backbone = BackboneShared(
            process_obs = process_pixels,
            encoder = encoder,
        )

        return ActorCritic(
            backbone = backbone,
            actor = LinearLayerDiscreteActor(
                [4, 8, 5, 2],
                num_channels,
            ),
            critic = LinearLayerCritic(num_channels),
        )
    
    else:
        # encoder = BackboneEncoder(
        #     net = MLP(
        #         input_dim = dim_info,
        #         num_channels = num_channels,
        #         num_layers = 3,
        #     ),
        # )
        
        encoder = RecurrentBackboneEncoder(
            net = MLP(
                input_dim = dim_info,
                num_channels = num_channels,
                num_layers = 3,
            ),
            rnn = LSTM(in_channels = num_channels,
                       hidden_channels = num_channels,
                       num_layers = 1)
        )

        if separate_value:
            backbone = BackboneSeparate(
                process_obs = process_obs,
                actor_encoder = encoder,
                critic_encoder = RecurrentBackboneEncoder(
                    net = MLP(
                        input_dim = dim_info,
                        num_channels = num_channels,
                        num_layers = 2,
                    ),
                    rnn = LSTM(
                        in_channels = num_channels,
                        hidden_channels = num_channels,
                        num_layers = 1,
                    ),
                )
            )
        else:
            backbone = BackboneShared(
                process_obs = process_obs,
                encoder = encoder,
            )

        return ActorCritic(
            backbone = backbone,
            actor = LinearLayerDiscreteActor(
                [4, 8, 5, 2],
                num_channels,
            ),
            critic = LinearLayerCritic(num_channels),
        )
