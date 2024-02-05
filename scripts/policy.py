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
            steps_remaining_tensor.view(batch_size, *steps_remaining_tensor.shape[2:]),
            id_tensor,
        ]

        num_obs_features = 0
        for tensor in obs_tensors:
            num_obs_features += math.prod(tensor.shape[1:])

        return obs_tensors, num_obs_features
        
    else:
        rgb_tensor = rgb_tensor[:, :, :, :, 0:3]

        # permuted_rgb = rgb_tensor.permute(0, 2, 3, 4, 1)
        # permuted_depth = depth_tensor.permute(0, 2, 3, 4, 1)
        agent_1_rgb = rgb_tensor[:, 0]
        agent_2_rgb = rgb_tensor[:, 1]
        # reshaped_rgb = torch.cat((agent_1_rgb, agent_2_rgb), dim=-1)
        agent_1_depth = depth_tensor[:, 0]
        agent_2_depth = depth_tensor[:, 1]
        # reshaped_depth = torch.cat((agent_1_depth, agent_2_depth), dim=-1) 
        
        # rgb_tensor = torch.cat((agent_1_rgb, agent_2_rgb), dim=0)
        # depth_tensor = torch.cat((agent_1_depth, agent_2_depth), dim=0)
        
        rgb_tensor = rgb_tensor.view(-1, *(rgb_tensor.shape[2:]))
        depth_tensor = depth_tensor.view(-1, *(depth_tensor.shape[2:]))
        # raw pixels
  
        obs_tensors = [
            rgb_tensor,
            depth_tensor,
        ]
        # obs_tensors = [
        #     rgb_tensor.view(batch_size, *rgb_tensor.shape[2:]),
        #     depth_tensor.view(batch_size, *depth_tensor.shape[2:]),
        # ]
        num_channels = rgb_tensor.shape[-1] + depth_tensor.shape[-1]
        
        return obs_tensors, num_channels

def process_obs(self_obs, partner_obs, room_ent_obs,
                door_obs, lidar, steps_remaining, ids):
    assert(not torch.isnan(self_obs).any())
    assert(not torch.isinf(self_obs).any())

    assert(not torch.isnan(partner_obs).any())
    assert(not torch.isinf(partner_obs).any())

    assert(not torch.isnan(room_ent_obs).any())
    assert(not torch.isinf(room_ent_obs).any())

    assert(not torch.isnan(lidar).any())
    assert(not torch.isinf(lidar).any())

    assert(not torch.isnan(steps_remaining).any())
    assert(not torch.isinf(steps_remaining).any())

    return torch.cat([
        self_obs.view(self_obs.shape[0], -1),
        partner_obs.view(partner_obs.shape[0], -1),
        room_ent_obs.view(room_ent_obs.shape[0], -1),
        door_obs.view(door_obs.shape[0], -1),
        lidar.view(lidar.shape[0], -1),
        steps_remaining.float() / 200,
        ids,
    ], dim=1).half()

def process_pixels(rgb, depth):
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

    CNN_input = torch.cat([rgb, depth], dim=-1) # shape = B (N * A), W, H, C
    # breakpoint()
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
        
        encoder = RecurrentBackboneEncoder(
            net = CNN(in_channels = dim_info),
            rnn = LSTM(in_channels = num_channels,
                       hidden_channels = num_channels,
                       num_layers = 1)
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
        encoder = BackboneEncoder(
            net = MLP(
                input_dim = dim_info,
                num_channels = num_channels,
                num_layers = 1,
            ),
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