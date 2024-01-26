import torch
import torch.nn as nn
import torch.nn.functional as F

from .action import DiscreteActionDistributions
from .actor_critic import ActorCritic, DiscreteActor, Critic

class CNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 2, 1)
        self.lin1  = nn.Linear(512, 256)
        self.lin2  = nn.Linear(256, 256)
        self.flatten = nn.Flatten()

    def forward(self, inputs):
        x = inputs.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(F.relu(self.conv4(x)))
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))

        return x

class LinearLayerDiscreteActor(DiscreteActor):
    def __init__(self, actions_num_buckets, in_channels):
        total_action_dim = sum(actions_num_buckets)
        impl = nn.Linear(in_channels, total_action_dim)

        super().__init__(actions_num_buckets, impl)

        nn.init.orthogonal_(self.impl.weight, gain=0.01)
        nn.init.constant_(self.impl.bias, 0)

class LinearLayerCritic(Critic):
    def __init__(self, in_channels):
        super().__init__(nn.Linear(in_channels, 1))

        nn.init.orthogonal_(self.impl.weight)
        nn.init.constant_(self.impl.bias, 0)

class DenseLayerDiscreteActor(DiscreteActor):
    def __init__(self, actions_num_buckets, dtype):
        total_action_dim = sum(actions_num_buckets)
        impl = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, total_action_dim),
        )

        super().__init__(actions_num_buckets, impl)

        nn.init.orthogonal_(self.impl[0].weight, gain=0.01)
        nn.init.constant_(self.impl[0].bias, 0)
        nn.init.orthogonal_(self.impl[3].weight, gain=0.01)
        nn.init.constant_(self.impl[3].bias, 0)

class DenseLayerCritic(Critic):
    def __init__(self, dtype):
        super().__init__(nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ))

        nn.init.orthogonal_(self.impl[0].weight)
        nn.init.constant_(self.impl[0].bias, 0)
        nn.init.orthogonal_(self.impl[3].weight)
        nn.init.constant_(self.impl[3].bias, 0)

