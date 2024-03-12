import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .action import DiscreteActionDistributions
from .actor_critic import ActorCritic, DiscreteActor, Critic

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResNet9(nn.Module):
    def __init__(self, in_channels):
        super(ResNet9, self).__init__()
        
        # BLOCK-1 (starting block)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3))
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))
        
        # BLOCK-2 (1)
        self.conv2_1_1 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.batchnorm2_1_1 = nn.BatchNorm2d(64)
        self.conv2_1_2 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm2_1_2 = nn.BatchNorm2d(64)
        self.se2_1 = SEBlock(64)
        
        # BLOCK-3 (1)
        self.conv3_1_1 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.batchnorm3_1_1 = nn.BatchNorm2d(128)
        self.conv3_1_2 = nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm3_1_2 = nn.BatchNorm2d(128)
        self.se3_1 = SEBlock(128)
        
        # BLOCK-4 (1)
        self.conv4_1_1 = nn.Conv2d(128, 128, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.batchnorm4_1_1 = nn.BatchNorm2d(128)
        self.conv4_1_2 = nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm4_1_2 = nn.BatchNorm2d(128)
        self.se4_1 = SEBlock(128)
        
        # Final layers to match the original CNN class output dimensions
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(512, 512)
        self.lay1 = nn.LayerNorm(512)
        self.lin2 = nn.Linear(512, 256)
        self.lay2 = nn.LayerNorm(256)

    def forward(self, inputs):
        x = inputs.permute(0, 3, 1, 2)
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = self.maxpool1(x)
        
        x = F.relu(self.batchnorm2_1_1(self.conv2_1_1(x)))
        x = F.relu(self.batchnorm2_1_2(self.conv2_1_2(x)))
        x = self.se2_1(x)
        
        x = F.relu(self.batchnorm3_1_1(self.conv3_1_1(x)))
        x = F.relu(self.batchnorm3_1_2(self.conv3_1_2(x)))
        x = self.se3_1(x)
        
        x = F.relu(self.batchnorm4_1_1(self.conv4_1_1(x)))
        x = F.relu(self.batchnorm4_1_2(self.conv4_1_2(x)))
        x = self.se4_1(x)
        
        x = self.flatten(x)
        x = F.relu(self.lin1(x))
        x = self.lay1(x)
        x = F.relu(self.lin2(x))
        x = self.lay2(x)
        
        return x

class CNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 2, 1)
        self.lin1  = nn.Linear(2048, 512)
        self.lay1  = nn.LayerNorm(512)
        
        self.lin2  = nn.Linear(512  + 0 + 0, 256)
        
        self.lay2  = nn.LayerNorm(256)
        self.flatten = nn.Flatten()

    def forward(self, inputs):
        x = inputs.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(F.relu(self.conv4(x)))
        x = F.relu(self.lin1(x))
        x = self.lay1(x)

        x = F.relu(self.lin2(x))
        x = self.lay2(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, num_channels, num_layers):
        super().__init__()

        layers = [
            nn.Linear(input_dim, num_channels),
            nn.LayerNorm(num_channels),
            nn.ReLU(),
        ]
        for i in range(num_layers - 1):
            layers.append(nn.Linear(num_channels, num_channels))
            layers.append(nn.LayerNorm(num_channels))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, inputs):
        return self.net(inputs)


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

