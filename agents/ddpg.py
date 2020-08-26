import argparse

import os, sys, random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
# from tensorboardX import SummaryWriter

'''
Implementation of Deep Deterministic Policy Gradients (DDPG) with pytorch 
riginal paper: https://arxiv.org/abs/1509.02971
Not the author's implementation !
'''



device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Actor(nn.Module):
    def __init__(self, max_action):
        super(Actor, self).__init__()

        # self.l1 = nn.Linear(state_dim, 400)
        # self.l2 = nn.Linear(400, 300)
        # self.l3 = nn.Linear(300, action_dim)
        self.l1 = nn.Linear(2, 16)
        self.l2 = nn.Linear(16, 16)
        self.l3 = nn.Linear(16, 16)
        self.l4 = nn.Linear(16, 2)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.max_action * torch.tanh(self.l4(x))
        return x


def test():
    actor = Actor(30.0)
    actor.load_state_dict(torch.load("actor.pth"))
    center = [480.0, 360.0]
    center = torch.Tensor(center)
    actions = actor(center)
    actions = actions.cpu().data.numpy()
    # print(type(actions))
    # print(actions.shape)
    print(-actions[0])
    print(actions[1])

# test()