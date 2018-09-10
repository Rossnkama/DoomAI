import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as f 
import torch.optim as optim
from torch.autograd import Variable

import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

import experience_replay, image_preprocessing 


# The brain of the AI is a convolutional network
class CNN(nn.Module):

	# Number of actions come from the ppaquette action space
	def __init__(self, no_actions):
		super(CNN, self).__init__()
		# in_channels = 1 because images are black and white so we don't...
		# need 3 images corresponding RBG channels.
		# Going to use 32 image kernels so want 32 convolved images...
		# as output channels
		self.convolution1 = nn.Conv2d(
			in_channels=1, 
			out_channels=32, 
			kernel_size=5
		)
		self.convolution2 = nn.Conv2d(
			in_channels=32, 
			out_channels=32, 
			kernel_size=3
		)
		self.convolution3 = nn.Conv2d(
			in_channels=32, 
			out_channels=64, 
			kernel_size=2
		)

		self.full_conn1 = nn.Linear(in_features=number_neurons , out_features=40)
		self.full_conn2 = nn.Linear(in_features=40, out_features=no_actions)