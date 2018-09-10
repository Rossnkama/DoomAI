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
		self.convolution1
		self.convolution2
		self.convolution3
		self.full_conn1
		self.full_conn2