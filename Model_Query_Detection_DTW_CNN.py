####################################################################
# This file contains the CNN architecture to be trained using 
# 'query_detection_dtw_cnn.py'

# Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
# Written by Dhananjay Ram <dhananjay.ram@idiap.ch>,

# This file is part of CNN_QbE_STD.

# CNN_QbE_STD is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.

# CNN_QbE_STD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with CNN_QbE_STD. If not, see <http://www.gnu.org/licenses/>.

#####################################################################

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


class ClassifierCNN(nn.Module):
	def __init__(self, depth, dropout=0.1):
		super(ClassifierCNN, self).__init__()
		self.conv1 = nn.Conv2d(1, depth, 3)
		self.conv2 = nn.Conv2d(depth, depth, 3)
		self.conv3 = nn.Conv2d(depth, depth, 3)
		self.conv4 = nn.Conv2d(depth, depth, 3)
		self.conv5 = nn.Conv2d(depth, depth, 3)
		self.conv6 = nn.Conv2d(depth, depth, 3)
		self.conv7 = nn.Conv2d(depth, depth, 3)
		self.conv8 = nn.Conv2d(depth, depth/2, 1)

		self.maxpool = nn.MaxPool2d(2, 2)
		self.dout_layer = nn.Dropout(dropout)
		self.dout_layer2d = nn.Dropout2d(dropout)
		self.length = (depth/2)*3*20
		self.fc1 = nn.Linear(self.length, 60)
		self.fc2 = nn.Linear(60, 2)
		self.sm = nn.LogSoftmax()

	def forward(self, x):
		x = self.maxpool(x)
		x = F.relu(self.dout_layer(self.conv1(x)))
		x = F.relu(self.maxpool(self.dout_layer(self.conv2(x))))
		x = F.relu(self.dout_layer(self.conv3(x)))
		x = F.relu(self.maxpool(self.dout_layer(self.conv4(x))))
		x = F.relu(self.dout_layer(self.conv5(x)))
		x = F.relu(self.maxpool(self.dout_layer(self.conv6(x))))
		x = F.relu(self.dout_layer(self.conv7(x)))
		x = F.relu(self.maxpool(self.dout_layer(self.conv8(x))))

		x = x.view(-1, self.length)
		x = F.relu(self.dout_layer(self.fc1(x)))
		x = self.fc2(x)
		x = self.sm(x)
		return x

