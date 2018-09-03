####################################################################
# This is the main training file to train CNN models for QbE-STD 
# The architecture is in 'Model_Query_Detection_DTW_CNN.py' and 
# the training data is prepared using 'Dataset_DTW_CNN.py'

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



import math
import random
import torch
import numpy
from torch.autograd import Variable

class Dataset(object):

	def __init__(self, filelist, query, search, maxWidth, maxLength, batchSize, maxBatch, cuda, shuffle, volatile=False):
		self.filelist = filelist
		self.query = query
		self.search = search
		self.maxWidth = maxWidth
		self.maxLength = maxLength
		self.batchSize = batchSize
		self.maxBatch = maxBatch
		self.cuda = cuda
		self.shuffle = shuffle
		self.volatile = volatile
		self.eps = 1e-5

	def _batchify(self, query, search, len_query, len_search, align_right=False):

		out = query[0].new(len(query), 1, self.maxWidth, self.maxLength).fill_(-1)
		for i in range(len(query)):

			dist = torch.log(torch.mm(query[i], search[i]) + self.eps)
			dist = -1 + 2* (dist - dist.min())/(dist.max() - dist.min())
			# length of the distance matrix
			length = len_search[i]
			# width of the distance matrix
			width = len_query[i]
			ind_length = torch.LongTensor(self.compression_index(length, self.maxLength))
			ind_width = torch.LongTensor(self.compression_index(width, self.maxWidth))
			dist = dist.index_select(0, ind_width)
			dist = dist.index_select(1, ind_length)
			out[i].narrow(1, (self.maxWidth - dist.size(0))/2, dist.size(0)).narrow(2, (self.maxLength - dist.size(1))/2, dist.size(1)).copy_(dist)
		return out

	def compression_index(self, length, max_length):
		# no of elements to be deleted
		n_del = length - max_length
		if n_del > 0:
			# index of the elements to be deleted
			ind_del = (length/n_del)*numpy.array(range(n_del))
			# index of the elements to choose for compression
			ind_keep = numpy.delete(numpy.array(range(length)), ind_del, axis=0)
		else:
			ind_keep = numpy.array(range(length))
		return ind_keep


	def makeBatches_dev(self):
		if self.shuffle:
			random.shuffle(self.filelist)
		distanceBatch, labelBatch = [], []
		count, batch = 0, 0
		data_query, data_search, data_label = [], [], []
		len_query, len_search = [], []
		for line in self.filelist:
			words = line.strip().split()
			data_query.append(self.query[words[0]])
			data_search.append(self.search[words[2]].transpose(0,1).contiguous())
			len_query.append(int(words[1]))
			len_search.append(int(words[3]))
			# obtain the corresponding label
			data_label.append(words[4])
			count += 1
			if count == self.batchSize:
				label = None
				label = map(int, data_label)
				label = torch.LongTensor(label)
				labelBatch.append(label)
				distanceBatch.append(self._batchify(data_query, data_search, len_query, len_search, align_right=False))
				data_query, data_search, data_label = [], [], []
				len_query, len_search = [], []
				count = 0
				batch = batch + 1
			if batch == self.maxBatch:
				break
		return zip(distanceBatch, labelBatch)

	def makeBatches(self):
		if self.shuffle:
			random.shuffle(self.filelist)
		distanceBatch, labelBatch = [], []
		count = 0
		batch = 0
		data_query, data_search, data_label = [], [], []
		len_query, len_search = [], []
		for line in self.filelist:
			words = line.strip().split()
			data_query.append(self.query[words[0]])
			data_search.append(self.search[words[2]].transpose(0,1).contiguous())
			len_query.append(int(words[1]))
			len_search.append(int(words[3]))
			# obtain the corresponding label
			data_label.append(words[4])
			count = count + 1
			if count == self.batchSize:
				label = None
				label = map(int, data_label)
				# if all the labels are 0, discard this batch
				if not any(label):
					data_query, data_search, data_label = [], [], []
					len_query, len_search = [], []
					count = 0
					continue
				label = torch.LongTensor(label)
				labelBatch.append(label)
				distanceBatch.append(self._batchify(data_query, data_search, len_query, len_search, align_right=False))
				data_query, data_search, data_label = [], [], []
				len_query, len_search = [], []
				count = 0
				batch = batch + 1
			if batch == self.maxBatch:
				break
		return zip(distanceBatch, labelBatch)

