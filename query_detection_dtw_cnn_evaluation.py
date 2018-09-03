####################################################################
# This is the evaluation file to test CNN models for QbE-STD 
# It generates a score file corresponding to each query. Each 
# score file contains a list of test utterances with corresponding 
# score.

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
import torch.nn.functional as F
import numpy
import random
import os
import argparse
import sys
import Model_Query_Detection_DTW_CNN as Model

parser = argparse.ArgumentParser(description='query_detection_dtw_cnn_evaluation.py')

## Model options

parser.add_argument('-depth', type=int, default=30,
					help='depth of features in each layer of the CNN')
parser.add_argument('-input_size', type=int, default=152,
					help='Length of the input feature vector')
parser.add_argument('-load_model', action='store_true',
					help='Load a previously trained model')
parser.add_argument('-modelpath', default="/path/to/model/",
					help="Path to saved model to be loaded for evaluation.")
parser.add_argument('-outdir', default="/path/to/output/",
					help="directory to save the outputs.")
parser.add_argument('-batch_size', type=int, default=50,
					help='Maximum batch size')
parser.add_argument('-dropout', type=float, default=0.2,
					help='Dropout probability.')

# miscellaneous

parser.add_argument('-gpus', action="store_true",
					help="Use CUDA on the listed devices.")
parser.add_argument('-datapath', default="/idiap/temp/dram/pytorch/data/",
					help="Path to data files for training.")
parser.add_argument('-mfcc', action="store_true",
					help="If you would like to use MFCC features.")
parser.add_argument('-query_list', default="/path/to/query_list/",
					help="Path to the list of queries to be evaluated.")
parser.add_argument('-search_list', default="/path/to/search_list/",
					help="Path to the list of files of search utterances.")

opt = parser.parse_args()
print(opt)

if not os.path.isdir(opt.outdir):
	os.mkdir(opt.outdir)

def _batchify(query, search, len_query, len_search, maxWidth, maxLength):
	out = query[0].new(len(query), 1, maxWidth, maxLength).fill_(-1)
	eps = 1e-5
	for i in range(len(query)):
		dist = torch.log(torch.mm(query[i], search[i]) + eps)
		dist = -1 + 2* (dist - dist.min())/(dist.max() - dist.min())
		# length of the distance matrix
		length = len_search[i]
		# width of the distance matrix
		width = len_query[i]
		ind_length = torch.LongTensor(compression_index(length, maxLength))
		ind_width = torch.LongTensor(compression_index(width, maxWidth))
		dist = dist.index_select(0, ind_width)
		dist = dist.index_select(1, ind_length)
		out[i].narrow(1, (maxWidth - dist.size(0))/2, dist.size(0)).narrow(2, (maxLength - dist.size(1))/2, dist.size(1)).copy_(dist)
	if opt.gpus:
		out = out.cuda()
	out = Variable(out, volatile=True)
	return out

def compression_index(length, max_length):
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


def main():
	if opt.mfcc:
		print('Loading mfcc features for evaluation queries and search audio')
		eval_queries = torch.load(opt.datapath + '/mfcc_feature_eval_queries.pt')
		audio = torch.load(opt.datapath + '/mfcc_feature_audio.pt')
	else:
		print('Loading posterior features for evaluation queries and search audio')
		eval_queries = torch.load(opt.datapath + '/posterior_feature_eval_queries.pt')
		audio = torch.load(opt.datapath + '/posterior_feature_audio.pt')

	maxWidth, maxLength = 200, 750
	model = Model.ClassifierCNN(opt.depth, opt.dropout)
	if opt.load_model:
		print("Loading learned model for evaluation ...")
		if opt.gpus:
			checkpoint = torch.load(opt.modelpath, map_location={'cuda:0':'cuda:0'})
			model.load_state_dict(checkpoint['state_dict'])
		else:
			checkpoint = torch.load(opt.modelpath, map_location={'cuda:0':'cpu'})
			model.load_state_dict(checkpoint['state_dict'])

	if opt.gpus:
		model = model.cuda()

	model.eval()
	print(model)
	fid_query = open(opt.query_list, 'r')
	for query in fid_query:
		query = query.strip()
		print query
		query_len = len(eval_queries[query])
		fid_result = open(opt.outdir +'/'+ query + '.txt','w')
		query_feature = eval_queries[query]
		fid_search = open(opt.search_list, 'r')
		query_batch, search_batch, query_length, search_length, search_name = [], [], [], [], []
		batch = 0
		for search in fid_search:
			search = search.strip()
			if search in audio.keys():
				search_len = len(audio[search])
				if search_len > 10:
					search_feature = audio[search]
					if search_len > 0.5* query_len:
						search_name += [search]
						query_batch += [query_feature]
						query_length += [query_len]
						search_batch += [search_feature.transpose(0,1).contiguous()]
						search_length += [search_len]
						batch += 1
						if batch == opt.batch_size:
							dist = _batchify(query_batch, search_batch, query_length, search_length, maxWidth, maxLength)
							output = model(dist)
							batch = 0
							for ind in range(opt.batch_size):
								fid_result.write("{0:s} {1:10d} {2:10d} {3:10.4f} {4:10.4f}\n".format(search_name[ind], 0, 100, output.data.cpu().numpy()[ind][0], output.data.cpu().numpy()[ind][1]))
							query_batch, search_batch, query_length, search_length, search_name = [], [], [], [], []
					else:
						fid_result.write("{0:s} {1:10d} {2:10d} {3:10.4f} {4:10.4f}\n".format(search, 0, 0, 0, -10))
				else:
					fid_result.write("{0:s} {1:10d} {2:10d} {3:10.4f} {4:10.4f}\n".format(search, 0, 0, 0, -10))
			else:
				fid_result.write("{0:s} {1:10d} {2:10d} {3:10.4f} {4:10.4f}\n".format(search, 0, 0, 0, -10))
		# evaluate the incomplete batch which has been processed
		if len(search_length) > 0:
			dist = _batchify(query_batch, search_batch, query_length, search_length, maxWidth, maxLength)
			output = model(dist)
			batch = 0
			for ind in range(len(search_length)):
				fid_result.write("{0:s} {1:10d} {2:10d} {3:10.4f} {4:10.4f}\n".format(search_name[ind], 0, 100, output.data.cpu().numpy()[ind][0], output.data.cpu().numpy()[ind][1]))
			query_batch, search_batch, query_length, search_length, search_name = [], [], [], [], []

		fid_result.close()
		fid_search.close()
	fid_query.close()

if __name__ == "__main__":
    main()

