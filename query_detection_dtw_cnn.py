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


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy
import random
import Dataset_DTW_CNN as Dataset
import Model_Query_Detection_DTW_CNN as Model
import os
import argparse


parser = argparse.ArgumentParser(description='query_detection_dtw_cnn.py')

## Model options

parser.add_argument('-depth', type=int, default=30,
					help='depth of features in each layer of the CNN')
parser.add_argument('-input_size', type=int, default=152,
					help='Length of the input feature vector')
parser.add_argument('-load_model', action='store_true',
					help='Load a previously trained model')
parser.add_argument('-modelpath', default="/path/to/models/",
					help="Path to saved model to be loaded for retraining.")

## Optimization options

parser.add_argument('-batch_size', type=int, default=20,
					help='Maximum batch size')
parser.add_argument('-epochs', type=int, default=1000,
					help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
					help='The epoch from which to start')
parser.add_argument('-optim', default='adam',
					help="Optimization method. [sgd | adagrad | adadelta | adam]")
parser.add_argument('-learning_rate', type=float, default=0.001,
					help="""Starting learning rate. If adagrad/adadelta/adam is
					used, then this is the global learning rate. Recommended
					settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")
parser.add_argument('-loss', default='nll-loss',
					help="Optimization method. [nll-loss | cross-entropy]")
parser.add_argument('-dropout', type=float, default=0.1,
					help='Dropout probability.')
parser.add_argument('-max_batch_dev', type=int, default=500,
					help='Maximum number of batches to be used for development')
parser.add_argument('-max_batch_train', type=int, default=1000,
					help='Maximum number of batches to be used for training in each iteration')
parser.add_argument('-shuffle', action="store_true",
					help="If you would like to shuffle the training examples.")

# miscellaneous
parser.add_argument('-gpus', action="store_true",
					help="Use CUDA on the listed devices.")
parser.add_argument('-datapath', default="/path/to/data/",
					help="Path to data files for training.")
parser.add_argument('-outpath', default="/path/to/output/",
					help="Path to save models.")
parser.add_argument('-n_valid', type=int, default=50,
					help="""Number no of batches after which we evaluate 
					the loss on validation set""")
parser.add_argument('-mfcc', action="store_true",
					help="If you would like to use mfcc features.")
parser.add_argument('-loss_threshold', type=float, default=0.1,
					help="Threshold value of loss to start saving models.")


opt = parser.parse_args()
print(opt)

if opt.mfcc:
	outdir = opt.outpath+'/model_mfcc_DTW_CNN_depth'+str(opt.depth)+'_dropout'+str(opt.dropout)+'_'+opt.optim+'_lr'+str(opt.learning_rate)+'_'+opt.loss
else:
	outdir = opt.outpath+'/model_posterior_DTW_CNN_depth'+str(opt.depth)+'_dropout'+str(opt.dropout)+'_'+opt.optim+'_lr'+str(opt.learning_rate)+'_'+opt.loss

if not os.path.isdir(outdir):
	os.mkdir(outdir) 

def eval(model, criterion, batches):
	model.eval()
	loss, total_loss = 0, 0
	total_correct, total_correct_p, total_correct_n = 0, 0, 0
	ind = 0
	for dbatch, lbatch in batches:
		if opt.gpus:
			dbatch, lbatch = dbatch.cuda(), lbatch.cuda()
		dbatch = Variable(dbatch, volatile=True)
		lbatch = Variable(lbatch, volatile=True)
		output = model(dbatch)
		loss = criterion(output, lbatch)
		total_loss = total_loss + round(float(loss.data[0]),5)
		correct_p, correct_n = 0, 0
		ind = lbatch.data.type_as(torch.LongTensor())
		output = output.data
		for i in range(len(output)):
			if ind[i] == 1:
				if numpy.exp(output[i,1]) > 0.5:
					correct_p = correct_p + 1
			else:
				if 1 - numpy.exp(output[i,1]) > 0.5:
					correct_n = correct_n + 1
		total_correct_p = total_correct_p + correct_p
		total_correct_n = total_correct_n + correct_n
	model.train()
	return total_loss, total_correct_p, total_correct_n

def main():
	print("Loading data from dev set ...")
	if opt.mfcc:
		print('Loading mfcc features for dev_queries')
		query = torch.load(opt.datapath+'/mfcc_feature_dev_queries.pt')
		print('Loading mfcc features for search space')
		search = torch.load(opt.datapath + '/mfcc_feature_audio.pt')
	else:
		print('Loading posteriors features for dev_queries')
		query = torch.load(opt.datapath + '/posterior_feature_dev_queries.pt')
		print('Loading posteriors features for search space')
		search = torch.load(opt.datapath + '/posterior_feature_audio.pt')

	print('Loading groundtruth file for training: GroundTruth_label_train.txt')
	ground_train_pos = open('GroundTruth_label_train_complete_pos.txt','r').readlines()
	ground_train_neg = open('GroundTruth_label_train_complete_neg.txt','r').readlines()
	print('Loading groundtruth file for development: GroundTruth_label_dev.txt')
	ground_dev = []
	with open('GroundTruth_label_dev.txt') as fid:
		ground_dev = [tuple(line.split()) for line in fid]
	ground_dev = sorted(ground_dev, key=lambda x:int(x[3]))
	maxWidth, maxLength = 200, 750
	data_dev = Dataset.Dataset(ground_dev, query, search, maxWidth, maxLength, opt.batch_size, opt.max_batch_dev, opt.gpus, False, volatile=True)
	batches_dev = data_dev.makeBatches_dev()

	model = Model.ClassifierCNN(opt.depth, opt.dropout)

	# define loss function (criterion) and optimizer
	if opt.loss == 'nll-loss':
		criterion = nn.NLLLoss()
	elif opt.loss == 'cross-entropy':
		criterion = nn.CrossEntropyLoss()
	else:
		raise RuntimeError("Invalid loss function: " + opt.loss)

	# define loss function (criterion) for evaluation
	if opt.loss == 'nll-loss':
		criterion_dev = nn.NLLLoss(torch.Tensor([0.0144,0.9856]))
		criterion_eval = nn.NLLLoss(torch.Tensor([0.0008,0.9992]))
	elif opt.loss == 'cross-entropy':
		criterion_dev = nn.CrossEntropyLoss(torch.Tensor([0.0144,0.9856]))
		criterion_eval = nn.CrossEntropyLoss(torch.Tensor([0.0008,0.9992]))
	else:
		raise RuntimeError("Invalid loss function: " + opt.loss)

	if opt.gpus:
		model = model.cuda()
		criterion = criterion.cuda()
		criterion_dev = criterion_dev.cuda()
		criterion_eval = criterion_eval.cuda()

	if opt.optim == 'sgd':
		optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate)
	elif opt.optim == 'adagrad':
		optimizer = optim.Adagrad(model.parameters(), lr=opt.learning_rate)
	elif opt.optim == 'adadelta':
		optimizer = optim.Adadelta(model.parameters(), lr=opt.learning_rate)
	elif opt.optim == 'adam':
		optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
	else:
		raise RuntimeError("Invalid optim method: " + opt.optim)

	if opt.load_model:
		print("Loading previously learned model ...")
		if opt.gpus:
			checkpoint = torch.load(opt.modelpath, map_location={'cuda:0':'cuda:0'})
		else:
			checkpoint = torch.load(opt.modelpath, map_location={'cuda:0':'cpu'})
		opt.start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])

	model.train()
	print(model)
	# threshold value of loss to start saving models
	loss_threshold = opt.loss_threshold
	n_batch, n_pos, n_neg = numpy.ceil(len(ground_dev)/float(opt.batch_size)), float(72), float(10508)

	for epoch in range(opt.start_epoch, opt.epochs+1):
		print "epoch:", epoch
		random.shuffle(ground_train_neg)
		ground_train = ground_train_pos + ground_train_neg[0:len(ground_train_pos)]
		data_train = Dataset.Dataset(ground_train, query, search, maxWidth, maxLength, opt.batch_size, opt.max_batch_train, opt.gpus, opt.shuffle, volatile=False)
		batches_train = data_train.makeBatches()
		data_train = None
		batch, train_loss = 0, 0
		for dbatch, lbatch in batches_train:
			if opt.gpus:
				dbatch, lbatch = dbatch.cuda(), lbatch.cuda()
			dbatch = Variable(dbatch, volatile=False)
			lbatch = Variable(lbatch, volatile=False)
			optimizer.zero_grad()
			output = model(dbatch)
			loss = criterion(output, lbatch)
			loss.backward()
			optimizer.step()
			batch += 1
			train_loss = train_loss + round(float(loss.data[0]),5)
			if batch % opt.n_valid == 0:
				valid_loss, n_correct_pos, n_correct_neg = eval(model, criterion_eval, batches_dev)
				valid_loss = round(valid_loss/n_batch,5)
				acc_pos = round(n_correct_pos/n_pos,5)
				acc_neg = round(n_correct_neg/n_neg,5)
				print 'total_correct_pos, total_pos, total_correct_neg, total_neg:', n_correct_pos, n_pos, n_correct_neg, n_neg
				print "avg_train_loss, eval_criterion_loss, accuracy_pos, accuracy_neg: ", round(train_loss/opt.n_valid,5), valid_loss, acc_pos, acc_neg
				train_loss = 0

				if valid_loss < loss_threshold:
					loss_threshold = valid_loss
					torch.save({'epoch':epoch,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict()}, outdir+'/model_epoch'+str(epoch)+'_eval_criterion_loss_'+str(valid_loss))
	
if __name__ == "__main__":
    main()

