#=========================================================#
#                     author:                             #                           
#                     date:                               #
#                     dateset:                            #
#                     model:                              #
#=========================================================#

#packages
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd

import torch
import torchvision
import torchvision.transforms as transforms

import time

from models.NNClassifier import NeuralNetwork
#from models.SVMClassifier import SVM

#cifar10 dataloader
def dataloader(opt):
    '''cifar10 dataloader
    download cifar10 from torchvision, return them as some form
    '''
    training_data = []
    training_label = []
    validation_data = []
    validation_label = []
    
    '''
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    '''
    
    training_set = torchvision.datasets.MNIST(root=opt.root, train=True, download=True)
    #training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True, num_workers=2)
    
    testing_set = torchvision.datasets.MNIST(root=opt.root, train=False, download=True)
    #testing_loader = torch.utils.data.DataLoader(testing_set, batch_size=4, shuffle=False, num_workers=2)
    
    for t in training_set:
        training_data.append(np.array(t[0]).reshape(-1))
        training_label.append(t[1])
    training_data = np.array(training_data)
    #print(len(training_data))
    training_label = np.array(training_label)
    
    for t in testing_set:
        validation_data.append(np.array(t[0]).reshape(-1))
        validation_label.append(t[1])
    validation_data = np.array(validation_data)
    validation_label = np.array(validation_label)
    
    return training_data, training_label, validation_data, validation_label


if __name__ == '__main__':
	parser = argparse.ArgumentParser('arguments for neural network classifier and SVM')

	parser.add_argument(
		'-m', '--model',
		help='decide which model, NN or SVM',
		default='NN')

	#sharable arguments for both classifiers
	parser.add_argument(
		'-r', '--root',
		help='data root',
		default="datasets")
	parser.add_argument(
		'-tr_s', '--trainingset_size', type=int,
		help='the size of the training size',
		default=60000)

	parser.add_argument(
		'-e', '--epoch', type=int,
		help='number of epoch, suggested for NN is 500000, suggested for SVM is 500000',
		default=5000)
	parser.add_argument(
		'-b', '--batch_size', type=int,
		help='batch size, suggested for NN is 50, suggested for SVM is 50',
		default=50)  
	parser.add_argument(
		'-lr', '--learning_rate', type=float,
		help='learning rate, suggested for NN is 1e-4, suggested for SVM is 1e-4',
		default=1e-4)

	#exclusive arguments for neural network
	parser.add_argument(
		'-lr_dk', '--learning_rate_decay', type=float,
		help='learning rate decay, exclusively for NN',
		default=0.999995)
	parser.add_argument(
		'-h_s', '--hidden_size', type=int,
		help='neural network hidden layer size, exclusively for NN',
		default=256)

	parser.add_argument(
		'-nn_std', '--NN_standard', type=float,
		help='initialization standard deviation for neural network',
		default=1e-4)
	parser.add_argument(
		'-nn_reg', '--NN_regulization', type=float,
		help='regulization term for neural network',
		default=1e-5)

	#exclusive arguments for SVM
	parser.add_argument(
		'-svm_std', '--SVM_standard', type=float,
		help='initialization standard deviation for SVM, suggested: 1e-3',
		default=1e-3)
	parser.add_argument(
		'-svm_reg', '--SVM_regulization', type=float,
		help='regulization term for SVM, suggested: 1e-4',
		default=1e-4)
	
	args = parser.parse_args()

	X, y, X_val, y_val = dataloader(args)

	input_size = 784
	num_classes = 10

	if args.model == 'NN':
		model = NeuralNetwork(input_size, num_classes, args)
	else:
		model = SVM(input_size, num_classes, args)

	model = NeuralNetwork(input_size, num_classes, args)

	log = model.train(X, y, X_val, y_val, args)

	'''
	plt.subplot(2, 1, 1)
	plt.plot(log['loss_log'])
	plt.title('Loss')
	plt.xlabel('batch')
	plt.ylabel('loss')

	plt.subplot(2, 1, 2)
	plt.plot(log['train_acc_log'], label='train')
	plt.plot(log['val_acc_log'], label='val')
	plt.title('accuracy')
	plt.xlabel('epoch')
	plt.ylabel('acc')
	plt.show()
	'''
