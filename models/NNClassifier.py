#=========================================================#
#                     author:                             #                           
#                     date:                               #
#                     dateset:                            #
#                     model:                              #
#=========================================================#

#packages
import numpy as np

#neural network
class NeuralNetwork(object):
	'''Two layer neural network
	methods: __init__ as constructor, loss to calculate loss, train to train, predict to predict
	structure: input -> dense -> relu -> dense -> softmax -> output
	'''
	def __init__(self, input_size, output_size, opt):   #opt: hidden_size, std, and std stands for standard deviation
		'''constructor
		initialize parameters
		'''
		self.params = {}
		self.params['W1'] = opt.NN_standard * np.random.randn(input_size, opt.hidden_size)
		self.params['b1'] = np.zeros(opt.hidden_size)
		self.params['W2'] = opt.NN_standard * np.random.randn(opt.hidden_size, output_size)
		self.params['b2'] = np.zeros(output_size)

	def loss(self, X, y, opt):   #opt: reg as regulization
		'''calculate loss and grads
		'''
		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']

		N, D = X.shape   #N as batch size, D as input size

		hidden_output = np.maximum(0, np.dot(X, W1) + b1)
		output = np.dot(hidden_output, W2) + b2

		f = output - np.max(output, axis=1, keepdims=True)
		loss = -f[range(N), y].sum() + np.log(np.exp(f).sum(axis=1)).sum()   #softmax loss
		loss = loss / N + 0.5 * opt.NN_regulization * (np.sum(W1 * W1) + np.sum(W2 * W2))   #regulization

		grads = {}

		doutput = np.exp(f) / np.exp(f).sum(axis=1, keepdims=True)   #softmax
		doutput[range(N), y] -= 1
		doutput /= N

		grads['W2'] = np.dot(hidden_output.T, doutput) + opt.NN_regulization * W2
		grads['b2'] = np.sum(doutput, axis = 0)

		dhidden = np.dot(doutput, W2.T)
		dhidden[hidden_output <= 0.00001] = 0

		grads['W1'] = np.dot(X.T, dhidden) + opt.NN_regulization * W1
		grads['b1'] = np.sum(dhidden, axis = 0)

		return loss, grads

	#def train(self, X, y, learning_rate=1e-3, learning_rate_decay=0.95, reg=1e-5, epoch=100, batch_size=200):
	def train(self, X, y, X_val, y_val, opt):
		#opt: learning_rate, learning_rate_decay, regulization, epoch, batch_size
		'''train
		'''
		lr = opt.learning_rate

		training_set_len = X.shape[0]   #length of training set
		#iterations_per_epoch = max(training_set_len / opt.batch_size, 1)

		loss_log = []   #log
		train_acc_log = []
		val_acc_log = []

		for e in range(opt.epoch):
			indices = np.random.choice(training_set_len, opt.batch_size, replace=True)   #sampler
			X_batch = X[indices]
			y_batch = y[indices]

			loss, grads = self.loss(X_batch, y_batch, opt)
			loss_log.append(loss)

			self.params['W1'] -= lr * grads['W1']
			self.params['b1'] -= lr * grads['b1']
			self.params['W2'] -= lr * grads['W2']
			self.params['b2'] -= lr * grads['b2']
			
			if e % 100 == 0:
				#print('epoch {} / {}: loss {}'.format(e, opt.epoch, loss))

			#if e % iterations_per_epoch == 0:
				train_acc = (self.predict(X_batch) == y_batch).mean()
				#print(self.predict(X_val) == y_val)
				#print(self.predict(X_val).shape)
				#print(y_val.shape)
				val_acc = (self.predict(X_val) == y_val).mean()
				print('epoch {} / {}: loss {}, acc {}, val {}'.format(e, opt.epoch, loss, train_acc, val_acc))
				train_acc_log.append(train_acc)
				val_acc_log.append(val_acc)

				lr *= opt.learning_rate_decay

		return {'loss_log': loss_log, 'train_acc_log': train_acc_log, 'val_acc_log': val_acc_log}

	def predict(self, X):
		'''pred
		'''
		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']

		hidden_output = np.maximum(0, np.dot(X, W1) + b1)
		output = np.dot(hidden_output, W2) + b2
		pred = np.argmax(output, axis=1)

		return pred
