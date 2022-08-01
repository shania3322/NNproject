import numpy as np
import numpy.typing as npt
from typing import Tuple, List

class Network(object):
	"""
	A smallest network with 2 hidden layers for MNIST hand-written digit
	classification. Each image has 784 (28*28) pixels with value range from 0 to
	255.

	 :sizes: tuple(784, 256, 64, 10). Two hidden layers with size (256,64)
	 :batch_size: the number of images in one batch,
	  number_of_batch= (totoal number of images)/batch_size
	"""
	def __init__(self, size:Tuple=(784,256,64,10), batch_size:int=5):
		self.size = size
		self.num_layers = len(size)
		self.batch_size = batch_size
		self.weights = [np.random.normal(0.0,0.05,size=(x)) for x in
				zip(size[:-1],size[1:])]
		self.bias = [np.zeros((1,x)) for x in size[1:]]
		# each layer of self.acts store values before and after activiation
		# [middle_state_before_activiation, middle_state_after_activiation]
		self.acts = []

	def sigmoid(self,z):
		"""
		z: [N, features]
		"""
		return 1/(1+np.exp(-z))

	def softmax(self,z):
		"""
		z: [N, n_classes]
		"""
		numerator = np.exp(z)
		denominator = np.sum(numerator, axis=1).reshape((-1,1))
		return numerator/denominator

	def normalization(self):
		...

	def onehot(self, num):
		"""
		:num:the output of last layer, size [N,1], N is the batch_size
		"""
		ret = np.zeros((num.size,10),dtype=np.int)
		l = ret.shape[0]
		if l==1:
			ret[0,num]=1
		else:
			for i in range(ret.shape[0]):
				ret[i,num[i]]=1
		return ret

	def SGD(self, train_data:Tuple[List,List], epochs: int=2, lr: float=1.0e-4,
			test_data=None):
		"""
		Stochastic gradient descent:
		optimizing an objective function by iteratively approximate the
		objective with the gradient calulated from a subset of data (or "batch").
		if test_data is given, evaluate agains test_data after each epoch.

		w_minibatch,b_minibatch =
		avg(weights_of_n_example),avg(bias_of_n_example)

		Then update the network with w_minibatch and b_minibatch
		:train_data: (data, labels)
		:test_data: (data, labels) labels are onehot format
		"""
		data,label = train_data
		num_examples = data.shape[0]
		rng = np.random.default_rng()
		for epoch in range(epochs):
			shuffle = rng.permutation(num_examples)
			data_, label_ = data[shuffle,:],label[shuffle,:]
			iteration = num_examples//self.batch_size
			for i in range(iteration):
				subset = data_[i*self.batch_size:(i+1)*self.batch_size]
				l = label_[i*self.batch_size:(i+1)*self.batch_size]
				d = self.mini_batch(subset,l)
				self.update_network(d,lr)

			# Handling the last batch
			num = iteration*self.batch_size
			if  num < num_examples:
				subset,l = data_[num:],label_[num:]
				d = self.mini_batch(subset,l)
				self.update_network(d,lr)

			if test_data is not None:
				t_d,t_l = test_data
				pred = []
				for i in range(t_d.shape[0]):
					x = np.argmax(self.forward(t_d[i]))
					print(f'prediction-{x}- label-{np.argmax(t_l[i])}')
					pred.append(self.onehot(x))

				accuracy = self.eval(np.asarray(pred).squeeze(), t_l)
				print(f'Epoch-{epoch}-----')
				print(f"Test-accuracy: {accuracy:.2f}")



	def mini_batch(self, x, label):
		"""
		Accumulating the weights and bias from one example. When reaching the
		batch_size, averaging weights and bias by multiplying 1/n.
		"""
		data = x
		n = data.shape[0]
		weights_acc = [np.zeros_like(w) for w in self.weights]
		bias_acc = [np.zeros_like(b) for b in self.bias]
		for i in range(n):
			self.forward(data[i])
			derivatives = self.backprop(data[i].reshape(1,-1),label[i].reshape(1,-1))
			weights_acc = [z[0]+z[1] for z in zip(weights_acc,derivatives["weights_d"])]
			bias_acc = [z[0]+z[1] for z in zip(bias_acc,derivatives["bias_d"])]

		weights = [z/n for z in weights_acc]
		bias = [z/n for z in bias_acc]
		return weights, bias

	def cross_validation(self):
		...

	# derivatives
	def MSE_loss_diff(self, pred, label):
		"""
		The derivative of MSE loss function f=1/n*sum((label-pred)^2), 1/n can
		be omitted as it is a scaler
		"""
		return (pred-label)

	def sigmoid_diff(self,z):
		# returned array size
		return self.sigmoid(z)*(1-self.sigmoid(z))

	def softmax_diff(self,z):
		...

	def forward(self,x):
		"""
		Passing one example(one image in the case of MNIST) as the input to the
		network and get the prediction as the output.

		Middle states need to be save for backpropagation:
		Middle states include w_i, b_i, and h_i(activiation).

		x: input, vector with size [1,784].T
		y: output, vector with size [1,10]
		w1: weights [784,256].T
		b1: bias [1,256].T
		z1: [1,256].T
		w2: [256,64].T
		b2: [1,64].T
		z2: [1,64].T
		w3: [64,10].T
		b3: [1,10].T
		z3: [1,10].T
		fn_1(): activiation function for hidden layers.Here we use sigmoid.
		fn_2(): output activiation function.
		"""

		fn_1 = getattr(self,'sigmoid')
		#fn_2 = getattr(self,'softmax') #for simplicity use sigmoid instead
		w1,w2,w3 = self.weights
		b1,b2,b3 = self.bias
		z1 = np.matmul(x,w1)+b1
		z1_ = fn_1(z1)
		self.acts.append([z1,z1_])
		z2 = np.matmul(z1,w2)+b2
		z2_ = fn_1(z2)
		self.acts.append([z2,z2_])
		z3 = np.matmul(z2,w3)+b3
		y = fn_1(z3)
		self.acts.append([z3,y])
		return y

	def backprop(self, x, label):
		"""
		Calculate derivatives with respect to nodes(w_i and b_i) and save the
		result for later update.

		 :x: input image with shape (1,784)
		 :label: the label of the input image with shape (1,10) in onehot format
		"""
		derivatives = {
				"weights_d": [np.zeros_like(w) for w in self.weights],
				"bias_d": [np.zeros_like(b) for b in self.bias],
				"acts": [[] for _ in range(self.num_layers-1)],
				}

		#for i in range(3):
		#    print(derivatives["weights_d"][i].shape)
		#    print(derivatives["bias_d"][i].shape)
		#print(derivatives["acts"])
		d1 = self.acts[-1][1]-label
		derivatives["acts"][-1] = d1
		d2 = self.sigmoid_diff(self.acts[-1][0])
		d_b3 = d1*d2 # shape (1,10)
		derivatives["bias_d"][-1] = d_b3
		d4 = np.tile(self.acts[-2][1].T, self.size[-1]) #shape (64,10)
		d_w3 = d4*np.squeeze(d_b3) #shape (64,10)
		derivatives["weights_d"][-1] = d_w3
		d_a2 = np.sum(self.weights[-1]*np.squeeze(d_b3), axis=1).reshape(1,-1)
		derivatives["acts"][-2] = d_a2 #shape (1,64)
		d_b2 = self.sigmoid_diff(self.acts[-2][0])*d_a2
		derivatives["bias_d"][-2] = d_b2 #shape (1,64)
		d5 = np.tile(self.acts[-3][1].T,self.size[-2])
		d_w2 = d5*np.squeeze(d_b2) #shape (256,64)
		derivatives["weights_d"][-2] = d_w2
		d_a1 = np.sum(self.weights[-2]*np.squeeze(d_b2),axis=1).reshape(1,-1)
		derivatives["acts"][-3] = d_a1 # shape (1,256)
		d_b1 = self.sigmoid_diff(self.acts[-3][0])*d_a1 # shape (1,256)
		derivatives["bias_d"][-3] = d_b1
		d6 = np.tile(x.T, self.size[-3])
		d_w1 = d6*np.squeeze(d_b1) # shape (784,256)
		derivatives["weights_d"][-3] = d_w1
		return derivatives


	def update_network(self, derivatives, lr):
		"""
		Update network with v'=v-lr*derivatives

		:lr: learning rate, default value 1.0e-4
		:derivatives: a dictionary with all the derivatives
		"""
		w,b = derivatives
		weights_update = [np.zeros_like(w) for w in self.weights]
		bias_update = [np.zeros_like(b) for b in self.bias]
		weights_update = [w[0]-lr*w[1] for w in zip(self.weights,w)]
		bias_update = [b[0]-lr*b[1] for b in zip(self.bias,b)]
		self.weights = weights_update
		self.bias = bias_update


	def computational_graph(self):
		# TODO: build a computational graph to make forward and backprop easier.
		...

	def eval(self, pred, label):
		# pred and label are onehot format
		if pred.shape==label.shape:
			res = [sum(np.abs(x[0]-x[1])) for x in zip(pred,label)]
			res = np.array(res)
			r = res[res==0] if res[res==0] else 0.0
			return r/len(label)
		else:
			print("Error: predition and label have to be the same shape.")

