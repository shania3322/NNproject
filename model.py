import time
import numpy as np
import numpy.typing as npt
from typing import Tuple, List
import warnings

warnings.filterwarnings('ignore')

class Network(object):
    """
    A smallest network with 2 hidden layers for MNIST hand-written digit
    classification. Each image has 784 (28*28) pixels with value range from 0 to
    255.

     :sizes: tuple(784, 256, 64, 10). Two hidden layers with size (256,64)
     :batch_size: the number of images in one batch,
      number_of_batch= (totoal number of images)/batch_size
    """
    def __init__(self, size:Tuple=(784,32,10), batch_size:int=5):
        """
        Each layer of self.acts store values before and after activiation
        [middle_state_before_activiation, middle_state_after_activiation]
        """
        self.size = size
        self.num_layers = len(size)
        self.batch_size = batch_size
        self.weights = [np.random.normal(0.0,1.0,size=(x)) for x in
                zip(size[:-1],size[1:])]
        self.bias = [np.random.normal(0.0,1.0,size=(1,x)) for x in size[1:]]
        self.acts = []

    def __repr__(self):
        ret = "Network:\n"
        for i in range(self.num_layers-1):
            ret=ret+f'  layer-{i}: weight {self.weights[i].shape}, '
            ret=ret+f'bias {self.bias[i].shape}\n'
        ret = ret+f'Parameters:\n  batch_size:{self.batch_size}'
        return ret

    def sigmoid(self,z):
        """
        z: [1, features]
        """
        # Warning for overflow is suppressed.
        return 1/(1+np.exp(-z))

    def softmax(self,z):
        """
        z: [1, n_classes]
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

    def SGD(self, train_data:Tuple[List,List], epochs:int, cv_split: int=4,\
            lr:float=0.1):
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
        begin = time.time()
        data,label = train_data
        print("----------------")
        for epoch in range(epochs):
            gen = self.shuffle_split(data, label, n_split=cv_split)
            acc_accuracy = []
            for kth_fold in range(cv_split):
                try:
                    train, test = next(gen)
                except StopIteration:
                    print(f'epoch {epoch} completed. ')

                data_,label_ = train
                label_ = self.onehot(label_)
                num_examples = data_.shape[0]
                iteration = num_examples//self.batch_size
                loss = 0
                for i in range(iteration):
                    subset = data_[i*self.batch_size:(i+1)*self.batch_size]
                    l = label_[i*self.batch_size:(i+1)*self.batch_size]
                    w,b,loss1 = self.mini_batch(subset,l)
                    self.update_network((w,b),lr)
                    loss = loss+loss1

                # Handling the last batch
                num = iteration*self.batch_size
                if  num < num_examples:
                    subset,l = data_[num:],label_[num:]
                    w,b,loss2 = self.mini_batch(subset,l)
                    self.update_network((w,b),lr)
                    loss = loss+loss2

                loss_avg = loss/num
                print(f'Epoch {epoch}--{kth_fold} fold--loss-{loss_avg:.4f}')

                t_d,t_label = test
                #print(t_d.shape, t_label.shape)
                pred = []
                for i in range(t_d.shape[0]):
                    logit = self.forward(t_d[i],mode="test")
                    x = np.argmax(logit)
                    #print(f'Prediction-{x}::label-{t_label[i]}')
                    pred.append(x)
                accuracy = self.eval(pred, t_label)
                acc_accuracy.append(accuracy)
            avg_accuracy = sum(acc_accuracy)/cv_split
            elapsed = time.time()-begin
            print(f'Elapsed:{elapsed:.4f} Epoch {epoch}--Test-accuracy:{avg_accuracy:.2f}\n')


    def mini_batch(self,x,label):
        """
        Accumulating the weights and bias from one example. When reaching the
        batch_size, averaging weights and bias by multiplying 1/n.
        """
        data = x
        n = data.shape[0]
        weights_acc = [np.zeros_like(w) for w in self.weights]
        bias_acc = [np.zeros_like(b) for b in self.bias]
        loss_acc = 0
        for i in range(n):
            y_h = self.forward(data[i],mode="train")
            loss = self.MSE_loss(y_h,label[i])
            loss_acc +=loss
            derivatives = self.backprop(data[i].reshape(1,-1),label[i].reshape(1,-1))
            weights_acc = [z[0]+z[1] for z in zip(weights_acc,derivatives["weights_d"])]
            bias_acc = [z[0]+z[1] for z in zip(bias_acc,derivatives["bias_d"])]

        weights = [z/n for z in weights_acc]
        bias = [z/n for z in bias_acc]
        return weights, bias, loss_acc


    def shuffle_split(self, data, label, n_split:int=4, test_ratio:float=0.01):
        """
        Alternative to cross-validation
        """
        label = label.reshape(-1,1)
        num_examples = data.shape[0]
        rng = np.random.default_rng()
        n = max(int(num_examples*test_ratio),1)
        for i in range(n_split):
            shuffle = rng.permutation(num_examples)
            data_, label_ = data[shuffle,:],label[shuffle,:]
            train = (data_[:-n], label_[:-n].squeeze())
            test = (data_[-n:], label_[-n:].squeeze())
            yield train, test


    def MSE_loss(self,y_h,y):
        """
        y_h: predictions
        y: labels
        """
        if y_h.size==y.size:
            n = y.size
            return sum(np.square(np.squeeze(y_h)-np.squeeze(y)))/n
        else:
            print("something wrong")
            return None

    # derivatives
    def MSE_loss_diff(self,y_h,y):
        """
        The derivative of MSE loss function f=1/n*sum((predction-label)^2), 1/n can
        be omitted as it is a scaler

        y_h: predictions
        y: labels
        """
        return (y_h-y)

    def sigmoid_diff(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def softmax_diff(self,z):
        ...

    def forward(self,x,mode:str):
        """
        Passing one example(one image in the case of MNIST) as the input to the
        network and get the prediction as the output.

        Middle states need to be save for backpropagation:
        Middle states include w_i, b_i, and h_i(activiation).

        z=aw+b (or z=xw+b for the input layer)
        a'=activiation(z)

        x: input, vector with size [1,784]
        y: output, vector with size [1,10]

        For a 4-layer network with size (784, 256, 64, 10) the size of
        parameters are:
        w1: [784,256]
        b1: [1,256]
        z1: [1,256]
        w2: [256,64]
        b2: [1,64]
        z2: [1,64]
        w3: [64,10]
        b3: [1,10]
        z3: [1,10]
        fn_1(): activiation function for hidden layers.Here we use sigmoid.
        fn_2(): output activiation function.
        """
        fn_1 = getattr(self,'sigmoid')
        #fn_2 = getattr(self,'softmax') #for simplicity use sigmoid instead
        a = x
        if mode == "train":
            for w,b in zip(self.weights[:-1], self.bias[:-1]):
                z=np.matmul(a,w)+b
                a=fn_1(z)
                self.acts.append([z,a])
            z=np.matmul(a,self.weights[-1])+self.bias[-1]
            y=fn_1(z)
            self.acts.append([z,y])
        elif mode == "test":
            # If test mode, do not save middle results
            for w,b in zip(self.weights[:-1], self.bias[:-1]):
                z=np.matmul(a,w)+b
                a=fn_1(z)
            z=np.matmul(a,self.weights[-1])+self.bias[-1]
            y=fn_1(z)

        return y

    def backprop(self, x, label):
        """
        calculate derivatives with respect to nodes(w_i and b_i) and save the
        result for later update.

        x: input image with shape (1,784)
        label: the label of the input image with shape (1,10) in onehot format
        """
        derivatives = {
                "weights_d": [np.zeros_like(w) for w in self.weights],
                "bias_d": [np.zeros_like(b) for b in self.bias],
                #"acts": [[] for _ in range(self.num_layers-1)],
                }

        d_a = self.MSE_loss_diff(self.acts[-1][1],label)
        d_z = (self.sigmoid_diff(self.acts[-1][0]).T*d_a.T).T # shape (1,10)
        d_w = np.matmul(self.acts[-2][1].T, d_z)
        #derivatives["acts"][-1] = d_a
        derivatives["bias_d"][-1] = d_z
        derivatives["weights_d"][-1] = d_w
        for i in range(2,self.num_layers):
            d_a = np.matmul(self.weights[-i+1],d_z.T).T
            d_z = (self.sigmoid_diff(self.acts[-i][0]).T*d_a.T).T #(1,64)
            if i == self.num_layers-1:
                d_w = np.matmul(x.T, d_z)
            else:
                d_w = np.matmul(self.acts[-i-1][1].T, d_z)
            derivatives["bias_d"][-i] = d_z
            derivatives["weights_d"][-i] = d_w
            #derivatives["acts"][-i] = d_a
        self.acts=[]

        return derivatives


    def update_network(self, derivatives, lr):
        """
        Update network with v'=v-lr*derivatives

        :lr: learning rate, default value 1.0e-4
        :derivatives: a dictionary with all the derivatives
        """
        w,b = derivatives
        self.weights = [w[0]-lr*w[1] for w in zip(self.weights,w)]
        self.bias = [b[0]-lr*b[1] for b in zip(self.bias,b)]


    def computational_graph(self):
        # TODO: build a computational graph to make forward and backprop easier.
        ...

    def test_eval(self, x, label):
        ...

    def eval(self, pred, label):
        # pred and label are onehot format
        pred =np.array(pred)
        #print(f'shape {pred.shape}, {label.shape}')
        if pred.shape==label.shape:
            res = [x[0]==x[1] for x in zip(pred,label)]
            r = sum(res)
            return r/len(label)
        else:
            print("Error: predition and label have to be the same shape.")
            return None

    def best_params(self):
        """
        return the one best network
        """
        ...

    def save_network(self):
        """
        save checkpoint
        """
        ...

    def load_network(self):
        ...


