from model import Network
from helper import DataLoader, onehot
import numpy as np

def run():
    # config
    train_size=20
    batch_size=5
    train=True

    # print network
    net = Network(batch_size=batch_size)
    print(f'number of layers: {net.num_layers}, batch size: {net.batch_size}')
    for i in net.bias:
        print(f'bias: {i.shape}')
    for i in net.weights:
        print(f'weights:{i.shape}')

    # read image into array
    dataloader = DataLoader(train=train,num=train_size)
    data, label = dataloader.load_data()
    print(f'data: {data.shape}, label: {label.shape}')

    # forward
    #for _ in range(1):
    #    out = net.forward(data.reshape(-1,28*28))
    #    pred = np.argmax(out, axis=1)
    #    pred_oh = onehot(pred)
    #    label_oh = onehot(label.reshape(-1,1))

    #    for i in range(batch_size):
    #        print(f'{pred[i]}-{pred_oh[i]}-{label[i]}')
    #    # loss function - mean-sqaure. In tutorial cost'=1/2*cost
    #    cost = np.sum(np.square(label_oh-pred_oh))/batch_size
    #    print(cost)


    #for i in net.acts:
    #    t1,t2 = i[0],i[1]
    #    if t1 is not None and t2 is not None:
    #        print(f'self.acts:{t1.shape,t2.shape}')
    #    else:
    #        print(f'self.acts:{t1.shape}')

    #net.backprop(data.reshape(-1,28*28),label_oh)
    label_onehot = onehot(label.reshape(-1,1))
    #print(label_onehot.shape)
    train_data,test_data = data[:(train_size-5)],data[-5:]
    train_label,test_label = label_onehot[:(train_size-5)],label_onehot[-5:]

    train = (train_data.reshape(-1,28*28), train_label)
    test = (test_data.reshape(-1,28*28), test_label)
    #print(f'{train[0].shape,train[1].shape,test[0].shape,test[1].shape}')
    net.SGD(train_data=train,test_data=test)


if __name__ == "__main__":
    run()

