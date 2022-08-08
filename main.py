from model import Network
from helper import DataLoader
import numpy as np

def run():
    # config
    train_size=40000
    test_ratio=0.01
    batch_size=10

    # print network
    net = Network(batch_size=batch_size)
    print(net)
    #print(f'number of layers: {net.num_layers}, batch size: {net.batch_size}')
    #for i in net.bias:
    #    print(f'bias: {i.shape}')
    #for i in net.weights:
    #    print(f'weights:{i.shape}')

    # read image into array
    dataloader = DataLoader()
    data, label = dataloader.load_data(train=True,num=train_size)
    #print(f'data: {data.shape}, label: {label.shape}')

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

    #train_data,test_data = data[:(train_size-test_size)],data[-test_size:]
    #train_label,test_label = label[:(train_size-test_size)],label[-test_size:]
    #train = (train_data.reshape(-1,28*28), train_label)
    #test = (test_data.reshape(-1,28*28), test_label)
    #net.SGD(train_data=train,test_data=test,epochs=30)

    net.SGD(train_data=(data.reshape(-1,28*28),label), epochs=15)
    #TODO:
    # 1. Rewrite forward and backward to build any size/layer network
    # 2. Always use (N,1) for a vector.
    # 3. Save and reload weights
    # 4. Adding type hints


if __name__ == "__main__":
    run()

