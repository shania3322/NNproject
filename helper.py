# Helper functions for data loading and processing
# Dataset is downloaded from http://yann.lecun.com/exdb/mnist/. Check the
# website for the reference of idx file format

import numpy as np
import io
from PIL import Image as im
import numpy.typing as npt
import glob

class DataLoader:
    def __init__(self, folder:str='data/'):
        self.root = folder
        self.paths = {}

    def get_path(self):
        itr = glob.iglob(self.root+"**")
        while True:
            try:
                p = next(itr)
                if p.startswith(self.root+'train-images'):
                    self.paths['train_data']=p
                elif p.startswith(self.root+'train-labels'):
                    self.paths['train_labels']=p
                elif p.startswith(self.root+'t10k-images'):
                    self.paths['test_data']=p
                elif p.startswith(self.root+'t10k-labels'):
                    self.paths['test_labels']=p
            except StopIteration:
                break

    def read_img(self, file:str,num: int=None) -> npt.ArrayLike:
        try:
            metainfo = np.fromfile(file, dtype=np.ubyte, count=16)
            data = np.fromfile(file, dtype=np.ubyte, count=28*28*num, offset=16)
            data = np.reshape(data,(num,28,28))
            return data,metainfo
        except BaseException as err:
            print(f'err {err=} {type(err)=}')
            raise

    def read_label(self, file: str, num: int=None):
        try:
            metainfo = np.fromfile(file, dtype=np.ubyte, count=8)
            data = np.fromfile(file, dtype=np.ubyte, count=num, offset=8)
            return data,metainfo
        except BaseException as err:
            print(f'err {err=} {type(err)=}')
            raise

    def load_data(self, train:bool=True, num:int=10) -> npt.ArrayLike:
        self.get_path()
        if train:
            data,_ = self.read_img(self.paths['train_data'], num)
            label,_ = self.read_label(self.paths['train_labels'], num)
        else:
            data,_ = self.read_img(self.paths['test_data'], num)
            label,_ = self.read_label(self.paths['test_labels'], num)

        return data, label


    def display(self, arr:npt.ArrayLike) -> None:
        try:
            img = im.fromarray(arr)
            img.show()
        except:
            print(f'Something wrong')


# debug
def main():
    dataloader = DataLoader(train=True,num=10)
    data, label = dataloader.load_data()
    print(f'data: {data.shape}, label: {label.shape}, {label}')
    dataloader.display(data[2])

if __name__=='__main__':
    main()

