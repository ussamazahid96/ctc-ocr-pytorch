import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np

class seq_mnist(Dataset):
    """docstring for seq_mnist_dataset"""
    def __init__(self, args, train_set):
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        self.data = datasets.MNIST('data', train=train_set, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
        self.args = args
        self.images = []
        self.labels = []
        self.input_lengths = np.ones(1, dtype=np.int32) * (self.args.seq_len * self.args.word_size)
        self.label_lengths = np.ones(1, dtype=np.int32) * (self.args.word_size) 
        self.build_dataset()  

    def build_dataset(self):
        imgs = []
        labels = []
        for j in range(len(self.data)//self.args.word_size): # this loop builds dataset
            img = np.zeros((self.args.input_size, self.args.word_size * self.args.seq_len))
            labs = np.zeros(self.args.word_size, dtype=np.int32)

            for i in range(self.args.word_size):  # this loop builds one example
                ims, labs[i] = self.data[(j*self.args.word_size)+i]
                ims = np.reshape(ims, (self.args.input_size, self.args.seq_len))               
                img[:, i*self.args.input_size : (i+1)*self.args.seq_len ] = ims
            
            # from PIL import Image
            # from matplotlib import cm
            # im = Image.fromarray(np.uint8(cm.gist_earth(img)*255))
            # im.save('test{}.png'.format(j))

            img = np.transpose(img)
            imgs.append(img)
            labels.append(labs)

        self.images = np.asarray(imgs, dtype=np.float32).transpose(1, 0, 2)
        self.labels.append(labels)

    def __len__(self):
        return self.images.shape[1]

    def __getitem__(self, index):
        return self.images[:,index,:], self.labels[0][index], self.input_lengths, self.label_lengths

class seq_mnist_train(seq_mnist):
    def __init__(self, args):
        print("Building Training Dataset . . . ")
        super(seq_mnist_train, self).__init__(args, train_set=True)
        

class seq_mnist_val(seq_mnist):
    def __init__(self, args):
        print("Building Testing Dataset . . . ")
        super(seq_mnist_val, self).__init__(args, train_set=False)
        