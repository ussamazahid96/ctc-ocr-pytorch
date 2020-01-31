import torch
import numpy as np
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from quantization.function.quantization_scheme import ActivationQuantizationScheme


class seq_mnist(Dataset):
    """docstring for seq_mnist_dataset"""
    def __init__(self, trainer_params, train_set):
        self.data = datasets.MNIST('data', train=train_set, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
        self.trainer_params = trainer_params
        self.images = []
        self.labels = []
        self.input_lengths = np.ones(1, dtype=np.int32) * (28 * self.trainer_params.word_size)
        self.label_lengths = np.ones(1, dtype=np.int32) * (self.trainer_params.word_size) 
        self.input_quantization_scheme = ActivationQuantizationScheme(trainer_params.recurrent_activation_bit_width, 
            trainer_params.recurrent_activation_quantization)
        self.build_dataset()  

    def build_dataset(self):
        imgs = []
        labels = []
        for j in range(len(self.data)//self.trainer_params.word_size): # this loop builds dataset
            img = np.zeros((self.trainer_params.input_size, self.trainer_params.word_size * 28))
            labs = np.zeros(self.trainer_params.word_size, dtype=np.int32)
            for i in range(self.trainer_params.word_size):  # this loop builds one example
                ims, labs[i] = self.data[(j*self.trainer_params.word_size)+i]
                labs[i] += 1 # because ctc assumes 0 as blank character
                ims = np.reshape(ims, (28,28))
                ims = np.pad(ims, ((2,2),(0,0)), mode='constant', constant_values=-1) 
                img[:, i*28 : (i+1)*28 ] = ims
            
            img = np.transpose(img)
            imgs.append(img)
            labels.append(labs)

        self.images = np.asarray(imgs, dtype=np.float32).transpose(1, 0, 2)
        
        if self.trainer_params.quantize_input:
            self.images = self.quantize_tensor_image(self.images)
            self.images = np.asarray(self.images)
        
        self.labels.append(labels)

    def quantize_tensor_image(self, tensor_image):
        tensor_image = Variable(torch.from_numpy(tensor_image), requires_grad=False)
        out = self.input_quantization_scheme.q_forward(tensor_image)
        return out.data

    def __len__(self):
        return self.images.shape[1]

    def __getitem__(self, index):
        return self.images[:,index,:], self.labels[0][index], self.input_lengths, self.label_lengths

class seq_mnist_train(seq_mnist):
    def __init__(self, trainer_params):
        print("Building Training Dataset . . . ")
        super(seq_mnist_train, self).__init__(trainer_params, train_set=True)
        

class seq_mnist_val(seq_mnist):
    def __init__(self, trainer_params):
        print("Building Testing Dataset . . . ")
        super(seq_mnist_val, self).__init__(trainer_params, train_set=False)
        