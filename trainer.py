import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import warpctc_pytorch as wp
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import BiLSTM
from decoder import seq_mnist_decoder
from data import seq_mnist_train, seq_mnist_val


class Seq_MNIST_Trainer():

    def __init__(self, trainer_params, args):
        self.args = args
        self.trainer_params = trainer_params
        
        random.seed(trainer_params.random_seed)
        torch.manual_seed(trainer_params.random_seed)
        if args.cuda:
                torch.cuda.manual_seed_all(trainer_params.random_seed)

        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}    
        self.train_data = seq_mnist_train(trainer_params)
        self.val_data = seq_mnist_val(trainer_params) 
        self.train_loader = DataLoader(self.train_data, batch_size=trainer_params.batch_size, shuffle=True, **kwargs)
        self.val_loader = DataLoader(self.val_data, batch_size=trainer_params.test_batch_size, shuffle=True, **kwargs)
        self.starting_epoch = 1
        self.prev_loss = 10000
    
        self.model = BiLSTM(trainer_params) 
        self.criterion = wp.CTCLoss(size_average=True)
        self.labels = [i for i in range(trainer_params.num_classes-1)]
        self.decoder = seq_mnist_decoder(labels=self.labels)

        if args.resume or args.eval or args.export:
            print("Loading model from {}".format(args.save_path))
            package = torch.load(args.save_path, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(package['state_dict'])

        if args.cuda:
            torch.cuda.set_device(args.gpus)
            self.model = self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=trainer_params.lr)

        if args.resume:
            self.optimizer.load_state_dict(package['optim_dict']) 
            self.starting_epoch = package['starting_epoch']
            self.prev_loss = package['prev_loss']
            if args.cuda:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

        if args.init_bn_fc_fusion:
            if not trainer_params.prefused_bn_fc:
                self.model.batch_norm_fc.init_fusion()
                self.trainer_params.prefused_bn_fc = True
            else:
                raise Exception("BN and FC are already fused.")

    def serialize(self, model, trainer_params, optimizer, starting_epoch, prev_loss):
        package = {'state_dict': model.state_dict(),
            'trainer_params': trainer_params,
            'optim_dict' : optimizer.state_dict(),
            'starting_epoch' : starting_epoch,
            'prev_loss': prev_loss
        }
        return package

    def save_model(self, epoch, loss_value):
        print("Model saved at: {}\n".format(self.args.save_path))
        self.prev_loss = loss_value
        torch.save(self.serialize(model=self.model, trainer_params=self.trainer_params, 
            optimizer=self.optimizer, starting_epoch=epoch + 1, prev_loss=self.prev_loss), self.args.save_path)


    def train(self, epoch):
        self.model.train()
        for i, (item) in enumerate(self.train_loader):
            data, labels, output_len, lab_len = item
            
            data = Variable(data.transpose(1,0), requires_grad=False)
            labels = Variable(labels.view(-1), requires_grad=False)
            output_len = Variable(output_len.view(-1), requires_grad=False)
            lab_len = Variable(lab_len.view(-1), requires_grad=False)
            
            if self.args.cuda:
                data = data.cuda()
 
            output = self.model(data)

            # print("Input = ", data.shape)
            # print("model output (x) = ", output)
            # print("GTs (y) = ", labels.type())
            # print("model output len (xs) = ", output_len.type())
            # print("GTs len (ys) = ", lab_len.type())
            # exit(0)

            loss = self.criterion(output, labels, output_len, lab_len)
            loss_value = loss.data[0]
            print("Loss value for epoch = {}/{} and batch {}/{} is = {:.4f}".format(epoch, 
                self.trainer_params.epochs, (i+1)*self.trainer_params.batch_size, len(self.train_data) , loss_value))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.args.cuda:
                torch.cuda.synchronize()                   

    def test(self, epoch=0, save_model_flag=False):
        self.model.eval()
        loss_value = 0
        for i, (item) in enumerate(self.val_loader):           
            data, labels, output_len, lab_len = item
            
            data = Variable(data.transpose(1,0), requires_grad=False)
            labels = Variable(labels.view(-1), requires_grad=False)
            output_len = Variable(output_len.view(-1), requires_grad=False)
            lab_len = Variable(lab_len.view(-1), requires_grad=False)
            
            if self.args.cuda:
                data = data.cuda()

            output = self.model(data)
            
            # print("Input = ", data)
            # print("model output (x) = ", output.shape)
            # print("model output (x) = ", output)        
            # print("Label = ", labels)
            # print("model output len (xs) = ", output_len)
            # print("GTs len (ys) = ", lab_len)
            
            index = random.randint(0,self.trainer_params.test_batch_size-1)      
            label = labels[index*self.trainer_params.word_size:(index+1)*self.trainer_params.word_size].data.numpy()
            label = label-1
            prediction = self.decoder.decode(output[:,index,:], output_len[index], lab_len[index])
            accuracy = self.decoder.hit(prediction, label)

            print("Sample Label      = {}".format(self.decoder.to_string(label))) 
            print("Sample Prediction = {}".format(self.decoder.to_string(prediction)))
            print("Accuracy on Sample = {:.2f}%\n\n".format(accuracy))

            loss = self.criterion(output, labels, output_len, lab_len)
            loss_value += loss.data.numpy()

        loss_value /= (len(self.val_data)//self.trainer_params.test_batch_size)
        print("Average Loss Value for Val Data is = {:.4f}\n".format(float(loss_value)))
        
        if loss_value < self.prev_loss and save_model_flag:
            self.save_model(epoch, loss_value)

    def eval_model(self):
        self.test()


    def train_model(self):
        for epoch in range(self.starting_epoch, self.trainer_params.epochs + 1):
            self.train(epoch)
            self.test(epoch=epoch, save_model_flag=True)
            if epoch%20==0:
                self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr']*0.98

    def export_model(self, simd_factor, pe):
        self.model.eval()
        self.model.export('r_model_fw_bw.hpp', simd_factor, pe)

    def export_image(self, idx=100):
        img, label = self.val_data.images[:,idx,:], self.val_data.labels[0][idx]
        img = img.transpose(1, 0)
        label -= 1
        label = self.decoder.to_string(label)
        
        from PIL import Image
        from matplotlib import cm

        im = Image.fromarray(np.uint8(cm.gist_earth(img)*255))
        im.save('test_image.png')
        img = img.transpose(1, 0)

        img = np.reshape(img, (-1, 1))
        np.savetxt("test_image.txt", img, fmt='%.10f')
        f = open('test_image_gt.txt','w')
        f.write(label)
        f.close()
        print("Exported image with label = {}".format(label))