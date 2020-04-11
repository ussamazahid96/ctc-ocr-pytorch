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
        
        self.train_data = seq_mnist_train(trainer_params)
        self.val_data = seq_mnist_val(trainer_params) 
        
        self.train_loader = DataLoader(self.train_data, batch_size=trainer_params.batch_size, \
                                        shuffle=True, num_workers=trainer_params.num_workers)
        
        self.val_loader = DataLoader(self.val_data, batch_size=trainer_params.test_batch_size, \
                                        shuffle=False, num_workers=trainer_params.num_workers)        

        self.starting_epoch = 1
        self.prev_loss = 10000
    
        self.model = BiLSTM(trainer_params) 
        self.criterion = wp.CTCLoss(size_average=False)
        self.labels = [i for i in range(trainer_params.num_classes-1)]
        self.decoder = seq_mnist_decoder(labels=self.labels)
        self.optimizer = optim.Adam(self.model.parameters(), lr=trainer_params.lr)

        if args.cuda:
            torch.cuda.set_device(args.gpus)
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        if args.resume or args.eval or args.export:
            print("Loading model from {}".format(args.resume))
            package = torch.load(args.resume, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(package['state_dict'])
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

    def save_model(self, epoch, name):
        path = self.args.experiments + '/' + name
        print("Model saved at: {}\n".format(path))
        torch.save(self.serialize(model=self.model, trainer_params=self.trainer_params, 
            optimizer=self.optimizer, starting_epoch=epoch + 1, prev_loss=self.prev_loss), path)


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

            loss = self.criterion(output, labels, output_len, lab_len)
            loss_value = loss.data[0]
            print("Loss value for epoch = {}/{} and batch {}/{} is = {:.4f}".format(epoch, 
                self.args.epochs, (i+1)*self.trainer_params.batch_size, len(self.train_data) , loss_value))
            
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
        loss_value = loss_value[0]
        print("Average Loss Value for Val Data is = {:.4f}\n".format(float(loss_value)))
        if loss_value < self.prev_loss and save_model_flag:
            self.prev_loss = loss_value
            self.save_model(epoch, "best.tar")
        elif save_model_flag:
            self.save_model(epoch, "checkpoint.tar")

    def eval_model(self):
        self.test()

    def train_model(self):
        for epoch in range(self.starting_epoch, self.args.epochs + 1):
            self.train(epoch)
            self.test(epoch=epoch, save_model_flag=True)
            if epoch%20==0:
                self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr']*0.98

    def export_model(self, simd_factor, pe):
        self.model.eval()
        self.model.export('r_model_fw_bw.hpp', simd_factor, pe)

    def export_image(self):
        random.seed()
        idx = random.randint(0,self.val_data.images.shape[1]-1)
        # idx = 100
        img, label = self.val_data.images[:,idx,:], self.val_data.labels[0][idx]
        
        inp = torch.from_numpy(img)
        inp = inp.unsqueeze(1)
        inp = Variable(inp, requires_grad=False)        
        
        out = self.model(inp)

        out = self.decoder.decode(out, self.val_data.input_lengths, self.val_data.label_lengths)
        out = self.decoder.to_string(out)
        
        img = img.transpose(1, 0)
        label -= 1
        label = self.decoder.to_string(label)
        assert label==out
        
        from PIL import Image, ImageOps
        from matplotlib import cm
        img1 = (img+1)/2.
        im = Image.fromarray(np.uint8(cm.gist_earth(img1)*255)).convert('L')
        im = ImageOps.invert(im)
        im.save('test_image.png')
        
        img = img.transpose(1, 0)
        img = np.reshape(img, (-1, 1))
        np.savetxt("test_image.txt", img, fmt='%.10f')

        f = open('test_image_gt.txt','w')
        f.write(label)
        f.close()
        
        print("Prediction on the image = {}".format(out))
        print("Label of exported image = {}".format(label))
