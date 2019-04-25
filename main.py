import os
import argparse
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

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Quantized BiLSTM Sequential MNIST Example')
parser.add_argument('--batch-size', type=int, default=50, metavar='N', help='input batch size for training (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=50, metavar='N', help='input batch size for testing (default: 50)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--gpus', default=0, help='gpus used for training - e.g 0,1,3')
parser.add_argument('--resume', default=False, action='store_true', help='Perform only evaluation on val dataset.')
parser.add_argument('--eval', default=False, action='store_true', help='perform evaluation of trained model')
parser.add_argument('--export', default=False, action='store_true', help='perform weights export as npz of trained model')
parser.add_argument('--word_size', type=int, default=8, metavar='N', help='word size i.e. no. of digits in a seq (default: 8)')
parser.add_argument('--input_size', default=28, type=int, help='Input size')
parser.add_argument('--num_classes', default=10+1, type=int, help='Number of classes + blank token.')
parser.add_argument('--num_units', default=128, type=int, help='Number of LSTM units.')
parser.add_argument('--seq_len', default=28, type=int, help='Length of a single digit')
parser.add_argument('--num_layers', default=1, type=int, help='Number of BiLSTM layers.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

save_path = "results/model.tar"

if not os.path.exists("results/"):
    os.mkdir("results/")

prev_loss = 10000

def serialize(model, optimizer, epoch, prev_loss):
    package = {'state_dict': model.state_dict(),
        'optim_dict' : optimizer.state_dict(),
        'epoch' : epoch,
        'prev_loss': prev_loss
    }
    return package

def train(epoch):
    global prev_loss
    model.train()
    for i, (item) in enumerate(train_loader):
        data, labels, output_len, lab_len = item
        data = Variable(data.transpose(1,0), requires_grad=False)
        labels = Variable(labels.view(-1), requires_grad=False)
        output_len = Variable(output_len.view(-1), requires_grad=False)
        lab_len = Variable(lab_len.view(-1), requires_grad=False)
        
        if args.cuda:
            data.cuda()
        output = model(data)

        # print("Input = ", data.shape)
        # print("model output (x) = ", output)
        # print("GTs (y) = ", labels.type())
        # print("model output len (xs) = ", output_len.type())
        # print("GTs len (ys) = ", lab_len.type())
        # exit(0)

        loss = criterion(output, labels, output_len, lab_len)
        loss_value = loss.data[0]
        print("Loss value for epoch = {}/{} and batch {}/{} is = {:.4f}".format(epoch, 
            args.epochs, (i+1)*args.batch_size, len(train_data) , loss_value))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.cuda:
            torch.cuda.synchronize()            

def test(epoch=0, save_model=False):
    model.eval()
    loss_value = 0
    global prev_loss
    for i, (item) in enumerate(val_loader):
        
        data, labels, output_len, lab_len = item
        
        data = Variable(data.transpose(1,0), requires_grad=False)
        labels = Variable(labels.view(-1), requires_grad=False)
        output_len = Variable(output_len.view(-1), requires_grad=False)
        lab_len = Variable(lab_len.view(-1), requires_grad=False)
        
        if args.cuda:
            data.cuda()
        output = model(data)
        
        # print("Input = ", data)
        # print("model output (x) = ", output.shape)
        # print("model output (x) = ", output)        
        # print("Label = ", labels)
        # print("model output len (xs) = ", output_len)
        # print("GTs len (ys) = ", lab_len)
        
        index = random.randint(0,args.test_batch_size-1)      
        label = labels[index*args.word_size:(index+1)*args.word_size].detach().numpy()
        prediction = decoder.decode(output[:,index,:], output_len[index], lab_len[index])
        accuracy = decoder.hit(prediction, label)

        label_str = ''
        pred_str = ''
        for i in range(lab_len[index]):
            label_str += str(label[i])
        for i in range(len(prediction)):
            pred_str += str(prediction[i])

        print("Example Original  = {}".format(label_str)) 
        print("Example Predicted = {}".format(pred_str))
        print("Accuracy on Example = {:.2f}%\n\n".format(accuracy))

        loss = criterion(output, labels, output_len, lab_len)
        loss_value += loss.item()
        # print("For batch {} Loss = {}".format(i+1, loss.item()))

    loss_value /= (len(val_data)//args.test_batch_size)
    print("Average Loss Value for Val Data is = {:.4f}".format(loss_value))
    
    if loss_value < prev_loss and save_model:
        print("Model saved at: ", save_path, "\n")
        prev_loss = loss_value
        torch.save(serialize(model=model, optimizer=optimizer,epoch=epoch + 1,prev_loss=prev_loss), save_path)


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    if args.cuda:
            torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}    

    model = BiLSTM(args)
    if args.cuda:
        torch.cuda.set_device(args.gpus)
        model.cuda()
    
    criterion = wp.CTCLoss(blank=10, size_average=True, length_average=False)
    labels = [i for i in range(10)]
    decoder = seq_mnist_decoder(labels=labels, blank=10)
    
    # test model
    if args.eval:
        val_data = seq_mnist_val(args) 
        val_loader = DataLoader(val_data, batch_size=args.test_batch_size, shuffle=True, **kwargs)
        print("Loading model from {}".format(save_path))
        package = torch.load(save_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(package['state_dict'])
        test()
    
    # export
    elif args.export:
        print("Loading model from {}".format(save_path))
        package = torch.load(save_path, map_location= 'cpu')
        model.load_state_dict(package['state_dict'])
        model.export()
    
    # train model
    else:
        train_data = seq_mnist_train(args)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_data = seq_mnist_val(args) 
        val_loader = DataLoader(val_data, batch_size=args.test_batch_size, shuffle=True, **kwargs)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        if args.resume:
            print("Loading model from {}".format(save_path))
            package = torch.load(save_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(package['state_dict'])
            optimizer.load_state_dict(package['optim_dict']) 
            prev_loss = package['prev_loss']
            starting_epoch = package['epoch']
            test()
        else:
            starting_epoch = 1
        for epoch in range(starting_epoch, args.epochs + 1):
            train(epoch)
            test(epoch, save_model=True)
            if epoch%10==0:
                optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.98

