from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
import numpy as np
from torchvision import datasets, transforms
import dpsgd


        

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1,16, kernel_size=(8, 8),stride=2)),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=1)),
            ('c3', nn.Conv2d(16,32, kernel_size=(4, 4), stride=2)),
            ('relu2', nn.ReLU()),
            ('s3', nn.MaxPool2d(kernel_size=(2, 2), stride=1)),
            
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(32*9, 30)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(30, 10)),
            ('softmax', nn.Softmax()),
        ]))


    def forward(self, x):
        '''if self.training:
            norm = x.pow(2).view(x.size()[0],-1).sum(dim=1).pow(0.5).view(-1,1,1,1)
            #print (norm.size())
            CC = 1.0
            #print (norm.mean())
            x=x*CC / torch.max(norm,CC*torch.ones_like(norm))
            rand= torch.zeros_like(x)
            rand.normal_(mean=0,std=0.5*CC)
            x+=rand'''
        output = self.convnet(x)
        output = output.view(x.size(0), -1)
        
        
        output = self.fc(output)
        
        return output

    
    
    
    
    

def train( model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0 
    counter = 0 
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        
        output = model(data)
        
        loss = F.cross_entropy(output, target )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        
        if batch_idx %1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss/(1.0+batch_idx)))

def test( model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    
train_batch = 1
test_batch = 1000



use_cuda = torch.cuda.is_available()



device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       
                   ])),
    batch_size=train_batch, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
     
        
                   ])),
    batch_size=test_batch , shuffle=True, **kwargs)

train_test_loader = torch.utils.data.DataLoader(
    
    datasets.MNIST('./data', train=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=test_batch , shuffle=True, **kwargs)



model = Net().to(device)


optimizer = dpsgd.DPSGD(model.parameters(),lr=0.15,batch_size=256,C=1.,noise_multiplier=1.0)


for epoch in range(0, 70):


    train( model, device, train_loader, optimizer, epoch)
    
    test( model, device, test_loader)
    test( model, device, train_test_loader)

