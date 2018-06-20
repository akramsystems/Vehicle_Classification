#Libraries

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(18, 17)
        self.l2 = nn.Linear(17, 16)
        self.l3 = nn.Linear(16, 13)
        self.l4 = nn.Linear(13, 12)
        self.l5 = nn.Linear(12, 10)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)  		#	no Activation Needed	

class Training_Dataset(Dataset):
    #initialize your data, download, etc.
    def __init__(self):
        testdata = pd.read_csv('data/data_cars_train.txt',header=None, sep=r"\s*", engine='python')
        x_data_test = testdata.drop(columns=[18])
        x_data_test = x_data_test.values
        y_data_test = testdata[18]
        min_max_scaler = preprocessing.MinMaxScaler() 
        x_data_test = min_max_scaler.fit_transform(x_data_test)
        x_data_test = np.asarray(x_data_test)
        le = preprocessing.LabelEncoder()
        le.fit(y_data_test)
        y_data_test = le.transform(y_data_test)
        
        
        self.len = testdata.shape[0]
        
        #create torch tensors
        self.x_data = torch.from_numpy(x_data_test).float()
        self.y_data = torch.from_numpy(y_data_test)
    
    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class Testing_Dataset(Dataset):
    #initialize your data, download, etc.
    def __init__(self):
        testdata = pd.read_csv('data/data_cars_test.txt',header=None, sep=r"\s*", engine='python')
        x_data_test = testdata.drop(columns=[18])
        x_data_test = x_data_test.values
        y_data_test = testdata[18]
        min_max_scaler = preprocessing.MinMaxScaler() 
        x_data_test = min_max_scaler.fit_transform(x_data_test)
        x_data_test = np.asarray(x_data_test)
        le = preprocessing.LabelEncoder()
        le.fit(y_data_test)
        y_data_test = le.transform(y_data_test)
        
        
        self.len = testdata.shape[0]
        
        #create torch tensors
        self.x_data = torch.from_numpy(x_data_test).float()
        self.y_data = torch.from_numpy(y_data_test)
    
    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len



def train(epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data,target = data
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).data[0]
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


#	Set the Model
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.5)


# 	Set Dataset
train_dataset = Training_Dataset()
train_loader = DataLoader(dataset=train_dataset,
						  batch_size=1,
						  shuffle=True,
						  num_workers=2)
test_dataset = Testing_Dataset()
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=1,
                          shuffle=True,
                          num_workers=2)


#	Training & Testing Implementation
for epoch in range(1000):
	train(epoch)
	test()



























