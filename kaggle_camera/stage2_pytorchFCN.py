
# coding: utf-8

# # Warming up

# In[9]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[10]:


dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
torch.cuda.is_available()
torch.cuda.current_stream()
torch.cuda.device_count()


# In[11]:


from torch.autograd import Variable
class SEScale(nn.Module):
    def __init__(self, channel, reduction):
        super(SEScale, self).__init__()
        self.fc1 = nn.Linear(channel, reduction).double()
        self.fc2 = nn.Linear(reduction, channel).double()
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x
    
class FcNet3(nn.Module):
    def __init__(self, in_shape = 2048, num_classes = 10):
        super(FcNet3, self).__init__()
        in_channels = in_shape
        
        self.scale = SEScale(in_channels, in_channels//2)
        self.linear1 = nn.Linear(in_channels,512).double()
        self.relu1 = nn.PReLU().double()
        
        self.fc = nn.Linear(512, num_classes).double()
        
    def forward(self, x):
              
        x = F.dropout(x, p = 0.7,training=self.training)
        x = self.scale(x)*x
        
        x = self.linear1(x)
        x = self.relu1(x)
        x = F.dropout(x,p=0.7,training=self.training)
        
        x = self.fc(x)
        return x


from torch.utils.data import Dataset, DataLoader
class FeatureDataset(Dataset):
    def __init__(self, X_,Y_):
        self.X_= X_
        self.Y_ = Y_

    def __len__(self):
        return len(self.X_)

    def __getitem__(self, idx):
        sample = (self.X_[idx],self.Y_[idx])
        return sample


# In[15]:


val_accs = np.load('code/val_acc.npy')
import matplotlib.pyplot as plt 
x_axis = range(len(val_accs))
plt.plot(x_axis,val_accs)
plt.show()


# In[14]:


X_test_1 = np.load('../feature_test/feature_test_inceptionResnetV2_aug2-1_200000_.npy')
X_test_2 = np.load('../feature_test/feature_test(inceptionResnetV2_aug_164000).npy')
X_test_3 = np.load('../feature_test/feature_test(SE_aug2-1_200000).npy')
X_test = np.concatenate((X_test_1,X_test_2,X_test_3),axis = 1)
print X_test_1.shape,X_test_2.shape,X_test_3.shape
print X_test.shape


# In[114]:


testset = FeatureDataset(X_test,np.array([0]*X_test.shape[0]))
testloader = torch.utils.data.DataLoader(testset, batch_size=100)

preds = []
total = 0

net = FcNet3(X_test.shape[0],10)
net.load_state_dict(torch.load('code/net_params.pkl'))


for data in testloader:
    images, _ = data
    
    outputs = net(Variable(images).cuda())
    _, predicted = torch.max(outputs.data, 1)
    preds.extend(predicted)
    
    total += images.size(0)
    
print total
np.save('./res_fc.npy',np.array(preds))


# In[52]:


len(preds)

