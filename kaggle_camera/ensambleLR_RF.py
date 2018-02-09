
# coding: utf-8

# In[13]:


from sklearn import datasets, neighbors, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier


# In[14]:


import os
def load_train_data(dir_name):
    data_train=[]
    for i in range(150):
        npy_cur = np.load(os.path.join(dir_name,'prob_train_%d.npy'%i))
        if(len(data_train)>0):
            data_train = np.concatenate((data_train,npy_cur),axis = 0)
        else:
            data_train = npy_cur
    return data_train
    
def load_val_data(dir_name):
    data_val=[]
    for i in range(5):
        npy_cur = np.load(os.path.join(dir_name,'prob_val_%d.npy'%i))
        if(len(data_val)>0):
            data_val = np.concatenate((data_val,npy_cur),axis = 0)
        else:
            data_val = npy_cur
    return data_val

def load_labels(file_name,one_hot = True):
    file_ = open(file_name,'rb')
    lines = file_.readlines()
    labels = []
    for line in lines:
        labels.append(line.split(' ')[1].split('\n')[0])
    if(one_hot):
        lb = LabelBinarizer()
        labels = np.array(labels,dtype = 'float32')
        lb.fit(labels)
        y_ = lb.transform(labels)
    else:
        y_ = np.array(labels,dtype = 'float32')
    return y_


# In[58]:


# x_train_1 = load_train_data('/home/dingyang/resnet50_aug2-2_190000/train_prob')
x_train_2 = load_train_data('/home/dingyang/inceptionResnetV2_aug2-1_200000/train_prob')
x_train_3 = load_train_data('/home/dingyang/se_aug2-1_200000/train_prob')
# x_train_4 = load_train_data('./resnet_aug2-2_208000/train_prob')
y_train = load_labels('/home/dingyang/TrainAug2_2.txt')

# x_val_1 = load_val_data('/home/dingyang/resnet50_aug2-2_190000/val_prob')
x_val_2 = load_val_data('/home/dingyang/inceptionResnetV2_aug2-1_200000/val_prob')
x_val_3 = load_val_data('/home/dingyang/se_aug2-1_200000/val_prob')
# x_val_4 = load_val_data('./resnet_aug2-2_208000/val_prob')
y_val = load_labels('/home/dingyang/val_aug0.txt')


# x_test_1 = np.load('./test/resnet50_aug2-2_190000.npy')
x_test_2 = np.load('./test/incepaug2-1_200000.npy')
x_test_3 = np.load('./test/seaug2-1_200000.npy')
# x_test_4 = np.load('./test/prob_1(resnet101_aug2-2_208000_multi).npy')


# In[59]:


x_ = np.array([x_train_2,x_train_3])
x_train = np.transpose(x_,(2,1,0))
print x_train.shape,y_train.shape
x_val = np.transpose(np.array([x_val_2,x_val_3]),(2,1,0))
print x_val.shape,y_val.shape

x_test = np.transpose(np.array([x_test_2,x_test_3]),(2,1,0))
print x_test.shape


# ## Train:

# In[60]:


def single_model_fit(x_,y_):
    """
    return the trained model.
    """
#     model_ = linear_model.LogisticRegression()
    model_ = RandomForestClassifier(n_estimators=100,max_depth=2,class_weight='balanced', random_state=0,max_features=None)
    model_.fit(x_,y_)
    
    print "feature importance: \n"
    print model_.feature_importances_
    return model_


def single_model_predict(x_,model):
    """
    output dimention: (num_samples,)
    """
    prob = model.predict_proba(x_)
    return prob[:,1] # return the prob of "1"


def single_model_test(x_test,y_test,model):
    pred = model.predict(x_test)
    print accuracy_score(pred,y_test)
    
    
def fit_models(x_,y_,num_classes):
    models = []
    for i in range(num_classes):
        model_ = single_model_fit(x_[i],y_[i])
        models.append(copy.deepcopy(model_))
    return models


def get_prediction(x_,models_,num_classes):
    preds = []
    for i in range(num_classes):
        preds.append(single_model_predict(x_[i],models_[i]))
    preds = np.transpose(np.array(preds,dtype = 'float32'))
    
    y_output = []
    for i in range(len(preds)):
        y_output.append(np.argmax(preds[i]))
    return y_output


def test_models(x_test,y_test,models,num_classes):
    """
    Dimentions:
        x_test: (num_classes, num_samples, num_models), each class corresbonds to one LR model.
        y_test: (num_samples, num_classes)
    """
    probs = []
    for i in range(num_classes):
        probs.append(single_model_predict(x_test[i],models[i])) # num_classes * num_samples
    
    output = []
    ground_t = []
    
    probs = np.transpose(np.array(probs,dtype = 'float32')) # num_samples * num_classes
    for i in range(len(probs)):
        output.append(np.argmax(probs[i]))
        ground_t.append(np.argmax(y_test[i]))
#     print output,ground_t
    return accuracy_score(output,ground_t)




def naive_stack_method(x_val,y_val):
    x_ = np.transpose(x_val,(1,0,2))
    x_mean = np.mean(x_,axis = -1)
    
    output = []
    ground_t = []
    
    for i in range(len(x_mean)):
        output.append(np.argmax(x_mean[i]))
        ground_t.append(np.argmax(y_val[i]))
    return accuracy_score(output,ground_t)


def naive_single_model(x_val,y_val):
   
    x_mean = x_val
    
    output = []
    ground_t = []
    
    for i in range(len(x_mean)):
        output.append(np.argmax(x_mean[i]))
        ground_t.append(np.argmax(y_val[i]))
    return accuracy_score(output,ground_t)
    


# In[61]:


import copy
num_classes = x_train.shape[0]
preds = []
models = fit_models(x_train,np.transpose(y_train),num_classes)


# ## Test 

# In[62]:


print test_models(x_val,y_val,models,num_classes)
print naive_stack_method(x_val,y_val)
print naive_single_model(np.zeros((4950,10),dtype = 'float32'),y_val)

print naive_single_model(x_val_2,y_val),naive_single_model(x_val_3,y_val)
np.transpose(x_val,(1,0,2)).shape


# In[54]:


output = get_prediction(x_test,models,num_classes)


# In[55]:


output


# In[56]:


file_ = open('./test0.txt')
imgs  = file_.readlines()
imgs = [x.split(' ')[0] for x in imgs]
imgs


# In[57]:


import csv

classes = ['Motorola-X', 'Motorola-Nexus-6', 'Samsung-Galaxy-S4', 'Samsung-Galaxy-Note3', 'LG-Nexus-5x',            'iPhone-4s', 'Motorola-Droid-Maxx', 'HTC-1-M7', 'Sony-NEX-7', 'iPhone-6']
csvfile = open("result.csv", "w")
fileheader = ["fname", "camera"]
writer = csv.writer(csvfile)
writer.writerow(fileheader)

for i in range(len(output)):
    context = []
    context.append(imgs[i])
    context.append(classes[output[i]])
    writer.writerow(context)
csvfile.close()

