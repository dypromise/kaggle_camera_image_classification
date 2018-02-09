
# coding: utf-8

# In[21]:


import  xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# ## Prepare dataset

# In[2]:


file_ = open('./kaggel_data/TrainAug2.txt')
lines = file_.readlines()
labels_p1 = []
for line in lines:
    labels_p1.append(line.split(' ')[1].split('\n')[0])
labels_p2 = []
file_tmp = open('./val_aug.txt')
lines_tmp = file_tmp.readlines()
for line in lines_tmp:
    labels_p2.append(line.split(' ')[1].split('\n')[0])
Y_train = np.array(labels_p1,dtype = 'float32')
Y_val = np.array(labels_p2,dtype = 'float32')

print Y_train.shape,Y_val.shape


# In[3]:


X_1 = np.load('./ensemble_pooling_data/feature_pooling_0(inceptionresv2_184000).npy')
X_2 = np.load('./ensemble_pooling_data/feature_pooling_0（incepresnetv2_model_aug_164000）.npy')
X_train = np.concatenate((X_1,X_2),axis = 1)
X_val_p1 = np.load('./ensemble_pooling_data/feature_val_pooling_0(inceptionresv2_184000).npy')
X_val_p2 = np.load('./ensemble_pooling_data/feature_val_pooling_0（incepresnetv2_model_aug_164000).npy')
X_val = np.concatenate((X_val_p1,X_val_p2),axis = 1)
print X_train.shape,X_val.shape


# ## Train model

# In[4]:


seed = 7
test_size = 0.1


# In[5]:


def modelfit(alg, X_, Y_,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_, label=Y_)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='mlogloss', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(X_,Y_,eval_metric='mlogloss')

    #Predict training set:
    dtrain_predictions = alg.predict(X_)
    dtrain_predprob = alg.predict_proba(X_)[:,1]

    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" %accuracy_score(Y_, dtrain_predictions)
  


# In[6]:


xgb1 = XGBClassifier(
    learning_rate =0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'multi:softmax',
    num_class=10,
    nthread=4,
    scale_pos_weight=1,
    seed=27)
modelfit(xgb1, X_train, Y_train)


# ## Test

# In[22]:


val_pred = xgb1.predict(X_val)
print accuracy_score(Y_val,val_pred)
print precision_score(Y_val, val_pred,average=None)
print recall_score(Y_val,val_pred,average=None)


# In[15]:


Y_val


# In[16]:


val_pred


# In[9]:


X_test_1 = np.load('./ensemble_pooling_data/feature_test_pooling_0(inceptionresv2_184000).npy')
X_test_2 = np.load('./ensemble_pooling_data/feature_test_pooling_0（incepresnetv2_model_aug_164000).npy')
X_to_pred = np.concatenate((X_test_1,X_test_2),axis = 1)
X_to_pred.shape


# In[10]:


Pred_submission = xgb1.predict(X_to_pred)


# In[11]:


Pred_submission


# In[12]:


np.save("./res_xgb.npy",Pred_submission)

