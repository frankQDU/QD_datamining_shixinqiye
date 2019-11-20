#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cbt
import category_encoders as ce
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split


# In[2]:


os.listdir('../data/')


# In[3]:


submission = pd.read_csv('../data/submission.csv')
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
train_label = pd.read_csv('../data/train_label.csv')


# In[4]:


feature = ['ID','企业所得税与增值税之比','增值税', '企业所得税']


# In[5]:


train = train.merge(train_label,on='ID',how = 'left')
data = train.append(test)


# In[6]:


train[['增值税', '企业所得税']].isnull().sum()/train.shape[0]


# In[7]:


data[['增值税', '企业所得税']].fillna(-999,inplace=True)


# In[8]:


data['企业所得税与增值税之比'] = train['增值税']/train['企业所得税']


# In[9]:


tr_index = ~data['Label'].isnull()
train = data[tr_index].reset_index(drop=True)
y = data[tr_index]['Label'].reset_index(drop=True).astype(int)
test = data[~tr_index].reset_index(drop=True)
print(train.shape,test.shape)


# In[10]:


from sklearn.metrics import roc_auc_score

def lgb_roc_auc_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) 
    return 'lgb_roc_auc_score', roc_auc_score(y_true, y_hat), True

lgb_paras = {'objective': 'binary',
             'learning_rate': 0.1 ,
             'max_depth': 6 ,
             'feature_fraction': 0.8, 
             'bagging_fraction' : 0.8,
             'num_threads':-1}


# In[11]:


fi = []
cv_score = []
test_pred = np.zeros((test.shape[0],))
skf = StratifiedKFold(n_splits=5, random_state=2019, shuffle=True)


# In[12]:


for index, (train_index, test_index) in enumerate(skf.split(train, y)):
    print(index)
    train_x, test_x, train_y, test_y = train.iloc[train_index],train.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    lgb_model = lgb.train(lgb_paras,
                          train_set = lgb.Dataset(train_x[feature], train_y),
                          valid_sets=[lgb.Dataset(test_x[feature],test_y)],
                          
                          num_boost_round=1000 ,
                          feval=lgb_roc_auc_score,
                          verbose_eval=50
                          )

    

    y_val = lgb_model.predict(test_x[feature])
    
    print( roc_auc_score( test_y , np.round( y_val) ) )
    
    
    cv_score.append(roc_auc_score(test_y,np.round(y_val)))
    
    print(cv_score[index])
    
    test_pred += lgb_model.predict(test[feature]) / 5


# In[13]:


test_pred


# In[15]:


submission['Label'] = test_pred


# In[17]:


submission.to_csv('submission.csv',index=False)


# In[ ]:





# In[ ]:




