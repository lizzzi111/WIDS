#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h2o
from h2o.automl import H2OAutoML


# In[2]:


###############################
#                             #
#        ENCODE FACTORS       #
#                             #
###############################

# performs label encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
def label_encoding(df_train, df_valid, df_test):
    
    factors = df_train.select_dtypes('object').columns
    
    lbl = LabelEncoder()

    for f in factors:        
        lbl.fit(list(df_train[f].values) + list(df_valid[f].values) + list(df_test[f].values))
        df_train[f] = lbl.transform(list(df_train[f].values))
        df_valid[f] = lbl.transform(list(df_valid[f].values))
        df_test[f]  = lbl.transform(list(df_test[f].values))

    return df_train, df_valid, df_test

from sklearn import base
class KFoldTargetEncoderTrain(base.BaseEstimator,
                               base.TransformerMixin):
    def __init__(self,colnames,targetName,
                  n_fold=5, verbosity=True,
                  discardOriginal_col=False):
        self.colnames = colnames
        self.targetName = targetName
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        assert(type(self.targetName) == str)
        assert(type(self.colnames) == str)
        assert(self.colnames in X.columns)
        assert(self.targetName in X.columns)
        mean_of_target = X[self.targetName].mean()
        kf = KFold(n_splits = self.n_fold,
                   shuffle = False, random_state=2019)
        col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'
        X[col_mean_name] = np.nan
        for tr_ind, val_ind in kf.split(X):
            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
            X.loc[X.index[val_ind], col_mean_name] =  X_val[self.colnames].map(X_tr.groupby(self.colnames)[self.targetName].mean())
            X[col_mean_name].fillna(mean_of_target, inplace = True)
        if self.verbosity:
            encoded_feature = X[col_mean_name].values
            print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,self.targetName, np.corrcoef(X[self.targetName].values, encoded_feature)[0][1]))
        if self.discardOriginal_col:
            X = X.drop(self.targetName, axis=1)
        return X

class KFoldTargetEncoderTest(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self,train,colNames,encodedName):
        
        self.train = train
        self.colNames = colNames
        self.encodedName = encodedName
        
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        mean =  self.train[[self.colNames,
                self.encodedName]].groupby(
                                self.colNames).mean().reset_index() 
        
        dd = {}
        for index, row in mean.iterrows():
            dd[row[self.colNames]] = row[self.encodedName]
        X[self.encodedName] = X[self.colNames]
        X = X.replace({self.encodedName: dd})
        return X


# In[4]:


print('Import Packages')
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('dark_background')

import os
import time
import datetime
import random
import multiprocessing
import pickle

import scipy.stats

import gc
gc.enable()

import warnings
warnings.filterwarnings('ignore')


from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import log_loss, roc_auc_score, confusion_matrix

import lightgbm as lgb


# In[5]:


############ RANDOMNESS

# seed function
def seed_everything(seed = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
# set seed
seed = 42
seed_everything(seed)


# ### IMPORT

# In[11]:


############ DATA IMPORT

# id data
train = pd.read_csv('../raw/training.csv')
test  = pd.read_csv('../raw/unlabeled.csv')

train = train[-train['hospital_death'].isnull()]

#train = train.head(100)
# check dimensions
print(train.shape)
print(test.shape)


# In[12]:
print('Preparing the data')

train['NAs'] = train.isnull().sum(axis=1)
test['NAs']  = test.isnull().sum(axis=1)


# In[13]:


train['hospital_id'] = train['hospital_id'].astype('object')
test['hospital_id']  = test['hospital_id'].astype('object')

train['icu_id'] = train['icu_id'].astype('object')
test['icu_id']  = test['icu_id'].astype('object')


# In[14]:


for feature in train.select_dtypes('object').columns:    
    targetc = KFoldTargetEncoderTrain(feature,'hospital_death',n_fold=10)
    train = targetc.fit_transform(train)

    test_targetc = KFoldTargetEncoderTest(train,
                                          feature,
                                          f'{feature}_Kfold_Target_Enc')
    test = test_targetc.fit_transform(test)


# In[15]:


y     = train['hospital_death']
train = train.drop('hospital_death', axis=1)


# In[16]:


############ FEAUTERS

# drop bad features
excluded_feats = ['encounter_id', 'patient_id', 'readmission_status', 'hospital_id', 'icu_id']
excluded_feats.extend(list(train.select_dtypes('object').columns))


# In[17]:


train['apache_prob_prod'] = train['apache_4a_hospital_death_prob'] * train[ 'apache_4a_icu_death_prob']
test['apache_prob_prod'] = test['apache_4a_hospital_death_prob'] * train[ 'apache_4a_icu_death_prob']


# In[18]:


features = [f for f in train.columns if f not in excluded_feats]
#features = ['apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob']
print(train[features].shape)


# In[19]:


############ PARAMETERS

# cores
cores = -1
# cross-validation
num_folds = 10
shuffle   = True

# number of trees
max_rounds = 10000
stopping   = 200
verbose    = 250

# LGB parameters
lgb_params = {
    'boosting_type':     'gbdt',
    'objective':         'binary',
    'metric':            'auc',
    'bagging_fraction':  0.9,
    'feature_fraction':  0.9,
    'lambda_l1':         0.1,
    'lambda_l2':         0.1,
    'min_split_gain':    0.1,
    'min_child_weight':  0,
    'min_child_samples': 10,
    'silent':            True,
    'verbosity':         -1,
    'learning_rate':     0.01,
    'max_depth':         5,
    'num_leaves':        64,
    'scale_pos_weight':  1,
    'n_estimators':      max_rounds,
    'nthread' :          cores,
    'random_state':      seed,
    #"device" : "gpu"
}

# data partitinoing
folds = StratifiedKFold(n_splits = num_folds, random_state = seed, shuffle = shuffle)
#folds = GroupKFold(n_splits = num_folds)
#folds = model_selection.TimeSeriesSplit(n_splits = 10)

# SMOTE settings
#from imblearn.over_sampling import SMOTE
#sm = SMOTE(random_state = seed, n_jobs = cores, sampling_strategy = 0.05)


# In[20]:



############ PLACEHOLDERS

# placeholders
clfs = []
importances = pd.DataFrame()

# predictions
preds_test   = np.zeros(test.shape[0])
preds_oof    = np.zeros(train.shape[0])


# In[21]:

x_ = list(train[features].columns)
y_ = 'hospital_death'

h2o.init(nthreads=20)


# In[ ]:

print('training')
############ CROSS-VALIDATION LOOP
cv_start  = time.time()
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train, y)):
    
    for column in train.select_dtypes('object').columns:
        train[column] = train[column].fillna('')
        test[column] = test[column].fillna('')
    
    X = pd.concat([train, test], axis=0)
    X = h2o.H2OFrame(X)
    
    train = X[~X["hospital_death"].isna(),:]
    test  = X[X["hospital_death"].isna(),:]
    
    # data partitioning
    
    trn_x, trn_y = train[trn_idx,features], y.iloc[trn_idx]
    val_x, val_y = train[val_idx,features], y.iloc[val_idx]
    test_x       = test[features]
            
    # label encoding
    #trn_x, val_x, test_x = label_encoding(trn_x, val_x, test_x)
       
    ## add noise to train to reduce overfitting
    #trn_x += np.random.normal(0, 0.01, trn_x.shape)
    
    trn_x['hospital_death'] = trn_x['hospital_death'].asfactor()
    val_x['hospital_death'] = val_x['hospital_death'].asfactor()
        
    #print(trn_x.types)
    #print(val_x.types)
    
    # print data dimensions
    print('Data shape:', trn_x.shape, val_x.shape)
    #print('Data shape:', trn_y.shape, val_y.shape)    
    # train lightGBM
    aml = H2OAutoML(max_models = 10, seed = seed, stopping_metric = 'auc')
    aml.train(x = x_, y = y_, training_frame = trn_x)
    
    print(aml.leader.predict(val_x)[:,'p1'].as_data_frame()['p1'])
    # save predictions
    preds_oof[val_idx] = aml.leader.predict(val_x)[:,'p1'].as_data_frame()['p1']
    preds_test        += aml.leader.predict(test_x)[:,'p1'].as_data_frame()['p1'] / folds.n_splits 

    
    # print performance
    print('--------------------------------')
    print('FOLD%2d: AUC = %.6f' % (n_fold + 1, roc_auc_score(y[val_idx], preds_oof[val_idx])))
    print('--------------------------------')
    print('')
        
    # clear memory
    del trn_x, trn_y, val_x, val_y
    gc.collect()
    
    
# print overall performance    
cv_perf = roc_auc_score(y, preds_oof)
print('--------------------------------')
print('- OOF AUC = %.6f' % cv_perf)
print('- CV TIME = {:.2f} min'.format((time.time() - cv_start) / 60))
print('--------------------------------')


# ### EVALUATION

# In[ ]:


############ RECHECK PERFORMANCE  

# check performance
print(np.round(roc_auc_score(y, preds_oof), 5))


############ TRACK RESULTS


# In[ ]:


# SUBMISSION

# In[ ]:


# file name
model = 'h2o_v1'
perf  = str(round(cv_perf, 6))[2:7]
name  = model + '_' + perf
name


# In[ ]:


# export OOF preds
oof = pd.DataFrame({'encounter_id': train['encounter_id'], 'hospital_death': preds_oof})
oof.to_csv('./oof_preds/' + str(name) + '.csv', index = False)
oof.head()


# In[ ]:



# export submission
sub = pd.DataFrame({'encounter_id': test['encounter_id'], 'hospital_death': preds_test})
sub.to_csv('./submissions/' + str(name) + '.csv', index = False)
sub.head()

