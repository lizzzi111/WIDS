#!/usr/bin/env python
# coding: utf-8

# In[1]:


###############################
#                             #
#        ENCODE FACTORS       #
#                             #
###############################

# performs label encoding
def reduce_mem_usage(df, verbose = True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

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


# In[34]:


from scipy.stats.mstats import winsorize
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('dark_background')
#get_ipython().run_line_magic('matplotlib', 'inline')

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler

import lightgbm as lgb


# In[3]:


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

# In[4]:


############ DATA IMPORT

# id data
train = pd.read_csv('../raw/training.csv')
test  = pd.read_csv('../raw/unlabeled.csv')


# check dimensions
print(train.shape)
print(test.shape)

train = train[-train['hospital_death'].isnull()]


# In[5]:


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)


# In[6]:


#X = pd.concat([train, test], axis=0)


# In[7]:


# Imputation transformer for completing missing values.
X = pd.concat([train, test], axis=0)
my_imputer = SimpleImputer()
new_data = pd.DataFrame(my_imputer.fit_transform(X[X.select_dtypes('number').columns]))
#new_data.columns = X.columns
X[X.select_dtypes('number').columns] = new_data

train = train[train['hospital_death'].isnull()]
test  = test[test['hospital_death'].isnull()]

# In[8]:


train['NAs'] = train.isnull().sum(axis=1)
test['NAs']  = test.isnull().sum(axis=1)


# In[9]:


train['hospital_id'] = train['hospital_id'].astype('object')
test['hospital_id']  = test['hospital_id'].astype('object')

train['icu_id'] = train['icu_id'].astype('object')
test['icu_id']  = test['icu_id'].astype('object')


# In[10]:


for feature in train.select_dtypes('object').columns:    
    targetc = KFoldTargetEncoderTrain(feature,'hospital_death',n_fold=10)
    train = targetc.fit_transform(train)

    test_targetc = KFoldTargetEncoderTest(train,
                                          feature,
                                          f'{feature}_Kfold_Target_Enc')
    test = test_targetc.fit_transform(test)


# In[11]:


y     = train['hospital_death']
train = train.drop('hospital_death', axis=1)


# In[12]:


train['age_factor'] = 0
train.loc[train['age']<10, 'age_factor'] = 'under_10'
train.loc[((train['age']>10) & (train['age']<20)), 'age_factor'] = 'b_10_20'
train.loc[((train['age']>20) & (train['age']<35)), 'age_factor'] = 'b_20_35'
train.loc[((train['age']>35) & (train['age']<50)), 'age_factor'] = 'b_35_50'
train.loc[((train['age']>50) & (train['age']<70)), 'age_factor'] = 'b_50_70'
train.loc[train['age']>70, 'age_factor'] = 'above_70'

test['age_factor'] = 0
test.loc[test['age']<10, 'age_factor'] = 'under_10'
test.loc[((test['age']>10) & (train['age']<20)), 'age_factor'] = 'b_10_20'
test.loc[((test['age']>20) & (train['age']<35)), 'age_factor'] = 'b_20_35'
test.loc[((test['age']>35) & (train['age']<50)), 'age_factor'] = 'b_35_50'
test.loc[((test['age']>50) & (train['age']<70)), 'age_factor'] = 'b_50_70'
test.loc[test['age']>70, 'age_factor'] = 'above_70'


# In[13]:


#train['apache_2_3j'] = train['apache_2_bodysystem'] + train['apache_3j_bodysystem']
#test['apache_2_3j']  = test['apache_2_bodysystem'] + test['apache_3j_bodysystem']


# In[14]:


train['apache_prob_prod'] = train['apache_4a_hospital_death_prob'] * train[ 'apache_4a_icu_death_prob']
test['apache_prob_prod'] = test['apache_4a_hospital_death_prob'] * train[ 'apache_4a_icu_death_prob']


# In[15]:


X = pd.concat([train, test], axis=0)


# In[16]:


nulls = pd.DataFrame(train.isnull().sum(axis=0))
excluded_feats = ['encounter_id', 'patient_id', 'readmission_status', 'hospital_id', 'icu_id']
excluded_feats.extend(list(nulls[nulls[0]>70000].index))


# In[17]:


features_without_nas = list(nulls[nulls[0]==0].index)
for feat in excluded_feats:
    if feat in features_without_nas:
        features_without_nas.remove(feat)


# In[18]:


features = [f for f in train.columns if f not in excluded_feats]
#features = ['apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob']
print(train[features].shape)


# In[19]:


############ PARAMETERS

# cores
cores = 20
# cross-validation
num_folds = 10
shuffle   = True

seed = 111

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


# In[36]:


############ CROSS-VALIDATION LOOP
cv_start  = time.time()
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train, y)):

    # data partitioning
    trn_x, trn_y = train[features].iloc[trn_idx], y.iloc[trn_idx]
    val_x, val_y = train[features].iloc[val_idx], y.iloc[val_idx]
    test_x       = test[features]
    
        
    # Fill Na
    for feature in ['ethnicity', 'gender']:
      trn_x[feature]  = trn_x[feature].fillna(trn_x[feature].mode()[0])
      val_x[feature]  = val_x[feature].fillna(trn_x[feature].mode()[0])
      test_x[feature] = test_x[feature].fillna(trn_x[feature].mode()[0])

    for feature in ['hospital_admit_source', 'icu_admit_source', 'icu_stay_type', 'icu_type']:
      trn_x[feature]  = trn_x[feature].fillna('missing')
      val_x[feature]  = val_x[feature].fillna('missing')
      test_x[feature] = test_x[feature].fillna('missing')
    
    '''for feature in ['apache_2_bodysystem', 'apache_3j_bodysystem']:
      trn_x[feature]  = trn_x[feature].fillna('ffill')
      val_x[feature]  = val_x[feature].fillna('ffill')
      test_x[feature] = test_x[feature].fillna('ffill')'''

    '''for feature in trn_x.select_dtypes('number').columns:
      trn_x[feature]  = trn_x[feature].fillna(-999)
      val_x[feature]  = val_x[feature].fillna(-999)
      test_x[feature] = test_x[feature].fillna(-999)'''

    '''for feature in trn_x.select_dtypes('number').columns:
      trn_x[feature]  = trn_x[feature].fillna(trn_x.groupby(['ethnicity','age_factor','gender','hospital_admit_source', 'icu_admit_source', 'icu_stay_type', 'icu_type'])[feature].transform('mean'))
      val_x[feature]  = val_x[feature].fillna(trn_x.groupby(['ethnicity','age_factor','gender','hospital_admit_source', 'icu_admit_source', 'icu_stay_type', 'icu_type'])[feature].transform('mean'))
      test_x[feature] = test_x[feature].fillna(trn_x.groupby(['ethnicity','age_factor','gender','hospital_admit_source', 'icu_admit_source', 'icu_stay_type', 'icu_type'])[feature].transform('mean'))
    '''

    for column in trn_x.select_dtypes('object').columns:
        trn_x[column] = trn_x[column].fillna('')
        val_x[column] = val_x[column].fillna('')
        test_x[column] = test_x[column].fillna('')
        
    '''for feature in ['d1_diasbp_invasive_max', 'd1_diasbp_invasive_min',
                   'd1_mbp_invasive_max', 'd1_mbp_invasive_min',
                   'd1_sysbp_invasive_max', 'd1_sysbp_invasive_min',
                   'd1_inr_max', 'd1_inr_min', 'h1_inr_max', 'h1_inr_min']:
      trn_x[feature]  = trn_x[feature].fillna(trn_x[feature].mode()[0])
      val_x[feature]  = val_x[feature].fillna(trn_x[feature].mode()[0])
      test_x[feature] = test_x[feature].fillna(trn_x[feature].mode()[0])'''
        
    # label encoding
    trn_x, val_x, test_x = label_encoding(trn_x, val_x, test_x)
    
    #ros = RandomOverSampler(random_state=seed)
    #trn_x, trn_y = ros.fit_resample(trn_x, trn_y)
    
    #print(trn_x.shape, ' ', trn_y.shape)
    # augment training data with SMOTE
    #trn_x, trn_y = sm.fit_sample(trn_x, trn_y)
    #trn_x = pd.DataFrame(trn_x, columns = features)
    #trn_y = pd.Series(trn_y)
    #test_x[features_without_nas] = test_x[features_without_nas].fillna(trn_x[features_without_nas].median())
    
    #for n in [10, 100]:
    #   #print(n)
    #    clf = KNeighborsClassifier(n)
    #    clf.fit(trn_x[features_without_nas], trn_y)
    #    trn_x['neighbors_{n}'] = clf.predict(trn_x[features_without_nas])
    #    val_x['neighbors_{n}'] = clf.predict(val_x[features_without_nas])
    #    test_x['neighbors_{n}'] = clf.predict(test_x[features_without_nas])
       
    ## add noise to train to reduce overfitting
    trn_x += np.random.normal(0, 0.01, trn_x.shape)
    
    # print data dimensions
    print('Data shape:', trn_x.shape, val_x.shape)
    #print('Data shape:', trn_y.shape, val_y.shape)    
    # train lightGBM
    clf = lgb.LGBMClassifier(**lgb_params) 
    clf = clf.fit(trn_x, trn_y, 
                  eval_set              = [(trn_x, trn_y), (val_x, val_y)], 
                  eval_metric           = 'auc', 
                  early_stopping_rounds = stopping,
                  verbose               = verbose)
    clfs.append(clf)
    
    # find the best iteration
    best_iter = clf.best_iteration_

    # save predictions
    preds_oof[val_idx] = clf.predict_proba(val_x,  num_iteration = best_iter)[:, 1]
    preds_test        += clf.predict_proba(test_x, num_iteration = best_iter)[:, 1] / folds.n_splits 

    # importance
    fold_importance_df               = pd.DataFrame()
    fold_importance_df['Feature']    = trn_x.columns
    fold_importance_df['Importance'] = clf.feature_importances_
    fold_importance_df['Fold']       = n_fold + 1
    importances                      = pd.concat([importances, fold_importance_df], axis = 0)
    
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


# ############ VARIABLE IMPORTANCE
# 
# # load importance    
# top_feats = 300
# cols = importances[['Feature', 'Importance']].groupby('Feature').mean().sort_values(by = 'Importance', ascending = False)[0:top_feats].index
# importance = importances.loc[importances.Feature.isin(cols)]
#     
# # plot variable importance
# plt.figure(figsize = (10, 150))
# sns.barplot(x = 'Importance', y = 'Feature', data = importance.sort_values(by = 'Importance', ascending = False))
# plt.tight_layout()
# plt.savefig('./var_importance.pdf')

# SUBMISSION

# In[24]:


# file name
model = 'lgb_v49_seed111'
perf  = str(round(cv_perf, 6))[2:7]
name  = model + '_' + perf
name


# In[25]:


# export OOF preds
oof = pd.DataFrame({'encounter_id': train['encounter_id'], 'hospital_death': preds_oof})
oof.to_csv('../oof_preds/' + str(name) + '.csv', index = False)
oof.head()


# In[26]:



# export submission
sub = pd.DataFrame({'encounter_id': test['encounter_id'], 'hospital_death': preds_test})
sub.to_csv('../submissions/' + str(name) + '.csv', index = False)
sub.head()


# In[ ]:




