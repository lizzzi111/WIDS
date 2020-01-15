###############################
#                             #
#        ENCODE FACTORS       #
#                             #
###############################

# performs label encoding
from sklearn.preprocessing import LabelEncoder
def label_encoding(df_train, df_valid, df_test):
    
    factors = df_train.select_dtypes('object').columns
    
    lbl = LabelEncoder()

    for f in factors:        
        lbl.fit(list(df_train[f].values) + list(df_valid[f].values) + list(df_test[f].values))
        df_train[f] = lbl.transform(list(df_train[f].values))
        df_valid[f] = lbl.transform(list(df_valid[f].values))
        df_test[f]  = lbl.transform(list(df_test[f].values))

    return df_train, df_valid, df_test

import numpy as np
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


import scipy.stats
def aggregate_data(df, group_var, num_stats = ['mean', 'sum'], factors = None, var_label = None, sd_zeros = False):
    
    
    ### SEPARATE FEATURES
  
    # display info
    print("- Preparing the dataset...")
    print(group_var)
    print("____________________________")

    # find factors
    if factors == None:
        df_factors = [f for f in df.columns if df[f].dtype == "object"]
        factors    = [f for f in df_factors if f != group_var]
    else:
        df_factors = factors
        df_factors.append(group_var)
        
    # partition subsets
    if type(group_var) == str:
        num_df = df[[group_var] + list(set(df.columns) - set(df_factors))]
        fac_df = df[df_factors]
    else:
        num_df = df[group_var + list(set(df.columns) - set(df_factors))]
        fac_df = df[df_factors] 
    
    # display info
    num_facs = fac_df.shape[1] - 1
    num_nums = num_df.shape[1] - 1
    print("- Extracted %.0f factors and %.0f numerics..." % (num_facs, num_nums))
    

    ##### AGGREGATION
 
    # aggregate numerics
    if (num_nums > 0):
        print("- Aggregating numeric features...")
        num_df = num_df.groupby([group_var]).agg(num_stats)
        num_df.columns = ["_".join(col).strip() for col in num_df.columns.values]
        num_df = num_df.sort_index()

    # aggregate factors
    if (num_facs > 0):
        print("- Aggregating factor features...")
        fac_int_df = fac_df.copy()
        for var in factors:
            fac_int_df[var], _ = pd.factorize(fac_int_df[var])
        fac_int_df[group_var] = fac_df[group_var]
        fac_df = fac_int_df.groupby([group_var]).agg([('count'), ('mode', lambda x: pd.Series.mode(x)[0])])
        fac_df.columns = ["_".join(col).strip() for col in fac_df.columns.values]
        fac_df = fac_df.sort_index()           


    ##### MERGER

    # merge numerics and factors
    if ((num_facs > 0) & (num_nums > 0)):
        agg_df = pd.concat([num_df, fac_df], axis = 1)
    
    # use factors only
    if ((num_facs > 0) & (num_nums == 0)):
        agg_df = fac_df
        
    # use numerics only
    if ((num_facs == 0) & (num_nums > 0)):
        agg_df = num_df
        

    ##### LAST STEPS

    # update labels
    if (var_label != None):
        agg_df.columns = [var_label + "_" + str(col) for col in agg_df.columns]
    
    # impute zeros for SD
    if (sd_zeros == True):
        stdevs = agg_df.filter(like = "_std").columns
        for var in stdevs:
            agg_df[var].fillna(0, inplace = True)
            
    # dataset
    agg_df = agg_df.reset_index()
    print("- Final dimensions:", agg_df.shape)
    return agg_df
