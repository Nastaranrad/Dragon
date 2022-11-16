#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 12:06:39 2022

@author: nastaran
"""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

#%%
def Preprocessing(selected_features,drop_features,id_):
    """
    

    Parameters
    ----------
    selected_features : TYPE
        DESCRIPTION.
    drop_features : TYPE
        DESCRIPTION.
    id_ : TYPE
        DESCRIPTION.

    Returns
    -------
    features : TYPE
        DESCRIPTION.

    """
    ### remove features in drop_features from the data
    features = selected_features.drop(drop_features, axis=1)
    selectedFeatures_names = features.keys().tolist()

    ### set the id as the index of dataframe
    features = features.set_index(id_)
    features = features.to_numpy()
    ######

    ######
    features[features[:,28]== 3] = np.NAN ## decode 3 to nan values
    ### decode 2 to nan value in the specified features 
    indices = [35,36]+[i for i in range(38,69)]
    for i in indices:
        features[features[:,i] == 2] = np.NaN        
    return features, selectedFeatures_names

#%%
def Imputer(features, features_type,feature_names,one_hot_encoder = False):
    """
    

    Parameters
    ----------
    feature_names : TYPE
        DESCRIPTION.
    features : TYPE
        DESCRIPTION.
    features_type : TYPE
        DESCRIPTION.
    one_hot_encoder should be set to True for ML classifiers such as logistic regression
    Returns
    -------
    X.

    """
    ImputedFeatures = [] 

    for f, feature in enumerate(feature_names):    

        if features_type[f] == 'bin':
            imputer = SimpleImputer(strategy='most_frequent')
            features[:,f:f+1] = imputer.fit_transform(features[:,f:f+1])
            features[features[:,f]==0,f] = -1
            ImputedFeatures.append(features[:,f:f+1])
        elif features_type[f] == 'con':
            imputer = SimpleImputer(strategy='median')
            features[:,f:f+1] = imputer.fit_transform(features[:,f:f+1])
            ImputedFeatures.append(features[:,f:f+1])
        elif features_type[f] == 'cat':
            imputer = SimpleImputer(strategy='most_frequent')
            features[:,f:f+1] = imputer.fit_transform(features[:,f:f+1])
            if one_hot_encoder:
                ohe = OneHotEncoder(sparse = False) 
                temp = ohe.fit_transform(features[:,f:f+1]) 
                ImputedFeatures.append(temp)
            else:   
                ImputedFeatures.append(features[:,f:f+1])
    X = np.concatenate(ImputedFeatures,axis=1)
    return X

