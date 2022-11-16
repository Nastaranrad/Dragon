#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: nastaran
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
#%%
def classifier(X, y, reps=25):
    """
      Inputs:
        X: numpy array (preprocessed data)
        y: numpy array (labels)

      outputs:
        m , s
      Description:
        this function applies a RF classifier on data and evaluate the performance
        
    """
    classNum = len(np.unique(y))
    scores = np.zeros([reps,])
    m = s = []
    for r in range(reps):
        ## train the classifier and evalute the result
        clf = RandomForestClassifier()
        skcv = StratifiedKFold(n_splits=10, shuffle=True)
        ypred = cross_val_predict(clf, X, y, cv=skcv, method='predict_proba')
        if classNum == 2:
            scores[r] = roc_auc_score(y, ypred[:,1])
        elif classNum > 2:
            scores[r] = roc_auc_score(y,ypred,multi_class='ovo')

    mean, std = scores.mean(), scores.std()
    print('Mean:%.2f, std:%.2f' %(mean, std))
    m.append(np.round(mean,2))
    s.append(np.round(std, 2))
    return m, s


#%%
def featureImportance(X,y, feature_names,n_repeats=1000):
    """
    

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    feature_names : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    clf = RandomForestClassifier()    
    clf.fit(X, y)
    auc_score = make_scorer(roc_auc_score,multi_class='ovo', needs_proba=True)
    result = permutation_importance(
    clf, X, y, n_repeats=n_repeats, n_jobs=-1,scoring = auc_score)
    forest_importances = pd.Series(result.importances_mean, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean roc_auc decrease")
    fig.tight_layout()
    plt.show()

#%%

 
    