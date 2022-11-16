#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: nastaran.mrad@maastrichtuniversity.nl


Analysis of clinical data (WP8), DRAGON project
"""

import numpy as np
import pandas as pd
import pyreadstat
from preprocessing import Preprocessing, Imputer
from classification import classifier,featureImportance
#%% read data
dataPath = 'path to data'
data, meta = pyreadstat.read_sav(dataPath + 'Data_MTDNA_InterimAnalysis_20092022.sav')

#%%

#### read the variable list and select the features based on that
VariableList = pd.read_excel(dataPath + 'List_of_variables_to_use.xlsx')
featureNames = VariableList['ID']        
selectedFeatures = data[featureNames]
id_ = data['ID']

### Select the target
y = np.asarray(selectedFeatures ['Hosp_Adm']) ##1 severe 0: non severe patients

### drop features that have problems (need to be decoded, have many missing values, or are repetitive)
dropFeatures = ['COVID_Date','Clinical_Centre', 'Meds_DM_Oth_Open','Meds_HTN_Oth_Open','Risk_Oth_Open', 'ICU_Adm', 'Sym_Blee_Site',
                 'Meds_Condition','Meds_Oth_Open','Sym_Oth','Hosp_Adm']
   

### set the type of features (binary, categorical, or continous values)

featuresType = ['cat','bin','con','con','con','bin','bin','bin','bin',
               'bin','bin','bin','bin','bin','bin','bin','bin','bin',
               'bin','bin','bin','bin','bin','bin','cat','bin','bin',
               'bin','cat','bin','con','con','con','con','con',               
               'bin', 'bin','con','bin','bin','bin','bin','bin','bin',
               'bin','bin','bin','bin','bin','bin','bin','bin','bin',
               'bin','bin','bin','bin','bin','bin','bin','bin','bin',
               'bin','bin','bin','bin','bin','bin','con']

features,selectedFeatures_names = Preprocessing(selectedFeatures,dropFeatures,id_)
X = Imputer(features, featuresType,selectedFeatures_names,one_hot_encoder = False)
m, s = classifier(X, y) 
featureImportance(X,y, selectedFeatures_names,n_repeats=1000)
 


