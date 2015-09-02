# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 18:03:55 2015

@author: sony
"""

"""import numpy as np
test['signal'] = np.random.choice(range(0,2 ), test.shape[0])"""


from pybrain.datasets import ClassificationDataSet
import pandas as pd
import numpy as np

features= ['LifeTime', 'dira', 'FlightDistance', 'FlightDistanceError', 'IP',
       'IPSig', 'VertexChi2', 'pt', 'DOCAone', 'DOCAtwo', 'DOCAthree',
       'IP_p0p2', 'IP_p1p2', 'isolationa', 'isolationb', 'isolationc',
       'isolationd', 'isolatione', 'isolationf', 'iso', 'CDF1', 'CDF2',
       
       
       'p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof','p0_IP',
       'p1_IP', 'p2_IP', 'p0_IPSig', 'p1_IPSig', 'p2_IPSig', 
            'p0_eta', 'p1_eta',
       'p2_eta']

#loading and configuring the training dataset
train = pd.read_csv('C:/Users/sony/Downloads/Compressed/CERN/training.csv', index_col='id')
train_input=(train[features]).as_matrix()
train_target=(train['signal']).as_matrix()
ds = ClassificationDataSet(34, 1 , nb_classes=2)
for k in xrange(len(train_input)): 
 ds.addSample((train_input[k,:]),(train_target[k]))
 
#loading and configuring agrrement module 
check_agreement = pd.read_csv('C:/Users/sony/Downloads/Compressed/CERN/check_agreement.csv', index_col='id') 
agree_input=(check_agreement[features]).as_matrix()
agree_target=(check_agreement['signal']).as_matrix()
ds_agree = ClassificationDataSet(34, 1 , nb_classes=2)
for k in xrange(len(agree_input)): 
 ds_agree.addSample((agree_input[k,:]),(agree_target[k]))
 
 
 #loading and configuring the corelation module
check_correlation = pd.read_csv('C:/Users/sony/Downloads/Compressed/CERN/check_correlation.csv', index_col='id')
corr_input=(check_correlation[features]).as_matrix()
# introducing dummy variable
check_correlation['signal']=np.random.choice(range(0,2),check_correlation.shape[0])
corr_target=(check_agreement['signal']).as_matrix()
ds_corr = ClassificationDataSet(34, 1 , nb_classes=2)
for k in xrange(len(corr_input)): 
 ds_corr.addSample((corr_input[k,:]),(corr_target[k]))
 
test = pd.read_csv('C:/Users/sony/Downloads/Compressed/CERN/test.csv', index_col='id')
test['signal']=np.random.choice(range(0,2),test.shape[0])
test_input=(test[features]).as_matrix()
test_target=(test['signal']).as_matrix()
DS = ClassificationDataSet(34, 1 , nb_classes=2)
for k in xrange(len(test_input)): 
 DS.addSample((test_input[k,:]),(test_target[k]))