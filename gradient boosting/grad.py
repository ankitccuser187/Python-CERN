# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 09:33:21 2015

@author: sony
"""

import pandas
from sklearn.ensemble import GradientBoostingClassifier
import evaluation

#train = pandas.read_csv('C:/Users/sony/Downloads/Compressed/CERN/training.csv', index_col='id')



variables = ['LifeTime', 'dira', 'FlightDistance', 'FlightDistanceError', 'IP',
       'IPSig', 'VertexChi2', 'pt', 'DOCAone', 'DOCAtwo', 'DOCAthree',
       'IP_p0p2', 'IP_p1p2', 'isolationa', 'isolationb', 'isolationc',
       'isolationd', 'isolatione', 'isolationf', 'iso', 'CDF1', 'CDF2',
       
       
       'p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof','p0_IP',
       'p1_IP', 'p2_IP', 'p0_IPSig', 'p1_IPSig', 'p2_IPSig', 
            'p0_eta', 'p1_eta',
       'p2_eta',
       ]
             
baseline = GradientBoostingClassifier(n_estimators=40, learning_rate=0.03, subsample=0.7,
                                      min_samples_leaf=10, max_depth=7, random_state=11)
baseline.fit(train[variables], train['signal'])


#check_agreement = pandas.read_csv('C:/Users/sony/Downloads/Compressed/CERN/check_agreement.csv', index_col='id')
agreement_probs = baseline.predict_proba(check_agreement[variables])[:, 1]

ks = evaluation.compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)

print 'KS metric', ks, ks < 0.09

#check_correlation = pandas.read_csv('C:/Users/sony/Downloads/Compressed/CERN/check_correlation.csv', index_col='id')
correlation_probs = baseline.predict_proba(check_correlation[variables])[:, 1]
cvm = evaluation.compute_cvm(correlation_probs, check_correlation['mass'])
print 'CvM metric', cvm, cvm < 0.002




train_eval = train[train['min_ANNmuon'] > 0.4]
train_probs = baseline.predict_proba(train_eval[variables])[:, 1]
AUC = evaluation.roc_auc_truncated(train_eval['signal'], train_probs)
print 'AUC', AUC

#test = pandas.read_csv('C:/Users/sony/Downloads/Compressed/CERN/test.csv', index_col='id')
result = pandas.DataFrame({'id': test.index})
result['prediction'] = baseline.predict_proba(test[variables])[:, 1]


result.to_csv('baseline.csv', index=False, sep=',')