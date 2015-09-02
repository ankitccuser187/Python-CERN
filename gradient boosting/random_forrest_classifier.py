# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 22:20:05 2015

@author: sony
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier

import evaluation
#import xgboost as xgb

print("Load the training/test data using pandas")
#train = pd.read_csv("../input/training.csv")
#test  = pd.read_csv("../input/test.csv")

train = pd.read_csv('C:/Users/sony/Downloads/Compressed/CERN/training.csv', index_col='id')

#randomize the training sample
train=train.iloc[np.random.permutation(len(train))]

test = pd.read_csv('C:/Users/sony/Downloads/Compressed/CERN/test.csv', index_col='id')

print("Eliminate SPDhits, which makes the agreement check fail")
features= ['LifeTime', 'dira', 'FlightDistance', 'FlightDistanceError', 'IP',
       'IPSig', 'VertexChi2', 'pt', 'DOCAone', 'DOCAtwo', 'DOCAthree',
       'IP_p0p2', 'IP_p1p2', 'isolationa', 'isolationb', 'isolationc',
       'isolationd', 'isolatione', 'isolationf', 'iso', 'CDF1', 'CDF2',
       
       
       'p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof','p0_IP',
       'p1_IP', 'p2_IP', 'p0_IPSig', 'p1_IPSig', 'p2_IPSig', 
            'p0_eta', 'p1_eta',
       'p2_eta',]

print("Train a Random Fores and gradient boos model model")
gd = GradientBoostingClassifier(n_estimators=100, random_state=5,learning_rate=0.25123,subsample=0.7,max_features=34)
rf = RandomForestClassifier(n_estimators=100,random_state=5)
#ada= AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=100,random_state=5),
                        n_estimators=600, random_state=5,learning_rate=0.2)
ada.fit(train[features],train["signal"])
rf.fit(train[features],train["signal"])
gd.fit(train[features], train["signal"])

check_agreement = pd.read_csv('C:/Users/sony/Downloads/Compressed/CERN/check_agreement.csv', index_col='id')
agreement_probs = gd.predict_proba(check_agreement[features])[:, 1]
"""
ks = evaluation.compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)

print 'KS metric gb', ks, ks < 0.09
"""
agreement_probs1 = rf.predict_proba(check_agreement[features])[:, 1]

ks1 = evaluation.compute_ks(
    0.3*agreement_probs1[check_agreement['signal'].values == 0]+
    0.7*agreement_probs[check_agreement['signal'].values == 0],
    0.3*agreement_probs1[check_agreement['signal'].values == 1]+
    0.7*agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)

print 'KS metric rf', ks1, ks1 < 0.09

check_correlation = pd.read_csv('C:/Users/sony/Downloads/Compressed/CERN/check_correlation.csv', index_col='id')
correlation_probs = gd.predict_proba(check_correlation[features])[:, 1]
#cvm = evaluation.compute_cvm(correlation_probs, check_correlation['mass'])
#print 'CvM metric for gb', cvm, cvm < 0.002

correlation_probs1 = rf.predict_proba(check_correlation[features])[:, 1]
cvm1 = evaluation.compute_cvm(0.3*correlation_probs1+0.7*correlation_probs, check_correlation['mass'])
print 'CvM metric for rf', cvm1, cvm1 < 0.002

train_eval = train[train['min_ANNmuon'] > 0.4]
train_probs = gd.predict_proba(train_eval[features])[:, 1]
#AUC = evaluation.roc_auc_truncated(train_eval['signal'], train_probs)
#print 'AUC metric for gb', AUC

#train_eval1 = train[train['min_ANNmuon'] > 0.4]
train_probs1 = rf.predict_proba(train_eval[features])[:, 1]
AUC = evaluation.roc_auc_truncated(train_eval['signal'], .3*train_probs1+0.7*train_probs)
print 'AUC metric for rf', AUC


"""
print("Make predictions on the test set")
test_probs = gd.predict_proba(test[features])[:,1] 
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("gd_xgboost_submission.csv", index=False)
"""
result = pd.DataFrame({'id': test.index})
result['prediction'] = 0.7*gd.predict_proba(test[features])[:, 1]+0.3*rf.predict_proba(test[features])[:, 1]


result.to_csv('GradientBoost_classifier.csv', index=False, sep=',')