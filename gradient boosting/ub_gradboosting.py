# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 17:50:40 2015

@author: sony
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 22:20:05 2015

@author: sony
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier

import evaluation

from hep_ml.gradientboosting import UGradientBoostingClassifier,BinFlatnessLossFunction
#import xgboost as xgb

print("Load the training/test data using pandas")
#train = pd.read_csv("../input/training.csv")
#test  = pd.read_csv("../input/test.csv")

train = pd.read_csv('C:/Users/sony/Downloads/Compressed/CERN/training.csv',index_col='id')

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
       'p2_eta','mass']

print("Train a Random Fores and gradient boos model model")
"""
gd = GradientBoostingClassifier(n_estimators=100, random_state=5,learning_rate=0.25123,subsample=0.7,max_features=34)


rf = RandomForestClassifier(n_estimators=100,random_state=5)
ada= AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=100,random_state=5),
                        n_estimators=600, random_state=5,learning_rate=0.2)
ada.fit(train[features],train["signal"])
rf.fit(train[features],train["signal"])

"""
print("train a UBoost classifier")
loss_funct=BinFlatnessLossFunction(uniform_features=["mass"],uniform_label=0,n_bins=10)
ub=UGradientBoostingClassifier(loss=loss_funct,n_estimators=100, random_state=3,learning_rate=0.2,subsample=0.7)
ub.fit(train[features],train["signal"])

print("train a Gradientboost classifier")
gb=GradientBoostingClassifier(n_estimators=120, random_state=3,learning_rate=0.2,subsample=0.7,max_features=34)
gb.fit(train[features[0:-1]],train["signal"])

print("loading aggrement data")
check_agreement = pd.read_csv('C:/Users/sony/Downloads/Compressed/CERN/check_agreement.csv', index_col='id')

print("calculating agreement probs")
agreement_probs = 0.5*ub.predict_proba(check_agreement[features[0:-1]])[:, 1]+0.5*gb.predict_proba(check_agreement[features[0:-1]])[:, 1] 

ks = evaluation.compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)

print 'KS metric gb', ks, ks < 0.09


print("loading correlation data")
check_correlation = pd.read_csv('C:/Users/sony/Downloads/Compressed/CERN/check_correlation.csv', index_col='id')

print("calculating correlation probs")
correlation_probs =0.5*ub.predict_proba(check_correlation[features])[:, 1]+0.5*gb.predict_proba(check_correlation[features[0:-1]])[:, 1]
cvm = evaluation.compute_cvm(correlation_probs, check_correlation['mass'])
print 'CvM metric for gb', cvm, cvm < 0.002


train_eval = train[train['min_ANNmuon'] > 0.4]
print("calculating train probs having min_annmuon>0.4")
train_probs = 0.5*ub.predict_proba(train_eval[features])[:, 1]+0.5*gb.predict_proba(train_eval[features[0:-1]])[:, 1]
AUC = evaluation.roc_auc_truncated(train_eval['signal'], train_probs)
print 'AUC metric for gb', AUC




"""
print("Make predictions on the test set")
test_probs = gd.predict_proba(test[features])[:,1] 
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("gd_xgboost_submission.csv", index=False)
"""
result = pd.DataFrame({'id': test.index})
result['prediction'] = 0.5*ub.predict_proba(test[features[0:-1]])[:, 1]+0.5*gb.predict_proba(test[features[0:-1]])[:, 1]


result.to_csv('UBoost_classifier.csv', index=False, sep=',')