# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 17:44:28 2015

@author: sony
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
import evaluation

clf_nb = GaussianNB()

clf_nb.fit(train_input,train_target)

agreement_probs=clf_nb.predict_proba(agree_input)[:,1]

ks = evaluation.compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)

print 'KS metric', ks, ks < 0.09

"""
correlation_probs=clf_nb.predict_proba(corr_input)
print 'network activated on correalation dataset'
cvm = evaluation.compute_cvm(correlation_probs, check_correlation['mass'])[:,1]
print 'CvM metric', cvm, cvm < 0.002
"""

train_probs=clf_nb.predict_proba(train_eval1)[:,1]

AUC = evaluation.roc_auc_truncated(train_eval['signal'], train_probs)
print 'AUC', AUC