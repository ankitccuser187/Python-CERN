# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:55:30 2015

@author: sony
"""
import evaluation
import load_data_module

# this a trainer module
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import ClassificationDataSet
import pandas as pd
import numpy as np
trainer=BackpropTrainer(n, dataset=ds, learningrate=0.3, lrdecay=1.0, momentum=0.0, verbose=False, batchlearning=False, weightdecay=0.0)

trainer.trainEpochs (20)
trainer.trainOnDataset(ds)

print 'backprop trained on training dataset'

agreement_probs= n.activateOnDataset(ds_agree)

print 'network activated on agreement dataset'
ks = evaluation.compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)

print 'KS metric', ks, ks < 0.09

"""
correlation_probs=n.activateOnDataset(ds_corr)
print 'network activated on correalation dataset'
cvm = evaluation.compute_cvm(correlation_probs, check_correlation['mass'])
print 'CvM metric', cvm, cvm < 0.002
"""

train_eval = train[train['min_ANNmuon'] > 0.4]
train_eval_target=(train_eval['signal']).as_matrix()
train_eval1=train_eval.ix[:,'LifeTime':'p2_eta']

train_eval1=train_eval1.as_matrix()

train_eval_ds= ClassificationDataSet(45, 1 , nb_classes=2)
for k in xrange(len(train_eval)): 
 train_eval_ds.addSample((train_eval1[k,:]),train_eval_target[k])
 
train_probs=n.activateOnDataset(train_eval_ds)
AUC = evaluation.roc_auc_truncated(train_eval['signal'], train_probs)
print 'AUC', AUC

"""
result = pd.DataFrame({'id': test.index})
result['prediction']=n.activateOnDataset(DS)
result.to_csv('result.csv', index=False, sep=',')
"""