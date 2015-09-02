# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 18:08:31 2015

@author: sony
"""
import pandas as pd
import numpy as np

from pybrain.structure import FeedForwardNetwork
n=FeedForwardNetwork()

from pybrain.structure import SigmoidLayer
from pybrain.structure.modules import SoftmaxLayer,LinearLayer
inLayer=SigmoidLayer(34)
hiddenLayer=SigmoidLayer(6)
hiddenLayer2=SigmoidLayer(3)
outLayer=SigmoidLayer(1)

n.addInputModule(inLayer)
n.addModule(hiddenLayer)
n.addModule(hiddenLayer2)
n.addOutputModule(outLayer)

from pybrain.structure import FullConnection
in_to_hidden=FullConnection(inLayer,hiddenLayer)
hidden_to_hidden2=FullConnection(hiddenLayer,hiddenLayer2)
hidden_to_out=FullConnection(hiddenLayer2,outLayer)

n.addConnection(in_to_hidden)
n.addConnection(hidden_to_hidden2)
n.addConnection(hidden_to_out)

#code to organize the network module 
n.sortModules()

#to look over the randomly intialized weights
n.params


#train=pd.read_csv("C:/Users/sony/Downloads/Compressed/CERN/train.csv")

#test=pd.read_csv("C:/Users/sony/Downloads/Compressed/CERN/test.csv")
"""
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
"""
#load dataframe as a 2d matrix
#train=np.loadtxt(open("C:/Users/sony/Downloads/Compressed/CERN/train.csv","rb"),delimiter=",",skiprows=1)

#trainer configuration on dataset
#trainer=BackpropTrainer(n,dataset=ds, learningrate=0.03, lrdecay=1.0, momentum=0.0, verbose=False, batchlearning=False, weightdecay=0.0)