import pandas, numpy
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
# this wrapper makes it possible to train on subset of features
from rep.estimators import SklearnClassifier

from hep_ml.commonutils import train_test_split
from hep_ml import uboost, gradientboosting as ugb, losses
from rep.metaml import ClassifiersFactory

train = pd.read_csv('C:/Users/sony/Downloads/Compressed/CERN/training.csv',index_col='id')

#randomize the training sample
train=train.iloc[np.random.permutation(len(train))]

classifiers = ClassifiersFactory()
train_features= ['LifeTime', 'dira', 'FlightDistance', 'FlightDistanceError', 'IP',
       'IPSig', 'VertexChi2', 'pt', 'DOCAone', 'DOCAtwo', 'DOCAthree',
       'IP_p0p2', 'IP_p1p2', 'isolationa', 'isolationb', 'isolationc',
       'isolationd', 'isolatione', 'isolationf', 'iso', 'CDF1', 'CDF2',
       
       
       'p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof','p0_IP',
       'p1_IP', 'p2_IP', 'p0_IPSig', 'p1_IPSig', 'p2_IPSig', 
            'p0_eta', 'p1_eta',
       'p2_eta']
       
uniform_features  = ["mass"]       

n_estimators = 150
base_estimator = DecisionTreeClassifier(max_depth=4)

base_ada = GradientBoostingClassifier(max_depth=4, n_estimators=100, learning_rate=0.1)
AdaBoost = SklearnClassifier(base_ada, features=train_features)


knnloss = ugb.KnnAdaLossFunction(uniform_features, knn=10, uniform_label=1)
ugbKnn = ugb.UGradientBoostingClassifier(loss=knnloss, max_depth=4, n_estimators=n_estimators,
                                        learning_rate=0.4, train_features=train_features)
uGB+knnAda = SklearnClassifier(ugbKnn) 

uboost_clf = uboost.uBoostClassifier(uniform_features=uniform_features, uniform_label=1,
                                     base_estimator=base_estimator, 
                                     n_estimators=n_estimators, train_features=train_features, 
                                     efficiency_steps=12, n_threads=4)
uBoost = SklearnClassifier(uboost_clf)

flatnessloss = ugb.KnnFlatnessLossFunction(uniform_features, fl_coefficient=3., power=1.3, uniform_label=1)
ugbFL = ugb.UGradientBoostingClassifier(loss=flatnessloss, max_depth=4, 
                                       n_estimators=n_estimators, 
                                       learning_rate=0.1, train_features=train_features)
uGB+FL = SklearnClassifier(ugbFL)


AdaBoost.fit(train_i)
uGB+knnAda.fit(train_i)
uBoost.fit(train_i)
uGB+FL.fit(train_i)

pass