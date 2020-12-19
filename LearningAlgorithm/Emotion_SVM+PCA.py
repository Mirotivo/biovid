# -*- coding: utf-8 -*-
"""
Created on Tue May  1 09:35:50 2018

@author: amirs
"""


import ClassifierModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# Importing the dataset
dataset = pd.read_csv('features/Table_Step2_159Features-85Subs-5Levels-z.csv')
Combined = dataset.iloc[:,7:].values

# Imputing the missing features and replacing it with the mean value excluding the first column
# axis 0 is columnwise && axis 1 is rowwise.
Combined[:,1:] = ClassifierModel.ImputeDataSet(Combined)

# Separating each level
# 0.1 level_zero_one, 1 level_one, 2 level_two,3 level_three,4 level_four
level_zero_one,level_one,level_two,level_three,level_four=ClassifierModel.SeparateEachLevel(Combined)
#Combined = np.concatenate((level_zero_one, level_two), axis=0)

Combined = np.concatenate((level_two, level_three), axis=0)
#Combined = np.concatenate((Combined, level_four), axis=0)

Combined = np.concatenate((level_one,level_four), axis=0)
level_one=level_one[0:500]
level_four=level_four[0:500]
print(len( level_one))
 # Classify between the baseline and pain threshold
 # axis 0 is columnwise && axis 1 is rowwise.
Combined = np.concatenate((level_one, level_four), axis=0)

# Encoding classes to integer levels
X,y = ClassifierModel.features_lables_split(Combined)




from sklearn.decomposition import PCA, KernelPCA
from sklearn import decomposition

pca = decomposition.PCA(n_components=83)
pca.fit(X)
X = pca.transform(X)
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 7)

# Feature Scaling

#X = ClassifierModel.NormalizeFeatures(X)

#X_train = ClassifierModel.NormalizeFeatures(X_train)
#X_test = ClassifierModel.NormalizeFeatures(X_test)


# Fitting Kernel SVM to the Training set
classifier = SVC(C=1.0, cache_size=100, class_weight=None, coef0=0.001, decision_function_shape='ovr', degree=1, gamma='auto', kernel='linear', max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.0001, verbose=False) 

#classifier.fit(X_train, y_train)

# Predicting the Test set results
#y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
#cm = confusion_matrix(y_test, y_pred)
#ClassifierModel.Visualize_CM(cm,[0,1])

# Evaluation
#accuracy,ave,precision,recall,sensitivity,specificity = ClassifierModel.EvaluateClassifier(y_test, y_pred)
#print('Accuracy (BvsT3): '+str(accuracy))
#print('Precision (BvsT3): '+str(precision))
#print('Recall (BvsT3): '+str(recall))
#print('Sensitivity (BvsT3): '+str(sensitivity))
#print('Specificity (BvsT3): '+str(specificity))
#print('Cramér''s V (BvsT3): '+str(cramersV))




















import ClassifierModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve, auc
classifier = SVC(C=1.0, cache_size=100, class_weight=None, coef0=0.001, decision_function_shape='ovr', degree=1, gamma='auto', kernel='linear', max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.0001, verbose=False) 
probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)
fpr_svm, tp_svmr, thresholds = roc_curve(y_test, probas_[:, 1])
roc_auc_svm = auc(fpr_svm, tp_svmr)


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)
probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)
fpr_knn, tp_knn, thresholds = roc_curve(y_test, probas_[:, 1])
roc_auc_knn= auc(fpr_knn, tp_knn)

classifier = RandomForestClassifier() 
probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)
fpr_rf, tp_rf, thresholds = roc_curve(y_test, probas_[:, 1])
roc_auc_rf = auc(fpr_rf, tp_rf)


classifier = SVC(kernel='rbf',probability=True) 
probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)
fpr_rbf, tp_rbf, thresholds = roc_curve(y_test, probas_[:, 1])
roc_auc_rbf = auc(fpr_rbf, tp_rbf)
    
plt.figure(figsize=(7,7))
plt.clf()
plt.plot(fpr_svm, tp_svmr, label='SVM (area = %0.2f)' % roc_auc_svm)
plt.plot(fpr_knn, tp_knn, label='KNN (area = %0.2f)' % roc_auc_knn)
plt.plot(fpr_rf, tp_rf, label='Random Forest (area = %0.2f)' % roc_auc_rf)
plt.plot(fpr_rbf, tp_rbf, label='RBF (area = %0.2f)' % roc_auc_rbf)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('OCR')
plt.legend(loc="lower right")
plt.savefig('EMO_7.png', format='png', dpi=1200)

plt.show()
























from sklearn import metrics
from sklearn.metrics import roc_curve, auc

probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])

roc_auc = auc(fpr, tpr)
plt.figure(figsize=(7,7))
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('OCR')
plt.legend(loc="lower right")
plt.savefig('AUC_0.png', format='png', dpi=1200)

plt.show()


fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()




from sklearn.model_selection import cross_val_score


accourises=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=20)
accourises.mean()

from sklearn import svm, grid_search


from sklearn.model_selection import GridSearchCV

Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas,'kernel':['linear','rbf']}
grid_search = GridSearchCV(classifier, param_grid, cv=10)
grid_search.fit(X_train, y_train)
best_params_=    grid_search.best_params_

print(grid_search.best_score_)

def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_



