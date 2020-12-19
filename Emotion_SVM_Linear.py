import ClassifierModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Importing the dataset
#dataset = pd.read_csv('features/Table_Step2_159Features-85Subs-5Levels-z.csv')
#Combined = dataset.iloc[:,7:].values
# Importing the dataset
dataset = pd.read_csv('features/Emotion Features Combined.csv')
Combined = dataset.iloc[:,:].values

# Imputing the missing features and replacing it with the mean value excluding the first column
# axis 0 is columnwise && axis 1 is rowwise.
Combined[:,1:] = ClassifierModel.ImputeDataSet(Combined)

# Separating each level
# 0.1 level_zero_one, 1 level_one, 2 level_two,3 level_three,4 level_four
level_zero_one,level_one,level_two,level_three,level_four=ClassifierModel.SeparateEachLevel(Combined)

# Classify between the baseline and pain threshold
# axis 0 is columnwise && axis 1 is rowwise.
Combined = np.concatenate((level_zero_one, level_one), axis=0)

# Encoding classes to integer levels
X,y = ClassifierModel.features_lables_split(Combined)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 7)

# Feature Scaling
X_train = ClassifierModel.NormalizeFeatures(X_train)
X_test = ClassifierModel.NormalizeFeatures(X_test)

# Fitting Kernel SVM to the Training set
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ClassifierModel.Visualize_CM(cm,[0,1])







# Fitting Kernel SVM to the Training set
classifier = SVC(C=1.0, cache_size=100, class_weight=None, coef0=0.001, decision_function_shape='ovr', degree=1, gamma='auto', kernel='linear', max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.0001, verbose=False) 



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
plt.savefig('AUC_0.png', format='png', dpi=1200)

plt.show()






# Below for loop iterates through your models list
for m in models:
    model = m['model'] # select the model
    model.fit(x_train, y_train) # train the model
    y_pred=model.predict(x_test) # predict the test data
# Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(x_test)[:,1])
# Calculate Area under the curve to display on the plot
    auc = metrics.roc_auc_score(y_test,model.predict(x_test))
# Now, plot the computed values
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], auc))
# Custom settings for the plot 
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()   # Display







# Evaluation
accuracy,precision,recall,sensitivity,specificity,cramersV,_ = ClassifierModel.EvaluateClassifier(y_test, y_pred)
print('Accuracy (BvsT1): '+str(accuracy))
print('Precision (BvsT1): '+str(precision))
print('Recall (BvsT1): '+str(recall))
print('Sensitivity (BvsT1): '+str(sensitivity))
print('Specificity (BvsT1): '+str(specificity))
print('Cramér''s V (BvsT1): '+str(cramersV))

# Classify between the baseline and pain threshold
# axis 0 is columnwise && axis 1 is rowwise.
Combined = np.concatenate((level_zero_one, level_two), axis=0)

# Encoding classes to integer levels
X,y = ClassifierModel.features_lables_split(Combined)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 7)

# Feature Scaling
X_train = ClassifierModel.NormalizeFeatures(X_train)
X_test = ClassifierModel.NormalizeFeatures(X_test)

# Fitting Kernel SVM to the Training set
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ClassifierModel.Visualize_CM(cm,[0,1])

# Evaluation
accuracy,precision,recall,sensitivity,specificity,cramersV = ClassifierModel.EvaluateClassifier(y_test, y_pred)
print('Accuracy (BvsT2): '+str(accuracy))
print('Precision (BvsT2): '+str(precision))
print('Recall (BvsT2): '+str(recall))
print('Sensitivity (BvsT2): '+str(sensitivity))
print('Specificity (BvsT2): '+str(specificity))
print('Cramér''s V (BvsT2): '+str(cramersV))


# Classify between the baseline and pain threshold
# axis 0 is columnwise && axis 1 is rowwise.
Combined = np.concatenate((level_zero_one, level_three), axis=0)

# Encoding classes to integer levels
X,y = ClassifierModel.features_lables_split(Combined)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 7)

# Feature Scaling
X_train = ClassifierModel.NormalizeFeatures(X_train)
X_test = ClassifierModel.NormalizeFeatures(X_test)

# Fitting Kernel SVM to the Training set
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ClassifierModel.Visualize_CM(cm,[0,1])

# Evaluation
accuracy,precision,recall,sensitivity,specificity,cramersV = ClassifierModel.EvaluateClassifier(y_test, y_pred)
print('Accuracy (BvsT3): '+str(accuracy))
print('Precision (BvsT3): '+str(precision))
print('Recall (BvsT3): '+str(recall))
print('Sensitivity (BvsT3): '+str(sensitivity))
print('Specificity (BvsT3): '+str(specificity))
print('Cramér''s V (BvsT3): '+str(cramersV))



# Classify between the baseline and pain threshold
# axis 0 is columnwise && axis 1 is rowwise.
Combined = np.concatenate((level_zero_one, level_four), axis=0)

# Encoding classes to integer levels
X,y = ClassifierModel.features_lables_split(Combined)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 7)

# Feature Scaling
X_train = ClassifierModel.NormalizeFeatures(X_train)
X_test = ClassifierModel.NormalizeFeatures(X_test)

# Fitting Kernel SVM to the Training set
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ClassifierModel.Visualize_CM(cm,[0,1])

# Evaluation
accuracy,precision,recall,sensitivity,specificity,cramersV = ClassifierModel.EvaluateClassifier(y_test, y_pred)
print('Accuracy (BvsT4): '+str(accuracy))
print('Precision (BvsT4): '+str(precision))
print('Recall (BvsT4): '+str(recall))
print('Sensitivity (BvsT4): '+str(sensitivity))
print('Specificity (BvsT4): '+str(specificity))
print('Cramér''s V (BvsT4): '+str(cramersV))


# Classify between the baseline and pain threshold
# axis 0 is columnwise && axis 1 is rowwise.
Combined = np.concatenate((level_zero_one, level_one), axis=0)
Combined = np.concatenate((Combined, level_four), axis=0)

# Encoding classes to integer levels
X,y = ClassifierModel.features_lables_split(Combined)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 7)

# Feature Scaling
X_train = ClassifierModel.NormalizeFeatures(X_train)
X_test = ClassifierModel.NormalizeFeatures(X_test)

# Fitting Kernel SVM to the Training set
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ClassifierModel.Visualize_CM(cm,[0,1,2])

# Evaluation
accuracy,precision,recall,sensitivity,specificity,cramersV = ClassifierModel.EvaluateClassifier(y_test, y_pred)
print('Accuracy (BvsT1vsT4): '+str(accuracy))
print('Precision (BvsT1vsT4): '+str(precision))
print('Recall (BvsT1vsT4): '+str(recall))
print('Sensitivity (BvsT1vsT4): '+str(sensitivity))
print('Specificity (BvsT1vsT4): '+str(specificity))
print('Cramér''s V (BvsT1vsT4): '+str(cramersV))




# Classify between the baseline and pain threshold
# axis 0 is columnwise && axis 1 is rowwise.
Combined = np.concatenate((level_zero_one, level_one), axis=0)
Combined = np.concatenate((Combined, level_two), axis=0)
Combined = np.concatenate((Combined, level_three), axis=0)
Combined = np.concatenate((Combined, level_four), axis=0)

# Encoding classes to integer levels
X,y = ClassifierModel.features_lables_split(Combined)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 7)

# Feature Scaling
X_train = ClassifierModel.NormalizeFeatures(X_train)
X_test = ClassifierModel.NormalizeFeatures(X_test)

# Fitting Kernel SVM to the Training set
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ClassifierModel.Visualize_CM(cm,[0,1,2,3,4])

# Evaluation
accuracy,precision,recall,sensitivity,specificity,cramersV = ClassifierModel.EvaluateClassifier(y_test, y_pred)
print('Accuracy (BvsT1vsT2vsT3vsT4): '+str(accuracy))
print('Precision (BvsT1vsT2vsT3vsT4): '+str(precision))
print('Recall (BvsT1vsT2vsT3vsT4): '+str(recall))
print('Sensitivity (BvsT1vsT2vsT3vsT4): '+str(sensitivity))
print('Specificity (BvsT1vsT2vsT3vsT4): '+str(specificity))
print('Cramér''s V (BvsT1vsT2vsT3vsT4): '+str(cramersV))
