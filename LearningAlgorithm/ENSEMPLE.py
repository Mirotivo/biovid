import ClassifierModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

print('--------------------------------------------------------')
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






from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Fitting Kernel SVM to the Training set
log_clf  = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
rnd_clf = RandomForestClassifier()
svm_clf = SVC()
voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
        voting='hard'
    )
voting_clf.fit(X_train, y_train)





# Fitting Kernel SVM to the Training set
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ClassifierModel.Visualize_CM(cm,[0,1])

# Evaluation
accuracy,average_precision,precision,recall,sensitivity,specificity,cramersV = ClassifierModel.EvaluateClassifier(y_test, y_pred)
print('Accuracy (BvsT1): '+str(accuracy))
print('Average Precision (BvsT1): '+str(average_precision))
print('Precision (BvsT1): '+str(precision))
print('Recall (BvsT1): '+str(recall))
print('Sensitivity (BvsT1): '+str(sensitivity))
print('Specificity (BvsT1): '+str(specificity))
print('Cramér''s V (BvsT1): '+str(cramersV))

print('--------------------------------------------------------')
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
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ClassifierModel.Visualize_CM(cm,[0,1])

# Evaluation
accuracy,average_precision,precision,recall,sensitivity,specificity,cramersV = ClassifierModel.EvaluateClassifier(y_test, y_pred)
print('Accuracy (BvsT2): '+str(accuracy))
print('Average Precision (BvsT2): '+str(average_precision))
print('Precision (BvsT2): '+str(precision))
print('Recall (BvsT2): '+str(recall))
print('Sensitivity (BvsT2): '+str(sensitivity))
print('Specificity (BvsT2): '+str(specificity))
print('Cramér''s V (BvsT2): '+str(cramersV))


print('--------------------------------------------------------')
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
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ClassifierModel.Visualize_CM(cm,[0,1])

# Evaluation
accuracy,average_precision,precision,recall,sensitivity,specificity,cramersV = ClassifierModel.EvaluateClassifier(y_test, y_pred)
print('Accuracy (BvsT3): '+str(accuracy))
print('Average Precision (BvsT3): '+str(average_precision))
print('Precision (BvsT3): '+str(precision))
print('Recall (BvsT3): '+str(recall))
print('Sensitivity (BvsT3): '+str(sensitivity))
print('Specificity (BvsT3): '+str(specificity))
print('Cramér''s V (BvsT3): '+str(cramersV))



print('--------------------------------------------------------')
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
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ClassifierModel.Visualize_CM(cm,[0,1])

# Evaluation
accuracy,average_precision,precision,recall,sensitivity,specificity,cramersV = ClassifierModel.EvaluateClassifier(y_test, y_pred)
print('Accuracy (BvsT4): '+str(accuracy))
print('Average Precision (BvsT4): '+str(average_precision))
print('Precision (BvsT4): '+str(precision))
print('Recall (BvsT4): '+str(recall))
print('Sensitivity (BvsT4): '+str(sensitivity))
print('Specificity (BvsT4): '+str(specificity))
print('Cramér''s V (BvsT4): '+str(cramersV))


print('--------------------------------------------------------')
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
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ClassifierModel.Visualize_CM(cm,[0,1,2])

# Evaluation
accuracy,average_precision,precision,recall,sensitivity,specificity,cramersV = ClassifierModel.EvaluateClassifier(y_test, y_pred)
print('Accuracy (BvsT1vsT4): '+str(accuracy))
print('Average Precision (BvsT1vsT4): '+str(average_precision))
print('Precision (BvsT1vsT4): '+str(precision))
print('Recall (BvsT1vsT4): '+str(recall))
print('Sensitivity (BvsT1vsT4): '+str(sensitivity))
print('Specificity (BvsT1vsT4): '+str(specificity))
print('Cramér''s V (BvsT1vsT4): '+str(cramersV))



print('--------------------------------------------------------')
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
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ClassifierModel.Visualize_CM(cm,[0,1,2,3,4])

# Evaluation
accuracy,average_precision,precision,recall,sensitivity,specificity,cramersV = ClassifierModel.EvaluateClassifier(y_test, y_pred)
print('Accuracy (BvsT1vsT2vsT3vsT4): '+str(accuracy))
print('Average Precision (BvsT1vsT2vsT3vsT4): '+str(average_precision))
print('Precision (BvsT1vsT2vsT3vsT4): '+str(precision))
print('Recall (BvsT1vsT2vsT3vsT4): '+str(recall))
print('Sensitivity (BvsT1vsT2vsT3vsT4): '+str(sensitivity))
print('Specificity (BvsT1vsT2vsT3vsT4): '+str(specificity))
print('Cramér''s V (BvsT1vsT2vsT3vsT4): '+str(cramersV))



print('--------------------------------------------------------')
# Classify between the baseline and pain threshold
# axis 0 is columnwise && axis 1 is rowwise.
Combined = np.concatenate((level_one, level_two), axis=0)

# Encoding classes to integer levels
X,y = ClassifierModel.features_lables_split(Combined)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 7)

# Feature Scaling
X_train = ClassifierModel.NormalizeFeatures(X_train)
X_test = ClassifierModel.NormalizeFeatures(X_test)

# Fitting Kernel SVM to the Training set
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier = classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ClassifierModel.Visualize_CM(cm,[0,1])

# Evaluation
accuracy,average_precision,precision,recall,sensitivity,specificity,cramersV = ClassifierModel.EvaluateClassifier(y_test, y_pred)
print('Accuracy (T1vsT2): '+str(accuracy))
print('Average Precision (T1vsT2): '+str(average_precision))
print('Precision (T1vsT2): '+str(precision))
print('Recall (T1vsT2): '+str(recall))
print('Sensitivity (T1vsT2): '+str(sensitivity))
print('Specificity (T1vsT2): '+str(specificity))
print('Cramér''s V (T1vsT2): '+str(cramersV))

print('--------------------------------------------------------')
# Classify between the baseline and pain threshold
# axis 0 is columnwise && axis 1 is rowwise.
Combined = np.concatenate((level_one, level_three), axis=0)

# Encoding classes to integer levels
X,y = ClassifierModel.features_lables_split(Combined)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 7)

# Feature Scaling
X_train = ClassifierModel.NormalizeFeatures(X_train)
X_test = ClassifierModel.NormalizeFeatures(X_test)

# Fitting Kernel SVM to the Training set
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier = classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ClassifierModel.Visualize_CM(cm,[0,1])

# Evaluation
accuracy,average_precision,precision,recall,sensitivity,specificity,cramersV = ClassifierModel.EvaluateClassifier(y_test, y_pred)
print('Accuracy (T1vsT3): '+str(accuracy))
print('Average Precision (T1vsT3): '+str(average_precision))
print('Precision (T1vsT3): '+str(precision))
print('Recall (T1vsT3): '+str(recall))
print('Sensitivity (T1vsT3): '+str(sensitivity))
print('Specificity (T1vsT3): '+str(specificity))
print('Cramér''s V (T1vsT3): '+str(cramersV))

print('--------------------------------------------------------')
# Classify between the baseline and pain threshold
# axis 0 is columnwise && axis 1 is rowwise.
Combined = np.concatenate((level_one, level_four), axis=0)

# Encoding classes to integer levels
X,y = ClassifierModel.features_lables_split(Combined)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 7)

# Feature Scaling
X_train = ClassifierModel.NormalizeFeatures(X_train)
X_test = ClassifierModel.NormalizeFeatures(X_test)

# Fitting Kernel SVM to the Training set
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier = classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ClassifierModel.Visualize_CM(cm,[0,1])

# Evaluation
accuracy,average_precision,precision,recall,sensitivity,specificity,cramersV = ClassifierModel.EvaluateClassifier(y_test, y_pred)
print('Accuracy (T1vsT4): '+str(accuracy))
print('Average Precision (T1vsT4): '+str(average_precision))
print('Precision (T1vsT4): '+str(precision))
print('Recall (T1vsT4): '+str(recall))
print('Sensitivity (T1vsT4): '+str(sensitivity))
print('Specificity (T1vsT4): '+str(specificity))
print('Cramér''s V (T1vsT4): '+str(cramersV))

print('--------------------------------------------------------')
# Classify between the baseline and pain threshold
# axis 0 is columnwise && axis 1 is rowwise.
Combined = np.concatenate((level_two, level_three), axis=0)

# Encoding classes to integer levels
X,y = ClassifierModel.features_lables_split(Combined)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 7)

# Feature Scaling
X_train = ClassifierModel.NormalizeFeatures(X_train)
X_test = ClassifierModel.NormalizeFeatures(X_test)

# Fitting Kernel SVM to the Training set
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier = classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ClassifierModel.Visualize_CM(cm,[0,1])

# Evaluation
accuracy,average_precision,precision,recall,sensitivity,specificity,cramersV = ClassifierModel.EvaluateClassifier(y_test, y_pred)
print('Accuracy (T2vsT3): '+str(accuracy))
print('Average Precision (T2vsT3): '+str(average_precision))
print('Precision (T2vsT3): '+str(precision))
print('Recall (T2vsT3): '+str(recall))
print('Sensitivity (T2vsT3): '+str(sensitivity))
print('Specificity (T2vsT3): '+str(specificity))
print('Cramér''s V (T2vsT3): '+str(cramersV))

print('--------------------------------------------------------')
# Classify between the baseline and pain threshold
# axis 0 is columnwise && axis 1 is rowwise.
Combined = np.concatenate((level_two, level_four), axis=0)

# Encoding classes to integer levels
X,y = ClassifierModel.features_lables_split(Combined)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 7)

# Feature Scaling
X_train = ClassifierModel.NormalizeFeatures(X_train)
X_test = ClassifierModel.NormalizeFeatures(X_test)

# Fitting Kernel SVM to the Training set
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier = classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ClassifierModel.Visualize_CM(cm,[0,1])

# Evaluation
accuracy,average_precision,precision,recall,sensitivity,specificity,cramersV = ClassifierModel.EvaluateClassifier(y_test, y_pred)
print('Accuracy (T2vsT4): '+str(accuracy))
print('Average Precision (T2vsT4): '+str(average_precision))
print('Precision (T2vsT4): '+str(precision))
print('Recall (T2vsT4): '+str(recall))
print('Sensitivity (T2vsT4): '+str(sensitivity))
print('Specificity (T2vsT4): '+str(specificity))
print('Cramér''s V (T2vsT4): '+str(cramersV))

print('--------------------------------------------------------')
# Classify between the baseline and pain threshold
# axis 0 is columnwise && axis 1 is rowwise.
Combined = np.concatenate((level_three, level_four), axis=0)

# Encoding classes to integer levels
X,y = ClassifierModel.features_lables_split(Combined)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 7)

# Feature Scaling
X_train = ClassifierModel.NormalizeFeatures(X_train)
X_test = ClassifierModel.NormalizeFeatures(X_test)

# Fitting Kernel SVM to the Training set
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier = classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ClassifierModel.Visualize_CM(cm,[0,1])

# Evaluation
accuracy,average_precision,precision,recall,sensitivity,specificity,cramersV = ClassifierModel.EvaluateClassifier(y_test, y_pred)
print('Accuracy (T3vsT4): '+str(accuracy))
print('Average Precision (T3vsT4): '+str(average_precision))
print('Precision (T3vsT4): '+str(precision))
print('Recall (T3vsT4): '+str(recall))
print('Sensitivity (T3vsT4): '+str(sensitivity))
print('Specificity (T3vsT4): '+str(specificity))
print('Cramér''s V (T3vsT4): '+str(cramersV))