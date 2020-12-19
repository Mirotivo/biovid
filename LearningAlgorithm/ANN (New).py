import ClassifierModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Importing the dataset
dataset = pd.read_csv('features/Table_Step2_159Features-85Subs-5Levels-z.csv')
Combined = dataset.iloc[:,7:].values

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

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import metrics

from sklearn.decomposition import PCA
pca = PCA(n_components=70)# adjust yourself
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 35, activation = 'relu',kernel_initializer="uniform", input_dim = 70))

# Adding the second hidden layer
#.add(Dense(output_dim = 80, init = 'uniform', activation = 'relu'))



#classifier.add(Dense(units = 40, activation = 'relu',kernel_initializer="uniform"))
#classifier.add(Dense(units = 50, activation = 'relu',kernel_initializer="uniform"))


#classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 10, activation = 'relu',kernel_initializer="uniform"))
#classifier.add(Dense(units = 15, activation = 'relu',kernel_initializer="uniform"))
# Adding the output layer
classifier.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ClassifierModel.Visualize_CM(cm,[0,1])

# Evaluation
accuracy,average_precision,precision,recall,sensitivity,specificity,cramersV = ClassifierModel.EvaluateClassifier(y_test, y_pred)
print('Accuracy (BvsT1): '+str(accuracy))
print('Precision (BvsT1): '+str(precision))
print('Recall (BvsT1): '+str(recall))
print('Sensitivity (BvsT1): '+str(sensitivity))
print('Specificity (BvsT1): '+str(specificity))

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

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import metrics


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 120, activation = 'relu',kernel_initializer="uniform", input_dim = 159))

# Adding the second hidden layer
#.add(Dense(output_dim = 80, init = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 90, activation = 'relu',kernel_initializer="uniform"))

classifier.add(Dense(units = 40, activation = 'relu',kernel_initializer="uniform"))
#classifier.add(Dense(units = 50, activation = 'relu',kernel_initializer="uniform"))


#classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 30, activation = 'relu',kernel_initializer="uniform"))
#classifier.add(Dense(units = 15, activation = 'relu',kernel_initializer="uniform"))
# Adding the output layer
classifier.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ClassifierModel.Visualize_CM(cm,[0,1])

# Evaluation
accuracy,average_precision,precision,recall,sensitivity,specificity,cramersV = ClassifierModel.EvaluateClassifier(y_test, y_pred)
print('Accuracy (BvsT2): '+str(accuracy))
print('Precision (BvsT2): '+str(precision))
print('Recall (BvsT2): '+str(recall))
print('Sensitivity (BvsT2): '+str(sensitivity))
print('Specificity (BvsT2): '+str(specificity))


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

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import metrics


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 120, activation = 'relu',kernel_initializer="uniform", input_dim = 159))

# Adding the second hidden layer
#.add(Dense(output_dim = 80, init = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 90, activation = 'relu',kernel_initializer="uniform"))

classifier.add(Dense(units = 40, activation = 'relu',kernel_initializer="uniform"))
#classifier.add(Dense(units = 50, activation = 'relu',kernel_initializer="uniform"))


#classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 30, activation = 'relu',kernel_initializer="uniform"))
#classifier.add(Dense(units = 15, activation = 'relu',kernel_initializer="uniform"))
# Adding the output layer
classifier.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ClassifierModel.Visualize_CM(cm,[0,1])

# Evaluation
accuracy,average_precision,precision,recall,sensitivity,specificity,cramersV = ClassifierModel.EvaluateClassifier(y_test, y_pred)
print('Accuracy (BvsT3): '+str(accuracy))
print('Precision (BvsT3): '+str(precision))
print('Recall (BvsT3): '+str(recall))
print('Sensitivity (BvsT3): '+str(sensitivity))
print('Specificity (BvsT3): '+str(specificity))



# Classify between the baseline and pain threshold
# axis 0 is columnwise && axis 1 is rowwise.
Combined = np.concatenate((level_zero_one, level_four), axis=0)

# Encoding classes to integer levels
X,y = ClassifierModel.features_lables_split(Combined)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 7)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import metrics


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 120, activation = 'relu',kernel_initializer="uniform", input_dim = 159))

# Adding the second hidden layer
#.add(Dense(output_dim = 80, init = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 90, activation = 'relu',kernel_initializer="uniform"))

classifier.add(Dense(units = 40, activation = 'relu',kernel_initializer="uniform"))
#classifier.add(Dense(units = 50, activation = 'relu',kernel_initializer="uniform"))


#classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 30, activation = 'relu',kernel_initializer="uniform"))
#classifier.add(Dense(units = 15, activation = 'relu',kernel_initializer="uniform"))
# Adding the output layer
classifier.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Fitting Kernel SVM to the Training set
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ClassifierModel.Visualize_CM(cm,[0,1])

# Evaluation
accuracy,average_precision,precision,recall,sensitivity,specificity,cramersV = ClassifierModel.EvaluateClassifier(y_test, y_pred)
print('Accuracy (BvsT4): '+str(accuracy))
print('Precision (BvsT4): '+str(precision))
print('Recall (BvsT4): '+str(recall))
print('Sensitivity (BvsT4): '+str(sensitivity))
print('Specificity (BvsT4): '+str(specificity))


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
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ClassifierModel.Visualize_CM(cm,[0,1,2])

# Evaluation
accuracy,average_precision,precision,recall,sensitivity,specificity,cramersV = ClassifierModel.EvaluateClassifier(y_test, y_pred)
print('Accuracy (BvsT1vsT4): '+str(accuracy))
print('Precision (BvsT1vsT4): '+str(precision))
print('Recall (BvsT1vsT4): '+str(recall))
print('Sensitivity (BvsT1vsT4): '+str(sensitivity))
print('Specificity (BvsT1vsT4): '+str(specificity))




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
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ClassifierModel.Visualize_CM(cm,[0,1,2,3,4])

# Evaluation
accuracy,average_precision,precision,recall,sensitivity,specificity,cramersV = ClassifierModel.EvaluateClassifier(y_test, y_pred)
print('Accuracy (BvsT1vsT2vsT3vsT4): '+str(accuracy))
print('Precision (BvsT1vsT2vsT3vsT4): '+str(precision))
print('Recall (BvsT1vsT2vsT3vsT4): '+str(recall))
print('Sensitivity (BvsT1vsT2vsT3vsT4): '+str(sensitivity))
print('Specificity (BvsT1vsT2vsT3vsT4): '+str(specificity))
