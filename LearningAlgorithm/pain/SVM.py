# Importing the dataset
import pandas as pd
dataset = pd.read_csv('features/Table_Step2_159Features-85Subs-5Levels-z.csv')
YX = dataset.iloc[:, 7:].values

# Imputing the missing features and replacing it with the mean value excluding the first column
# axis 0 is columnwise && axis 1 is rowwise.
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(YX[:,1:])
YX[:,1:] = imputer.transform(YX[:,1:])

# Separating each level
level_zero_one=([i for i in YX if i[0] == 'level_zero_one'])
level_one=[i for i in YX if i[0] == 'level_one']
level_two=[i for i in YX if i[0] == 'level_two']
level_three=[i for i in YX if i[0] == 'level_three']
level_four=[i for i in YX if i[0] == 'level_four']

import numpy as np
YX = np.concatenate((level_zero_one, level_one), axis=0)



from sklearn.preprocessing import LabelEncoder
labelencoder_Y_1 = LabelEncoder()
YX[:,0]=labelencoder_Y_1.fit_transform(YX[:,0])

Y_label=YX[:,0]


import keras
y = keras.utils.to_categorical(Y_label, num_classes=2)
y=y[:,1:]


X = YX[:, 2:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 7)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)