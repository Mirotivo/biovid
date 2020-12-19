# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 13:29:13 2017

@author: amirs
"""

# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv('features/Table_Step2_159Features-85Subs-5Levels-z.csv')
all = dataset.iloc[:, :].values


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(all[:,1:])
all[:,1:] = imputer.transform(all[:,1:])

level_zero_one=([i for i in all if i[0] == 'level_zero_one'])
level_one=[i for i in all if i[0] == 'level_one']
level_two=[i for i in all if i[0] == 'level_two']
level_three=[i for i in all if i[0] == 'level_three']
level_four=[i for i in all if i[0] == 'level_four']

all = np.concatenate((level_zero_one, level_one), axis=0)



from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Y_1 = LabelEncoder()
all[:,0]=labelencoder_Y_1.fit_transform(all[:,0])

Y_label=all[:,0]

y = keras.utils.to_categorical(Y_label, num_classes=2)
y=y[:,1:]


X = all[:, 2:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 7)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


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
# Adding the output layer
classifier.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model
scores =classifier.evaluate(X_test, y_test, batch_size=384, verbose=1)
print("%s: %.2f%%" % (classifier.metrics_names[1], scores[1]*100))


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)