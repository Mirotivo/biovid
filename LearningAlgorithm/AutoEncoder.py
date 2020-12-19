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


from keras.layers import Input, Dense
from keras.models import Model

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(159,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(159, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))

encoded_imgs = encoder.predict(X_test)
decoded_imgs = decoder.predict(encoded_imgs)




y_pred = encoder.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ClassifierModel.Visualize_CM(cm,[0,1])

# Evaluation
accuracy,precision,recall,sensitivity,specificity = ClassifierModel.EvaluateClassifier(y_test, y_pred)
print('Accuracy (BvsT1): '+str(accuracy))
print('Precision (BvsT1): '+str(precision))
print('Recall (BvsT1): '+str(recall))
print('Sensitivity (BvsT1): '+str(sensitivity))
print('Specificity (BvsT1): '+str(specificity))


