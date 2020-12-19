import os
import scipy
import sys
from pipeline_preprocessing import pipeline_preprocessing
import pandas as pd
import numpy as np
from helpers import from_csv_to_nparray
from csv_helpers import csv_helpers
from sklearn.preprocessing import Imputer



   
   


def main():
        import matplotlib.pyplot as plot
        import sklearn.ensemble as ensemble
        import pandas as pd
        from sklearn.preprocessing import Imputer 
        
        
        data= pd.read_csv('D:/ASU/Thieses/Pain/BioVid/PartB/biosignals_featured.csv')
        
        X=data.iloc[:,:-1].values
        
        Y=data.iloc[:,-1].values
        

        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(X)
        y = sc_y.fit_transform(Y)
        
        
        from sklearn.cross_validation import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
        
        
        
        
        # Fitting SVR to the dataset
        from sklearn.svm import SVR
        regressor = SVR(kernel = 'rbf')
        regressor.fit(X_train, y_train)
        X_predict=regressor.predict(X_test)
        
        
        
        
        
        
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        
        # Predicting the Test set results
        y_pred = regressor.predict(X_test)
        
        
        
        # Fitting SVM to the Training set
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(X_train, y_train)
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)

        
        



main()  


def _assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype) 

