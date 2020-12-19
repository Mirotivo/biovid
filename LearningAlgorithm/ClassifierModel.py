def ImportDataSet():
    import pandas as pd
    dataset = pd.read_csv('features/Table_Step2_159Features-85Subs-5Levels-z.csv') 
    Combined = dataset.iloc[:,7:].values
    return Combined

def ImputeDataSet(Combined):
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(Combined[:,1:])
    return imputer.transform(Combined[:,1:])

def SeparateEachLevel(Combined):
    return [i for i in Combined if i[0] == 0.1],[i for i in Combined if i[0] == 1],[i for i in Combined if i[0] == 2],[i for i in Combined if i[0] == 3],[i for i in Combined if i[0] == 4]

def features_lables_split(Combined):
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    return Combined[:,1:],labelencoder.fit_transform(Combined[:,0])

def NormalizeFeatures(X):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    return sc.fit_transform(X)

def Visualize_CM(cm,classes):
    import numpy as np
    import matplotlib.pyplot as plt
    import itertools
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return

def cramers_stat(confusion_matrix):  
    import numpy as np
    import scipy.stats as ss
    chi2 = ss.chi2_contingency(confusion_matrix)[0]  
    n = confusion_matrix.sum().sum()  
    return np.sqrt(chi2 / (n*(min(confusion_matrix.shape)-1)))  

def EvaluateClassifier(y_test, y_pred):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import recall_score
    '''
    Binary:
        TP cm[1,1]
        TN cm[0,0]
        FP cm[1,0]
        FN cm[0,1]
    M-ary:
        TP cm[1,1] cm[2,2] cm[3,3] cm[4,4] 
        TN cm[0,0]
        FP cm[1,0]
        FN cm[0,1]
    '''
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    average_precision = 0#average_precision_score(y_test, y_pred)
    precision = 0#precision_score(y_test, y_pred, average=None)
    recall = 0#recall_score(y_test, y_pred, average=None)
    #TP/P = TP/(TP+FN)
    sensitivity = 0#cm[0,0]/(cm[0,0]+cm[0,1])
    #TN/N = TN/(TN+FP)
    specificity = 0#cm[1,1]/(cm[1,0]+cm[1,1])
    cramersV = 0#cramers_stat(cm)
    return accuracy,average_precision,precision,recall,sensitivity,specificity,cramersV



def Feature_Selection():
    import numpy as np
    feat_corr = np.zeros(shape=(160))
    corr = 0
    for ind in range(1,160):
        corr = np.corrcoef(Combined[:,1], Combined[:,ind])[0, 1]
        if (corr > 0.95 or corr < -0.95):
            feat_corr[ind] = corr

def Precision_Recall_Curve():
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(average_precision))

def Receiver_Operating_Characteristic():
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    from sklearn.multiclass import OneVsRestClassifier
    y = label_binarize(y, classes=[0, 1])
    n_classes = y.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=7)
    random_state = np.random.RandomState(0)
    classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()