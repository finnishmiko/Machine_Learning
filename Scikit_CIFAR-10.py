##
## Classification testing with Scikit-learn:
## http://scikit-learn.org/stable/index.html
##
## CIFAR-10 dataset from:
## http://www.cs.toronto.edu/~kriz/cifar.html
##
## Python 3.5.1


import numpy as np
from sklearn import preprocessing, neighbors, svm
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

import pandas as pd

try:
    import _pickle as pickle
except:
    import cPickle as pickle

print ("Loading data")

def unpickle(file):
    fo = open(file, 'rb')
    #dict = pickle.load(fo)
    dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict

######## Load training data ##
dictionary = unpickle("data/cifar-10-batches-py/data_batch_1")
print (dictionary.keys())
## python27: ['data', 'labels', 'batch_label', 'filenames']
## Python35: [b'filenames', b'batch_label', b'data', b'labels']
##X = np.array(dictionary['data'])
##y = np.array(dictionary['labels'])
X = np.array(dictionary[b'data'])
y = np.array(dictionary[b'labels'])


dictionary = unpickle("data/cifar-10-batches-py/data_batch_2")
#print dictionary.keys()
## ['data', 'labels', 'batch_label', 'filenames']
X = np.append(X, np.array(dictionary[b'data']), axis=0)
y = np.append(y, np.array(dictionary[b'labels']))

dictionary = unpickle("data/cifar-10-batches-py/data_batch_3")
#print dictionary.keys()
## ['data', 'labels', 'batch_label', 'filenames']
X = np.append(X, np.array(dictionary[b'data']), axis=0)
y = np.append(y, np.array(dictionary[b'labels']))

dictionary = unpickle("data/cifar-10-batches-py/data_batch_4")
#print dictionary.keys()
## ['data', 'labels', 'batch_label', 'filenames']
X = np.append(X, np.array(dictionary[b'data']), axis=0)
y = np.append(y, np.array(dictionary[b'labels']))

dictionary = unpickle("data/cifar-10-batches-py/data_batch_5")
#print dictionary.keys()
## ['data', 'labels', 'batch_label', 'filenames']
X = np.append(X, np.array(dictionary[b'data']), axis=0)
y = np.append(y, np.array(dictionary[b'labels']))

print (X.shape, y.shape)
del dictionary
print ("Loading ready")

    
######## Scale input data to fit to the training algorithm ##
X_train = preprocessing.scale(X)
y_train = y
#X_train = preprocessing.scale(X[:1000])
#y_train = y[:1000]

######## separate data to train and test chunks ##
##X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

######## Load saved classifier ##
#pickle_in = open('data/clf-SVM_SVC-CIFAR-10.pickle', 'rb')
#clf = pickle.load(pickle_in)


######## select classifier ##
#clf = neighbors.KNeighborsClassifier()
#clf = svm.SVC(max_iter=15)
clf = svm.SVC(verbose=1)
#clf = svm.SVC(kernel='poly')
#clf = svm.NuSVC()
#clf = RandomForestClassifier(n_estimators=200)
#clf = SGDClassifier(loss="hinge", penalty="l2")

clf.fit(X_train, y_train)
print("Training set score: %f" % clf.score(X_train, y_train))

######## Load test data ##
dictionary = unpickle("data/cifar-10-batches-py/test_batch")
#print dictionary.keys()
## ['data', 'labels', 'batch_label', 'filenames']
X_test = preprocessing.scale(np.array(dictionary[b'data']))
y_test = np.array(dictionary[b'labels'])
print (X_test.shape, y_test.shape)


print("Test set score: %f" % clf.score(X_test, y_test))

######## save training results to pickle file: ###
with open('data/clf--CIFAR-10.pickle', 'wb') as f:
    pickle.dump(clf, f)


######## Test with few samples that are not in the training set
example_measures = np.array([X_test[100], X_test[200]])
prediction = clf.predict(example_measures)
print(prediction)
print (y_test[100], y_test[200])

