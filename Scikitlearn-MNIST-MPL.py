import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D

# from sklearn.datasets import fetch_mldata
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import numpy as np
import random
try:
    import _pickle as pickle
except:
    import cPickle as pickle

# mnist = fetch_mldata("MNIST original")
mnist = fetch_openml('mnist_784', version=1, cache=True)
mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings

# rescale the data, use the traditional train/test split
X, y = mnist.data / 255., mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

###### Load saved classifier ######
##pickle_in = open('data/clf-MPL-MNIST-layer100_100-iter400.pickle', 'rb')
##clf = pickle.load(pickle_in)

###### Train classifier ######
clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1)
##clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
##                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
##                    learning_rate_init=.1)

clf.fit(X_train, y_train)

print("Training set score: %f" % clf.score(X_train, y_train))
print("Test set score: %f" % clf.score(X_test, y_test))

###### save training results to pickle file: ######
with open('data/clf-MPL-MNIST-layer100_100-iter400.pickle', 'wb') as f:
    pickle.dump(clf, f)


######## Test with samples that are not in the training set
def numberpredict(example):
    prediction = clf.predict(example)
    #print(prediction)
    #plt.imshow(example.reshape(28,28), cmap='Greys')
    #plt.show()
    return int(prediction)
sample = random.randint(0,9999) # 200 was not recognized with 97.1% accuracy
print(sample)
print (y_test[sample])
example_measure = np.array([X_test[sample]])   
print(numberpredict(example_measure))


######## Visualize weights ######
##fig, axes = plt.subplots(4, 4)
### use global min / max to ensure all weights are shown on the same scale
##vmin, vmax = clf.coefs_[0].min(), clf.coefs_[0].max()
##for coef, ax in zip(clf.coefs_[0].T, axes.ravel()):
##    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
##               vmax=.5 * vmax)
##    ax.set_xticks(())
##    ax.set_yticks(())
##
##plt.show()

