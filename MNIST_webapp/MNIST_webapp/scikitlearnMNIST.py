try:
    import _pickle as pickle
except:
    import cPickle as pickle

###### Load saved classifier ######
pickle_in = open('/workspaces/Machine_Learning/data/clf-MPL-MNIST-layer100_100-iter400.pickle', 'rb')
# pickle_in = open('/workspaces/Machine_Learning/MNIST_webapp/MNIST_webapp/clf-MPL-MNIST-layer100_100-iter400.pickle', 'rb')
clf = pickle.load(pickle_in)


## Test with samples that are not in the training set
def numberpredict(example):
    prediction = clf.predict(example)
    return int(prediction)

