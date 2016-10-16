# Machine_Learning
## MNIST
- [Dataset](http://yann.lecun.com/exdb/mnist/)
- Deep Neural Network with [Tensorflow](https://www.tensorflow.org/)
  * Test setup: Layer sizes 784, 500, 500, 500 and 10
  * Accuracy after 10 epochs, three runs: 0.9772, 0.9759, 0.9744
  * Run time few minutes with Azure Ubuntu virtual machine type DS1_V2 (1 core, 3.5 GB)
- Deep Convolutional Neural Network with [Tensorflow](https://www.tensorflow.org/)
  * Accuracy after 10 epochs: 0.9910, 0.9896
  * Run time was 62 minutes with Azure Ubuntu virtual machine type DS1_V2 (1 core, 3.5 GB) and 67 minutes when accuracy was calculated after every epoch
- [Scikit-learn](http://scikit-learn.org/stable/index.html) (ver 0.18) Multi-Layer Perceptron (MLP)
  * Hidden layers: 100, 100
  * Max iterations 400, actual 304 with tol=1e-4
  * Accuracy: test set 0.9758 and training set 0.9978
  * Run time 7 min
- [Scikit-learn](http://scikit-learn.org/stable/index.html) (ver 0.18) Support Vector Machines (SVM)
  * Accuracy: test set 0.9446, training set 0.9430
  * Run time 37 min


- Wep app with Python Flask to recognize drawn digits
  * This version uses Scikit-learn's MLP-classifier - does not work very reliably with drawn digits 

## CIFAR-10
- [Dataset](http://www.cs.toronto.edu/~kriz/cifar.html)
- [Scikit-learn](http://scikit-learn.org/stable/index.html) SVM accuracy: Training set 0.713 and Test set 0.547


## Results for reference
Collection of [classification results](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html) for reference.
