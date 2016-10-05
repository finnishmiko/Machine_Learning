import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
# one_hot means that from the 10 outputs only one will be selected at a time

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder(tf.float32, [None, 784]) # Here None means any size is ok. 784 = 28x28 pixels
y = tf.placeholder(tf.float32, [None, 10]) # each row is a one-hot 10-dimensional vector

def neural_network_model(data):
    # initialize weights and biases (these are needed in case all inputs are zeros)
    # with random numbers
    hidden_1_layer = {'weights':tf.Variable(tf.truncated_normal([784, n_nodes_hl1], stddev=0.1)),
                      'biases':tf.Variable(tf.constant(0.1, shape=[n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2], stddev=0.1)),
                      'biases':tf.Variable(tf.constant(0.1, shape=[n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl2, n_nodes_hl3], stddev=0.1)),
                      'biases':tf.Variable(tf.constant(0.1, shape=[n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl3, n_classes], stddev=0.1)),
                    'biases':tf.Variable(tf.constant(0.1, shape=[n_classes])),}

    # y = x * Weights + Bias
    # apply the ReLU function
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    # cost or loss or cross_entropy function
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # Training the model is done by repeatedly runnint this optimizer operation
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost) # select optimization algorithm
    
    # Test model accuracy by checking how many predictions matches to their labels.
    # argmax gives the index of highest entry along some axis. Output is list of booleans
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    # Cast to floating point numbers and take the mean
    # F.ex. [True, False, True, True] --> [1, 0, 1, 1] --> 0.75
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    hm_epochs = 10 # Select how many training cycles
    with tf.Session() as sess: # Open and close session with with syntax
        sess.run(tf.initialize_all_variables())

        # Training cycles
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)): # train all batches
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                # Fit the model and calculate cost
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c # calculate cost per epoch

            # Print the cost to see the training is improving
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        
        # Calculate the accuracy
        print('Accuracy:',accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)
