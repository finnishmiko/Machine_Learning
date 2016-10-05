import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
# one_hot means that from the 10 outputs only one will be selected at a time

batch_size = 50

x = tf.placeholder(tf.float32, [None, 784]) # Here None means any size is ok. 784 = 28x28 pixels
y = tf.placeholder(tf.float32, [None, 10]) # each row is a one-hot 10-dimensional vector

def neural_network_model(data):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    
    x_image = tf.reshape(data, [-1,28,28,1]) # Reshape input data to 4d tensor

    # convolve reshaped data with the weight tensor, add the bias, apply the ReLU function
    # and finally max pool
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    output = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return output, keep_prob

# Helper functions to initialize variables with small amount of noise
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Convolution and Pooling functions
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def train_neural_network(x):
    prediction, keep_prob = neural_network_model(x)
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
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y, keep_prob: 0.5})
                epoch_loss += c # calculate cost per epoch

            # Print the cost to see the training is improving
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        
        # Calculate the accuracy
        print('Accuracy:',accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob: 1.0}))

train_neural_network(x)
