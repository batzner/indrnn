"""Module using IndRNNCell to solve the Sequential MNIST problem

The problem is described in https://arxiv.org/abs/1803.04831. The
hyper-parameters are taken from that paper as well.
"""
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops.rnn_cell_impl import MultiRNNCell

from ind_rnn_cell import IndRNNCell

# Parameters taken from https://arxiv.org/abs/1803.04831
TIME_STEPS = 784
NUM_UNITS = 128
LEARNING_RATE_INIT = 0.0002
LEARNING_RATE_DECAY_STEPS = 600000
NUM_LAYERS = 6
RECURRENT_MAX = pow(2, 1 / TIME_STEPS)
NUM_CLASSES = 10

# Parameters taken from https://arxiv.org/abs/1511.06464
BATCH_SIZE = 50


def main():
  sess = tf.Session()
  inputs, labels = get_mnist_inputs(sess)

  # Build the graph
  cell = MultiRNNCell([IndRNNCell(NUM_UNITS, recurrent_max_abs=RECURRENT_MAX)
                       for _ in range(NUM_LAYERS)])

  output, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
  last = output[:, -1, :]

  weight = tf.get_variable("softmax_weight", shape=[NUM_UNITS, NUM_CLASSES])
  bias = tf.get_variable("softmax_bias", shape=[NUM_CLASSES],
                         initializer=tf.constant_initializer(0.1))
  logits = tf.matmul(last, weight) + bias

  loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=labels))

  global_step = tf.Variable(0, trainable=False)
  learning_rate = tf.train.exponential_decay(LEARNING_RATE_INIT, global_step,
                                             LEARNING_RATE_DECAY_STEPS, 0.1,
                                             staircase=True)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  optimize = optimizer.minimize(loss_op, global_step=global_step)

  correct_pred = tf.equal(tf.argmax(logits, 1), labels)
  accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

  # Train the model
  sess.run(tf.global_variables_initializer())
  step = 0
  while True:
    losses = []
    accuracies = []
    for _ in range(10):
      loss, accuracy, _ = sess.run([loss_op, accuracy_op, optimize])
      losses.append(loss)
      accuracies.append(accuracy)
      step += 1

    #
    print("Step {} Loss {} Acc {}"
          .format(step, np.mean(losses), np.mean(accuracy)))


def get_mnist_inputs(sess):
  # From https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5_DataManagement/tensorflow_dataset_api.py
  # Import MNIST data (Numpy format)
  mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

  # Create a dataset tensor from the images and the labels
  dataset = tf.data.Dataset.from_tensor_slices(
    (mnist.train.images, mnist.train.labels))
  # Create batches of data
  dataset = dataset.batch(BATCH_SIZE)
  # Create an iterator, to go over the dataset
  iterator = dataset.make_initializable_iterator()
  # It is better to use 2 placeholders, to avoid to load all data into memory,
  # and avoid the 2Gb restriction length of a tensor.
  _data = tf.placeholder(tf.float32, [None, TIME_STEPS])
  _labels = tf.placeholder(tf.float32, [None, NUM_CLASSES])
  # Initialize the iterator
  sess.run(iterator.initializer, feed_dict={_data: mnist.train.images,
                                            _labels: mnist.train.labels})
  # Neural Net Input
  inputs, targets = iterator.get_next()
  return tf.expand_dims(inputs, -1), tf.argmax(targets, axis=1)


if __name__ == "__main__":
  main()
