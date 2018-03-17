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

BN_FRAME_WISE = False
CLIP_GRADIENTS = False

# Import MNIST data (Numpy format)
MNIST = input_data.read_data_sets("/tmp/data/")


def get_bn_rnn(inputs, is_training=True):
  # Add a batch normalization layer after each
  layer_input = inputs
  layer_output = None
  cells = []
  for layer in range(NUM_LAYERS):
    cell = IndRNNCell(NUM_UNITS, recurrent_max_abs=RECURRENT_MAX)
    cells.append(cell)
    layer_output, _ = tf.nn.dynamic_rnn(cell, layer_input,
                                        dtype=tf.float32,
                                        scope="rnn%d" % layer)

    # For frame-wise normalization, put the time steps dimension into the last
    # dimension so that it doesn't get normalized
    if BN_FRAME_WISE:
      batch_size = tf.shape(layer_output)[0]
      layer_output = tf.reshape(layer_output,
                                [batch_size, TIME_STEPS * NUM_UNITS])

    layer_output = tf.contrib.layers.batch_norm(layer_output,
                                                is_training=is_training)
    if BN_FRAME_WISE:
      layer_output = tf.reshape(layer_output,
                                [batch_size, TIME_STEPS, NUM_UNITS])

    if is_training:
      # Tie the BN population statistics updates to the output op
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        layer_output = tf.identity(layer_output)
  return layer_output


def build(inputs, labels):
  # Build the graph
  output = get_bn_rnn(inputs)
  last = output[:, -1, :]

  weight = tf.get_variable("softmax_weight", shape=[NUM_UNITS, NUM_CLASSES])
  bias = tf.get_variable("softmax_bias", shape=[NUM_CLASSES],
                         initializer=tf.constant_initializer(0.1))
  logits = tf.matmul(last, weight) + bias

  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels))

  global_step = tf.Variable(0, trainable=False)
  learning_rate = tf.train.exponential_decay(LEARNING_RATE_INIT, global_step,
                                             LEARNING_RATE_DECAY_STEPS, 0.1,
                                             staircase=True)
  optimizer = tf.train.AdamOptimizer(learning_rate)

  if CLIP_GRADIENTS:
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimize = optimizer.apply_gradients(zip(gradients, variables))
  else:
    optimize = optimizer.minimize(loss, global_step=global_step)

  correct_pred = tf.equal(tf.argmax(logits, 1, output_type=tf.int32), labels)
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  return loss, accuracy, optimize


def main():
  sess = tf.Session()

  iterator = get_dataset_iterator()
  inputs_ph = tf.placeholder(tf.float32, [None, TIME_STEPS])
  labels_ph = tf.placeholder(tf.int8, [None])

  # Initialize the iterator
  sess.run(iterator.initializer, feed_dict={inputs_ph: MNIST.train.images,
    labels_ph: MNIST.train.labels})

  # Neural Net Input
  inputs, labels = iterator.get_next()
  inputs = tf.expand_dims(inputs, -1)  # expand to [BATCH_SIZE, TIME_STEPS, 1]
  labels = tf.cast(labels, tf.int32)

  # From https://github.com/cooijmanstim/recurrent-batch-normalization
  inputs = gaussian_noise_layer(inputs, 0.1)

  loss_op, accuracy_op, train_op = build(inputs, labels)

  # Train the model
  sess.run(tf.global_variables_initializer())
  step = 0
  while True:
    losses = []
    accuracies = []
    for _ in range(10):
      # Execute one training step
      try:
        loss, accuracy, _ = sess.run([loss_op, accuracy_op, train_op])
      except tf.errors.OutOfRangeError:
        # Reload the iterator when it reaches the end of the dataset
        sess.run(iterator.initializer, {
          inputs_ph: MNIST.train.images,
          labels_ph: MNIST.train.labels})
        loss, accuracy, _ = sess.run([loss_op, accuracy_op, train_op])

      losses.append(loss)
      accuracies.append(accuracy)
      step += 1

    print("Step {} Loss {} Acc {}"
          .format(step, np.mean(losses), np.mean(accuracies)))


def gaussian_noise_layer(input_layer, std):
  noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std,
                           dtype=tf.float32)
  return input_layer + noise


def get_dataset_iterator():
  # Create a dataset tensor from the images and the labels
  dataset = tf.data.Dataset.from_tensor_slices(
      (MNIST.train.images, MNIST.train.labels))
  # Create batches of data
  dataset = dataset.batch(BATCH_SIZE)
  # Create an iterator, to go over the dataset
  return dataset.make_initializable_iterator()


if __name__ == "__main__":
  main()
