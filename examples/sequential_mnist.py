"""Module using IndRNNCell to solve the Sequential MNIST problem

The problem is described in https://arxiv.org/abs/1803.04831. The
hyper-parameters are taken from that paper as well.
"""
import itertools
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

BATCH_SIZE = 50
BN_FRAME_WISE = False
CLIP_GRADIENTS = False

# Import MNIST data (Numpy format)
MNIST = input_data.read_data_sets("/tmp/data/")


def get_bn_rnn(inputs, training):
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

    layer_output = tf.layers.batch_normalization(layer_output,
                                                 training=training,
                                                 scale=True)
    # Undo the reshape above
    if BN_FRAME_WISE:
      layer_output = tf.reshape(layer_output,
                                [batch_size, TIME_STEPS, NUM_UNITS])

    # Tie the BN population statistics updates to the layer_output op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      layer_output = tf.identity(layer_output)

  return layer_output


def build(inputs, labels):
  # Build the graph
  training = tf.placeholder_with_default(True, [])
  output = get_bn_rnn(inputs, training)
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
  data_handle = tf.placeholder(tf.string, shape=[])

  # Neural Net Input
  main_iter, training_iter, validation_iter = get_data(data_handle)
  # Generate handles for each iterator
  training_handle = sess.run(training_iter.string_handle())
  validation_handle = sess.run(validation_iter.string_handle())

  inputs, labels = main_iter.get_next()
  # TODO: Put these into the dataset preprocessing
  inputs = tf.expand_dims(inputs, -1)  # expand to [BATCH_SIZE, TIME_STEPS, 1]
  labels = tf.cast(labels, tf.int32)

  loss_op, accuracy_op, train_op = build(inputs, labels)

  # Train the model
  sess.run(tf.global_variables_initializer())
  for step in itertools.count():
    # Execute one training step
    loss, accuracy, _ = sess.run([loss_op, accuracy_op, train_op],
                                 feed_dict={data_handle: training_handle})

    if step % 100 == 1:
      print('Step {} Loss {} Acc {}'.format(step+1, loss, accuracy))

    if step % 1000 == 0:
      # Validation
      losses, accuracies = [], []
      for _ in range(1000):
        valid_loss, valid_accuracy = sess.run(
            [loss_op, accuracy_op],
            feed_dict={data_handle: validation_handle})
        losses.append(valid_loss)
        accuracies.append(valid_accuracy)
      print('Step {} valid_loss {} valid_acc {}'
            .format(step+1, np.mean(losses), np.mean(accuracies)))


def add_noise(inputs, labels):
  # Values taken https://github.com/cooijmanstim/recurrent-batch-normalization
  inputs = inputs + tf.random_normal([], mean=0.0, stddev=0.1, dtype=tf.float32)
  return inputs, labels


def get_data(handle):
  training_dataset = tf.data.Dataset.from_tensor_slices(
      (MNIST.train.images, MNIST.train.labels))
  # Apply random perturbations to the training data
  training_dataset.map(add_noise)
  training_dataset = training_dataset.repeat()
  training_dataset = training_dataset.batch(BATCH_SIZE)

  # Create the validation dataset
  validation_dataset = tf.data.Dataset.from_tensor_slices(
      (MNIST.validation.images, MNIST.validation.labels))
  validation_dataset = validation_dataset.repeat()
  validation_dataset = validation_dataset.batch(BATCH_SIZE)

  # Create an iterator for switching between datasets
  iterator = tf.data.Iterator.from_string_handle(
      handle, training_dataset.output_types, training_dataset.output_shapes)

  # Create iterators for each dataset that the main iterator can use for the
  # next element
  training_iterator = training_dataset.make_one_shot_iterator()
  validation_iterator = validation_dataset.make_one_shot_iterator()
  return iterator, training_iterator, validation_iterator


if __name__ == "__main__":
  main()
