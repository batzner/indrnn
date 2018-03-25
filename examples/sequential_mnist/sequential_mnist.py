"""Module using IndRNNCell to solve the Sequential MNIST problem

The problem is described in https://arxiv.org/abs/1803.04831. The
hyper-parameters are taken from that paper as well.
"""
import itertools
from datetime import datetime

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from ind_rnn_cell import IndRNNCell

# Parameters taken from https://arxiv.org/abs/1803.04831
TIME_STEPS = 784
NUM_UNITS = 128
LEARNING_RATE_INIT = 0.0002
LEARNING_RATE_DECAY_STEPS = 600000
NUM_LAYERS = 6
RECURRENT_MAX = pow(2, 1 / TIME_STEPS)
LAST_LAYER_LOWER_BOUND = pow(0.5, 1 / TIME_STEPS)
NUM_CLASSES = 10

BATCH_SIZE = 32
BN_FRAME_WISE = False
BN_MOMENTUM = 0.99 if BN_FRAME_WISE else 0.9
CLIP_GRADIENTS = True


def get_bn_rnn(inputs, training):
  # Add a batch normalization layer after each
  layer_input = inputs
  layer_output = None
  input_init = tf.random_uniform_initializer(-0.001, 0.001)
  for layer in range(1, NUM_LAYERS + 1):
    recurrent_init_lower = 0 if layer < NUM_LAYERS else LAST_LAYER_LOWER_BOUND
    recurrent_init = tf.random_uniform_initializer(recurrent_init_lower,
                                                   RECURRENT_MAX)

    cell = IndRNNCell(NUM_UNITS,
                      recurrent_max_abs=RECURRENT_MAX,
                      input_kernel_initializer=input_init,
                      recurrent_kernel_initializer=recurrent_init)
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
                                                 momentum=BN_MOMENTUM)
    # Undo the reshape above
    if BN_FRAME_WISE:
      layer_output = tf.reshape(layer_output,
                                [batch_size, TIME_STEPS, NUM_UNITS])

    # Tie the BN population statistics updates to the layer_output op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      layer_output = tf.identity(layer_output)
    layer_input = layer_output

  return layer_output


def build(inputs, labels):
  # Build the graph
  is_training = tf.placeholder_with_default(True, [])
  output = get_bn_rnn(inputs, is_training)
  last = output[:, -1, :]

  weight = tf.get_variable("softmax_weight", shape=[NUM_UNITS, NUM_CLASSES],
                           initializer=tf.glorot_uniform_initializer())
  bias = tf.get_variable("softmax_bias", shape=[NUM_CLASSES],
                         initializer=tf.constant_initializer(0.))
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
  return loss, accuracy, optimize, is_training


def main():
  sess = tf.Session()

  # Import MNIST data (Numpy format)
  mnist = input_data.read_data_sets("/tmp/data/")

  # Create placeholders for feeding the data
  data_handle = tf.placeholder(tf.string, shape=[], name="data_handle")
  all_inputs_ph = tf.placeholder(mnist.train.images.dtype, [None, 784],
                                 name="all_inputs")
  all_labels_ph = tf.placeholder(mnist.train.labels.dtype, [None],
                                 name="all_labels")

  main_iter, train_iter, valid_iter = get_iterators(data_handle,
                                                    all_inputs_ph,
                                                    all_labels_ph)
  sess.run(train_iter.initializer, feed_dict={
    all_inputs_ph: mnist.train.images,
    all_labels_ph: mnist.train.labels})

  # Generate handles for each iterator
  train_handle = sess.run(train_iter.string_handle())
  valid_handle = sess.run(valid_iter.string_handle())

  inputs, labels = main_iter.get_next()
  loss_op, accuracy_op, train_op, train_switch = build(inputs, labels)

  # Train the model
  sess.run(tf.global_variables_initializer())

  train_losses = []
  train_accuracies = []
  for step in itertools.count():
    # Execute one training step
    loss, accuracy, _ = sess.run([loss_op, accuracy_op, train_op],
                                 feed_dict={data_handle: train_handle})
    train_losses.append(loss)
    train_accuracies.append(accuracy)

    if step % 100 == 0:
      print('{} Step {} Loss {} Acc {}'.format(
          datetime.utcnow(), step + 1, np.mean(train_losses),
          np.mean(train_accuracies)))
      train_losses.clear()
      train_accuracies.clear()

    if step % 1000 == 0:
      # Initialize the validation dataset
      sess.run(valid_iter.initializer, feed_dict={
        all_inputs_ph: mnist.validation.images,
        all_labels_ph: mnist.validation.labels})

      # Run one pass over the validation dataset.
      losses, accuracies = [], []
      while True:
        try:
          valid_loss, valid_accuracy = sess.run(
              [loss_op, accuracy_op],
              feed_dict={data_handle: valid_handle, train_switch: False})

          losses.append(valid_loss)
          accuracies.append(valid_accuracy)
        except tf.errors.OutOfRangeError:
          break
      print('{} Step {} valid_loss {} valid_acc {}'.format(datetime.utcnow(),
                                                           step + 1,
                                                           np.mean(losses),
                                                           np.mean(accuracies)))


def add_gaussian_noise(inputs, labels):
  # Values taken https://github.com/cooijmanstim/recurrent-batch-normalization
  inputs = inputs + tf.random_normal(inputs.shape, mean=0.0, stddev=0.1)
  return inputs, labels


def preprocess_data(inputs, labels):
  # General preprocessing for train, valid and test dataset
  inputs = 2 * inputs - 1
  inputs = tf.expand_dims(inputs, -1)  # expand to [?, TIME_STEPS, 1]
  labels = tf.cast(labels, tf.int32)
  return inputs, labels


def get_iterators(handle, inputs_ph, labels_ph, add_noise=BN_FRAME_WISE,
                  batch_size=BATCH_SIZE, valid_batch_size=2000, shuffle=True):
  training_dataset = tf.data.Dataset.from_tensor_slices((inputs_ph, labels_ph))
  if shuffle:
    training_dataset = training_dataset.shuffle(buffer_size=1000)
  if add_noise:
    # Apply random perturbations to the training data
    training_dataset = training_dataset.map(add_gaussian_noise)
  training_dataset = training_dataset.map(preprocess_data)
  training_dataset = training_dataset.repeat().batch(batch_size)

  # Create the validation dataset
  validation_dataset = tf.data.Dataset.from_tensor_slices(
      (inputs_ph, labels_ph))
  validation_dataset = validation_dataset.map(preprocess_data)
  validation_dataset = validation_dataset.batch(valid_batch_size)

  # Create an iterator for switching between datasets
  iterator = tf.data.Iterator.from_string_handle(
      handle, training_dataset.output_types, training_dataset.output_shapes)

  # Create iterators for each dataset that the main iterator can use for the
  # next element
  training_iterator = training_dataset.make_initializable_iterator()
  validation_iterator = validation_dataset.make_initializable_iterator()
  return iterator, training_iterator, validation_iterator


if __name__ == "__main__":
  main()
