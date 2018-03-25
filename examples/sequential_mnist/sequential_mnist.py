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

BATCH_SIZE_TRAIN = 32
BATCH_SIZE_BN_STATS = 500
BATCH_SIZE_VALID = 2000
CLIP_GRADIENTS = True

PHASE_TRAIN = 'train'
PHASE_BN_STATS = 'bn_stats'
PHASE_VALID = 'validation'
PHASE_TEST = 'test'

OUT_PATH = "out/%s/" % datetime.utcnow()

# Import MNIST data (Numpy format)
MNIST = input_data.read_data_sets("/tmp/data/")


def main():
  sess = tf.Session()

  # Create placeholders for feeding the data
  data_handle = tf.placeholder(tf.string, shape=[], name="data_handle")
  iterator, handles, init_validation_set = get_iterators(sess, data_handle)
  inputs, labels = iterator.get_next()

  # Build the graph depending on the current phase
  phase = tf.placeholder(tf.string, shape=[], name="phase")
  loss_op, accuracy_op, train_op = build(inputs, labels, phase)

  # Train the model
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()

  train_losses = []
  train_accuracies = []
  for step in itertools.count():
    # Execute one training step
    loss, accuracy, _ = sess.run(
        [loss_op, accuracy_op, train_op],
        feed_dict={data_handle: handles[PHASE_TRAIN], phase: PHASE_TRAIN})
    train_losses.append(loss)
    train_accuracies.append(accuracy)

    if step % 100 == 0:
      print('{} Step {} Loss {} Acc {}'.format(
          datetime.utcnow(), step + 1, np.mean(train_losses),
          np.mean(train_accuracies)))
      train_losses.clear()
      train_accuracies.clear()

    if step % 1000 == 0:
      # Update the population statistics without changing the parameters
      sess.run([loss_op], feed_dict={
        data_handle: handles[PHASE_BN_STATS],
        phase: PHASE_BN_STATS})

      # Run one pass over the validation dataset.
      init_validation_set()
      feed_dict = {data_handle: handles[PHASE_VALID], phase: PHASE_VALID}
      loss, accuracy = evaluate(sess, loss_op, accuracy_op, feed_dict)
      print('{} Step {} valid_loss {} valid_acc {}'.format(datetime.utcnow(),
                                                           step + 1,
                                                           loss,
                                                           accuracy))

      if accuracy > 0.99:
        # Run the final test
        feed_dict = {data_handle: handles[PHASE_TEST], phase: PHASE_TEST}
        loss, accuracy = evaluate(sess, loss_op, accuracy_op, feed_dict)
        print('{} Step {} test_loss {} test_acc {}'.format(datetime.utcnow(),
                                                           step + 1,
                                                           loss,
                                                           accuracy))
        # Exit
        return

    if step % 2000 == 0:
      save_path = saver.save(sess, OUT_PATH + "model.ckpt")
      print("Model saved in path: %s" % save_path)


def evaluate(session, loss_op, accuracy_op, feed_dict):
  losses, accuracies = [], []
  while True:
    try:
      loss, accuracy = session.run(
          [loss_op, accuracy_op],
          feed_dict=feed_dict)

      losses.append(loss)
      accuracies.append(accuracy)
    except tf.errors.OutOfRangeError:
      break
  return np.mean(losses), np.mean(accuracies)


def build(inputs, labels, phase):
  # Build the graph
  output = get_bn_rnn(inputs, phase=phase)
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
  return loss, accuracy, optimize


def get_bn_rnn(inputs, phase):
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

    is_training = tf.logical_or(tf.equal(phase, PHASE_TRAIN),
                                tf.equal(phase, PHASE_BN_STATS))
    layer_output = tf.layers.batch_normalization(layer_output,
                                                 training=is_training,
                                                 momentum=0)

    # Tie the BN population statistics updates to the layer_output op
    def update_population_stats():
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        return tf.identity(layer_output)

    layer_output = tf.cond(tf.equal(phase, PHASE_BN_STATS),
                           true_fn=update_population_stats,
                           false_fn=lambda: layer_output)

    layer_input = layer_output

  return layer_output


def preprocess_data(inputs, labels):
  # General preprocessing for every dataset
  inputs = 2 * inputs - 1
  inputs = tf.expand_dims(inputs, -1)  # expand to [?, TIME_STEPS, 1]
  labels = tf.cast(labels, tf.int32)
  return inputs, labels


def get_training_set(inputs, labels):
  # Create the training set
  dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
  dataset = dataset.shuffle(buffer_size=1000)
  dataset = dataset.map(preprocess_data)
  return dataset.repeat().batch(BATCH_SIZE_TRAIN)


def get_bn_stats_set(inputs, labels):
  # Create the "set BN stats" set
  dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
  dataset = dataset.shuffle(buffer_size=1000)
  dataset = dataset.map(preprocess_data)
  return dataset.repeat().batch(BATCH_SIZE_BN_STATS)


def get_prediction_set(inputs, labels):
  # Create the validation dataset
  dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
  dataset = dataset.map(preprocess_data)
  return dataset.batch(BATCH_SIZE_VALID)


def get_iterators(session, handle):
  inputs_ph = tf.placeholder(MNIST.train.images.dtype, [None, 784],
                             name="all_inputs")
  labels_ph = tf.placeholder(MNIST.train.labels.dtype, [None],
                             name="all_labels")

  training_dataset = get_training_set(inputs_ph, labels_ph)
  bn_stats_dataset = get_bn_stats_set(inputs_ph, labels_ph)
  validation_dataset = get_prediction_set(inputs_ph, labels_ph)
  test_dataset = get_prediction_set(inputs_ph, labels_ph)

  # Create an iterator for switching between datasets
  iterator = tf.data.Iterator.from_string_handle(
      handle, training_dataset.output_types, training_dataset.output_shapes)

  # Create iterators for each dataset that the main iterator can use for the
  # next element
  training_iterator = training_dataset.make_initializable_iterator()
  bn_stats_iterator = bn_stats_dataset.make_initializable_iterator()
  validation_iterator = validation_dataset.make_initializable_iterator()
  test_iterator = test_dataset.make_initializable_iterator()

  # Initialize the datasets
  session.run(training_iterator.initializer, feed_dict={
    inputs_ph: MNIST.train.images,
    labels_ph: MNIST.train.labels})
  session.run(bn_stats_iterator.initializer, feed_dict={
    inputs_ph: MNIST.train.images,
    labels_ph: MNIST.train.labels})
  session.run(test_iterator.initializer, feed_dict={
    inputs_ph: MNIST.test.images,
    labels_ph: MNIST.test.labels})

  def init_validation_set():
    # Initialize the validation dataset
    session.run(validation_iterator.initializer, feed_dict={
      inputs_ph: MNIST.validation.images,
      labels_ph: MNIST.validation.labels})

  # Generate handles for each iterator
  handles = {
    PHASE_TRAIN: session.run(training_iterator.string_handle()),
    PHASE_BN_STATS: session.run(bn_stats_iterator.string_handle()),
    PHASE_VALID: session.run(validation_iterator.string_handle()),
    PHASE_TEST: session.run(test_iterator.string_handle())
  }
  return iterator, handles, init_validation_set


if __name__ == "__main__":
  main()
