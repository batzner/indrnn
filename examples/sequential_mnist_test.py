import tensorflow as tf
from tensorflow.python.platform import test
import numpy as np

from examples.sequential_mnist import get_iterators


class TestSequentialMnist(test.TestCase):
  def testTrainingOutputs(self):
    batch_size = 2
    train_inputs = np.array([[1, 2], [3, 4], [5, 6]])
    train_labels = np.array([11, 12, 13])

    expected_input_batches = [[[1, 2], [3, 4]], [[5, 6], [1, 2]]]
    expected_input_batches = np.array(expected_input_batches).reshape(
        (2, 2, 2, 1))
    expected_input_labels = [[11, 12], [13, 11]]

    data_handle = tf.placeholder(tf.string, shape=[])
    all_inputs_ph = tf.placeholder(tf.float32, [None, 2])
    all_labels_ph = tf.placeholder(tf.int32, [None])

    main_iter, train_iter, _ = get_iterators(data_handle,
                                             all_inputs_ph,
                                             all_labels_ph,
                                             add_noise=False,
                                             batch_size=batch_size,
                                             shuffle=False)
    sess = tf.Session()
    sess.run(train_iter.initializer, feed_dict={
      all_inputs_ph: train_inputs,
      all_labels_ph: train_labels})

    train_handle = sess.run(train_iter.string_handle())
    inputs_op, labels_op = main_iter.get_next()
    # Generate the first batch
    inputs, labels = sess.run([inputs_op, labels_op],
                              feed_dict={data_handle: train_handle})
    self.assertAllEqual(inputs, expected_input_batches[0])
    self.assertAllEqual(labels, expected_input_labels[0])

    # Generate the second batch
    inputs, labels = sess.run([inputs_op, labels_op],
                              feed_dict={data_handle: train_handle})
    self.assertAllEqual(inputs, expected_input_batches[1])
    self.assertAllEqual(labels, expected_input_labels[1])

  def testValidationOutputs(self):
    batch_size = 2
    train_inputs = np.random.rand(10, 2)
    train_labels = np.random.rand(10)
    valid_inputs = np.array([[1, 2], [3, 4], [5, 6]])
    valid_labels = np.array([11, 12, 13])

    expected_input_batches = [[[[1], [2]], [[3], [4]]], [[[5], [6]]]]
    expected_input_labels = [[11, 12], [13]]

    data_handle = tf.placeholder(tf.string, shape=[])
    all_inputs_ph = tf.placeholder(tf.float32, [None, 2])
    all_labels_ph = tf.placeholder(tf.int32, [None])

    main_iter, train_iter, valid_iter = get_iterators(data_handle,
                                                      all_inputs_ph,
                                                      all_labels_ph,
                                                      batch_size=batch_size)
    sess = tf.Session()
    sess.run(train_iter.initializer, feed_dict={
      all_inputs_ph: train_inputs,
      all_labels_ph: train_labels})
    sess.run(valid_iter.initializer, feed_dict={
      all_inputs_ph: valid_inputs,
      all_labels_ph: valid_labels})

    # Generate handles for each iterator
    train_handle = sess.run(train_iter.string_handle())
    valid_handle = sess.run(valid_iter.string_handle())
    inputs_op, labels_op = main_iter.get_next()

    # Generate some train labels first
    sess.run([inputs_op, labels_op], feed_dict={data_handle: train_handle})

    # Generate the first batch
    inputs, labels = sess.run([inputs_op, labels_op],
                              feed_dict={data_handle: valid_handle})
    self.assertAllEqual(inputs, expected_input_batches[0])
    self.assertAllEqual(labels, expected_input_labels[0])

    # Generate the second batch
    inputs, labels = sess.run([inputs_op, labels_op],
                              feed_dict={data_handle: valid_handle})
    self.assertAllEqual(inputs, expected_input_batches[1])
    self.assertAllEqual(labels, expected_input_labels[1])

  def testTrainingValidationMix(self):
    batch_size = 2
    train_inputs = np.array([[1, 2], [3, 4], [5, 6]])
    train_labels = np.array([11, 12, 13])
    valid_inputs = np.random.rand(10, 2)
    valid_labels = np.random.rand(10)

    expected_input_batches = [[[1, 2], [3, 4]], [[5, 6], [1, 2]]]
    expected_input_batches = np.array(expected_input_batches).reshape(
        (2, 2, 2, 1))
    expected_input_labels = [[11, 12], [13, 11]]

    data_handle = tf.placeholder(tf.string, shape=[])
    all_inputs_ph = tf.placeholder(tf.float32, [None, 2])
    all_labels_ph = tf.placeholder(tf.int32, [None])

    main_iter, train_iter, valid_iter = get_iterators(data_handle,
                                                      all_inputs_ph,
                                                      all_labels_ph,
                                                      batch_size=batch_size,
                                                      add_noise=False,
                                                      shuffle=False)
    sess = tf.Session()
    sess.run(train_iter.initializer, feed_dict={
      all_inputs_ph: train_inputs,
      all_labels_ph: train_labels})

    # Generate handles for each iterator
    train_handle = sess.run(train_iter.string_handle())
    valid_handle = sess.run(valid_iter.string_handle())
    inputs_op, labels_op = main_iter.get_next()

    # Generate the first batch
    inputs, labels = sess.run([inputs_op, labels_op],
                              feed_dict={data_handle: train_handle})
    self.assertAllEqual(inputs, expected_input_batches[0])
    self.assertAllEqual(labels, expected_input_labels[0])

    # Iterate through the validation set
    sess.run(valid_iter.initializer, feed_dict={
      all_inputs_ph: valid_inputs,
      all_labels_ph: valid_labels})
    while True:
      try:
        sess.run([inputs_op, labels_op], feed_dict={data_handle: valid_handle})
      except tf.errors.OutOfRangeError:
        break

    # Generate the second batch
    inputs, labels = sess.run([inputs_op, labels_op],
                              feed_dict={data_handle: train_handle})
    self.assertAllEqual(inputs, expected_input_batches[1])
    self.assertAllEqual(labels, expected_input_labels[1])
