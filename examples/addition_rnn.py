"""Module using IndRNNCell to solve the addition problem

The addition problem is stated in https://arxiv.org/abs/1803.04831. The
hyper-parameters are taken from that paper as well. The network should converge
to a MSE around zero after 1500-20000 steps, depending on the number of time
steps.
"""
import tensorflow as tf
import numpy as np

from ind_rnn_cell import IndRNNCell

# Parameters taken from https://arxiv.org/abs/1803.04831
TIME_STEPS = 100
NUM_UNITS = 128
LEARNING_RATE_INIT = 0.0002
LEARNING_RATE_DECAY_STEPS = 20000
NUM_LAYERS = 2
RECURRENT_MAX = pow(2, 1 / TIME_STEPS)

# Parameters taken from https://arxiv.org/abs/1511.06464
BATCH_SIZE = 50


def main():
  # Generate the dataset directly in the computational graph
  inputs, targets = get_batch_variables()

  # Build the graph
  cell = tf.nn.rnn_cell.MultiRNNCell([
    IndRNNCell(NUM_UNITS, recurrent_max_abs=RECURRENT_MAX) for _ in
    range(NUM_LAYERS)
  ])
  # cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_UNITS) uncomment this for LSTM runs

  output, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
  last = output[:, -1, :]

  weight = tf.get_variable("softmax_weight", shape=[NUM_UNITS, 1])
  bias = tf.get_variable("softmax_bias", shape=[1],
                         initializer=tf.constant_initializer(0.1))
  prediction = tf.squeeze(tf.matmul(last, weight) + bias)

  loss_op = tf.losses.mean_squared_error(tf.squeeze(targets), prediction)

  global_step = tf.get_variable("global_step", shape=[], trainable=False,
                                initializer=tf.zeros_initializer)
  learning_rate = tf.train.exponential_decay(LEARNING_RATE_INIT, global_step,
                                             LEARNING_RATE_DECAY_STEPS, 0.1,
                                             staircase=True)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  optimize = optimizer.minimize(loss_op, global_step=global_step)

  # Train the model
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    while True:
      losses = []
      for _ in range(100):
        loss, _ = sess.run([loss_op, optimize])
        losses.append(loss)
        step += 1
      print("Step [x100] {} MSE {}".format(int(step / 100), np.mean(losses)))


def get_batch_variables():
  """Generate the adding problem dataset in the computational graph"""
  input_values = tf.random_uniform([BATCH_SIZE, TIME_STEPS])

  # Build the input indices by choosing two random integers (one per half) and
  # concatenating their one-hot-encodings
  half = int(TIME_STEPS / 2)
  input_index_first = tf.random_uniform([BATCH_SIZE], 0, half - 1, tf.int32)
  input_index_second = tf.random_uniform([BATCH_SIZE], 0, half - 1, tf.int32)
  input_indices = tf.concat([tf.one_hot(input_index_first, half),
                              tf.one_hot(input_index_second, half)], axis=1)

  targets = tf.reduce_sum(tf.multiply(input_values, input_indices), axis=1)
  # Zip the values and indices in a third dimension:
  inputs = tf.stack([input_values, input_indices], axis=-1)
  # inputs has the shape (batch_size, time_steps, 2)

  return inputs, targets


if __name__ == "__main__":
  main()
