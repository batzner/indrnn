"""Module using IndRNNCell to solve the addition problem

The addition problem is stated in https://arxiv.org/abs/1803.04831. The
hyper-parameters are taken from that paper as well. The network should
converge to a MSE around zero after 5000-10000 steps.
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell_impl import MultiRNNCell

from ind_rnn_cell import IndRNNCell

# Parameters taken from https://arxiv.org/abs/1803.04831
TIME_STEPS = 100
NUM_UNITS = 128
LEARNING_RATE = 0.0002
NUM_LAYERS = 2
RECURRENT_MAX = pow(2, 1 / TIME_STEPS)

# Parameters taken from https://arxiv.org/abs/1511.06464
BATCH_SIZE = 50


def main():
  # Placeholders for training data
  inputs_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, TIME_STEPS, 2))
  targets_ph = tf.placeholder(tf.float32, shape=BATCH_SIZE)

  # Build the graph
  cell = MultiRNNCell([IndRNNCell(NUM_UNITS, RECURRENT_MAX),
                       IndRNNCell(NUM_UNITS, RECURRENT_MAX)])

  output, state = tf.nn.dynamic_rnn(cell, inputs_ph, dtype=tf.float32)
  last = output[:, -1, :]

  weight = tf.Variable(tf.truncated_normal([NUM_UNITS, 1], stddev=0.01))
  bias = tf.Variable(tf.constant(0.1, shape=[1]))
  prediction = tf.squeeze(tf.matmul(last, weight) + bias)

  loss_op = tf.losses.mean_squared_error(tf.squeeze(targets_ph), prediction)
  optimize = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_op)

  # Train the model
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    while True:
      losses = []
      for _ in range(100):
        # Generate new input data
        inputs, targets = get_batch()
        loss, _ = sess.run([loss_op, optimize],
                           {inputs_ph: inputs, targets_ph: targets})
        losses.append(loss)
        step += 1

      print(
        "Step [x100] {} MSE {}".format(int(step / 100), np.mean(losses)))


def get_batch():
  """Generate the adding problem dataset"""
  # Build the first sequence
  add_values = np.random.rand(BATCH_SIZE, TIME_STEPS)

  # Build the second sequence with one 1 in each half and 0s otherwise
  add_indices = np.zeros_like(add_values)
  half = int(TIME_STEPS / 2)
  for i in range(BATCH_SIZE):
    first_half = np.random.randint(half)
    second_half = np.random.randint(half, TIME_STEPS)
    add_indices[i, [first_half, second_half]] = 1

  # Zip the values and indices in a third dimension:
  # inputs has the shape (batch_size, time_steps, 2)
  inputs = np.dstack((add_values, add_indices))
  targets = np.sum(np.multiply(add_values, add_indices), axis=1)
  return inputs, targets


if __name__ == "__main__":
  main()
