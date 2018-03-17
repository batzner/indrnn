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
  # Placeholders for training data
  inputs_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, TIME_STEPS, 2))
  targets_ph = tf.placeholder(tf.float32, shape=BATCH_SIZE)

  # Build the graph
  cell = tf.nn.rnn_cell.MultiRNNCell([
    IndRNNCell(NUM_UNITS, recurrent_max_abs=RECURRENT_MAX)
    for _ in range(NUM_LAYERS)
  ])
  # cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_UNITS) uncomment this for LSTM runs

  output, state = tf.nn.dynamic_rnn(cell, inputs_ph, dtype=tf.float32)
  last = output[:, -1, :]

  weight = tf.get_variable("softmax_weight", shape=[NUM_UNITS, 1])
  bias = tf.get_variable("softmax_bias", shape=[1],
                         initializer=tf.constant_initializer(0.1))
  prediction = tf.squeeze(tf.matmul(last, weight) + bias)

  loss_op = tf.losses.mean_squared_error(tf.squeeze(targets_ph), prediction)

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
        # Generate new input data
        inputs, targets = get_batch()
        loss, _ = sess.run([loss_op, optimize],
                           {inputs_ph: inputs, targets_ph: targets})
        losses.append(loss)
        step += 1
      print("Step [x100] {} MSE {}".format(int(step / 100), np.mean(losses)))


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
