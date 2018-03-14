import tensorflow as tf
import numpy as np

SEQUENCE_LENGTH = 100
BATCH_SIZE = 50
NUM_UNITS = 128
LEARNING_RATE = 0.002


def main():
    inputs_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, SEQUENCE_LENGTH, 2))
    targets_ph = tf.placeholder(tf.float32, shape=BATCH_SIZE)

    cell = tf.contrib.rnn.LSTMCell(NUM_UNITS)
    output, _ = tf.nn.dynamic_rnn(cell, inputs_ph, dtype=tf.float32)
    last = output[:, -1, :]

    weight, bias = weight_and_bias(NUM_UNITS, 1)
    prediction = tf.squeeze(tf.matmul(last, weight) + bias)
    loss_op = tf.losses.mean_squared_error(tf.squeeze(targets_ph), prediction)
    optimize = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_op)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        for epoch in range(100):
            losses = []
            for _ in range(100):
                inputs, targets = get_batch()
                step += 1
                loss, _ = sess.run([loss_op, optimize], {inputs_ph: inputs, targets_ph: targets})
                losses.append(loss)
            print('Epoch {} Step [hundreds] {} MSE {}'.format(epoch + 1, int(step / 100), np.mean(losses)))


def weight_and_bias(in_size, out_size):
    weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
    bias = tf.constant(0.1, shape=[out_size])
    return tf.Variable(weight), tf.Variable(bias)


# Generate the adding problem dataset as described in https://arxiv.org/abs/1803.04831
def get_batch():
    add_values = np.random.rand(BATCH_SIZE, SEQUENCE_LENGTH)
    add_indices = np.zeros_like(add_values)
    half = int(SEQUENCE_LENGTH/2)
    for i in range(BATCH_SIZE):
        first_half = np.random.randint(half)
        second_half = np.random.randint(half, SEQUENCE_LENGTH)
        add_indices[i, [first_half, second_half]] = 1

    inputs = np.dstack((add_values, add_indices))
    targets = np.sum(np.multiply(add_values, add_indices), axis=1)
    return inputs, targets


if __name__ == '__main__':
    main()
