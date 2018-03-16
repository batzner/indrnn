import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import backend as K


class IndRNNCell(keras.layers.Layer):
  def __init__(self, units, **kwargs):
    self.units = units
    self.state_size = units
    super(IndRNNCell, self).__init__(**kwargs)

  def build(self, input_shape):
    self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                  initializer='glorot_uniform',
                                  name='kernel')
    self.recurrent_kernel = self.add_weight(
        shape=self.units,
        initializer='glorot_uniform',
        name='recurrent_kernel')

    self.recurrent_kernel_clip = K.clip(self.recurrent_kernel,
                                   -pow(2, 1/784), pow(2, 1/784))

    self.bias = self.add_weight(
        shape=self.units,
        initializer='zero',
        name='bias')
    self.built = True

  def call(self, inputs, states):
    prev_output = states[0]
    h = K.dot(inputs, self.kernel)
    output = h + tf.multiply(prev_output, self.recurrent_kernel_clip) + self.bias
    return output, [output]
