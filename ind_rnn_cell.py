"""Module implementing the IndRNN cell"""
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.layers import base as base_layer


class IndRNNCell(rnn_cell_impl.RNNCell):
  """Independently RNN Cell. Adapted from `rnn_cell_impl.BasicRNNCell

  The implementation is based on:
    https://arxiv.org/abs/1803.04831
    Shuai Li, Wanqing Li, Chris Cook, Ce Zhu, Yanbo Gao
    "Independently Recurrent Neural Network (IndRNN): Building A Longer and
      Deeper RNN"

  Args:
    num_units: int, The number of units in the RNN cell.
    recurrent_max: float, maximum absolute value of each recurrent weight. For
      relu activation, use pow(2, 1/timesteps). The IndRNN paper gives further
      recommendations for other activations functions. If None, recurrent
      weights will not be clipped. Default: None.
    activation: Nonlinearity to use.  Default: `relu`.
    reuse: (optional) Python boolean describing whether to reuse variables
      in an existing scope.  If not `True`, and the existing scope already has
      the given variables, an error is raised.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
  """

  def __init__(self, num_units, recurrent_max=None, activation=None,
               reuse=None, name=None):
    super(IndRNNCell, self).__init__(_reuse=reuse, name=name)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._recurrent_max = recurrent_max
    self._activation = activation or nn_ops.relu

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value
    self._input_kernel = self.add_variable(
      "input_%s" % rnn_cell_impl._WEIGHTS_VARIABLE_NAME,
      shape=[input_depth, self._num_units])

    recurrent_init = init_ops.random_uniform_initializer(
        minval=0.0,
        maxval=1.0 if self._recurrent_max is None else self._recurrent_max
    )

    self._recurrent_kernel = self.add_variable(
      "recurrent_%s" % rnn_cell_impl._WEIGHTS_VARIABLE_NAME,
      shape=[self._num_units], initializer=recurrent_init)
    if self._recurrent_max:
      self._recurrent_kernel = clip_ops.clip_by_value(self._recurrent_kernel, 0, self._recurrent_max)

    self._bias = self.add_variable(
      rnn_cell_impl._BIAS_VARIABLE_NAME,
      shape=[self._num_units],
      initializer=init_ops.zeros_initializer(dtype=self.dtype))

    self.built = True

  def call(self, inputs, state):
    """IndRNN: output = new_state = act(W @ input + u * state + b),

    where @ is the matrix multiplication and * is the element-wise
    multiplication of two vectors.
    """
    gate_inputs = math_ops.matmul(inputs, self._input_kernel)
    gate_inputs = math_ops.add(gate_inputs,
                               math_ops.multiply(state, self._recurrent_kernel))
    gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
    output = self._activation(gate_inputs)
    return output, output
