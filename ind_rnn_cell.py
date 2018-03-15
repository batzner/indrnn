"""Module implementing the IndRNN cell"""
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.layers import base as base_layer


class IndRNNCell(rnn_cell_impl._LayerRNNCell):
  """Independently RNN Cell. Adapted from `rnn_cell_impl.BasicRNNCell

  The implementation is based on:
    https://arxiv.org/abs/1803.04831
    Shuai Li, Wanqing Li, Chris Cook, Ce Zhu, Yanbo Gao
    "Independently Recurrent Neural Network (IndRNN): Building A Longer and
      Deeper RNN"

  Args:
    num_units: int, The number of units in the RNN cell.
    recurrent_min: float, minimum absolute value of each recurrent weight. This
      prevents vanishing gradients.
    recurrent_max: float, maximum absolute value of each recurrent weight. This
      prevents exploding gradients. For relu activation, use
      pow(2, 1/timesteps) is recommended. If None, recurrent weights will not
      be clipped. Default: None.
    recurrent_initializer: (optional) The initializer to use for the recurrent
      weights. The default is a uniform distribution in the range [-1, 1] if
      `recurrent_max` is not set, or in [-recurrent_max, recurrent_max] if it
      is.
    activation: Nonlinearity to use.  Default: `relu`.
    reuse: (optional) Python boolean describing whether to reuse variables
      in an existing scope.  If not `True`, and the existing scope already has
      the given variables, an error is raised.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
  """

  def __init__(self, num_units, recurrent_min=0, recurrent_max=None,
               recurrent_initializer=None, activation=None, reuse=None,
               name=None):
    super(IndRNNCell, self).__init__(_reuse=reuse, name=name)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._recurrent_min = recurrent_min
    self._recurrent_max = recurrent_max
    self._recurrent_initializer = recurrent_initializer
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

    if self._recurrent_initializer is None:
      self._recurrent_initializer = init_ops.random_uniform_initializer(
        minval=-1.0 if self._recurrent_max is None else -self._recurrent_max,
        maxval=1.0 if self._recurrent_max is None else self._recurrent_max
      )

    self._recurrent_kernel = self.add_variable(
      "recurrent_%s" % rnn_cell_impl._WEIGHTS_VARIABLE_NAME,
      shape=[self._num_units], initializer=self._recurrent_initializer)

    # Clip the absolute values of the recurrent weights
    if self._recurrent_min:
      abs_kernel = math_ops.abs(self._recurrent_kernel)
      min_abs_kernel = math_ops.minimum(abs_kernel, self._recurrent_min)
      self._recurrent_kernel = math_ops.multiply(
        math_ops.sign(self._recurrent_kernel),
        min_abs_kernel
      )

    if self._recurrent_max:
      self._recurrent_kernel = clip_ops.clip_by_value(self._recurrent_kernel,
                                                      -self._recurrent_max,
                                                      self._recurrent_max)

    self._bias = self.add_variable(
      rnn_cell_impl._BIAS_VARIABLE_NAME,
      shape=[self._num_units],
      initializer=init_ops.zeros_initializer(dtype=self.dtype))

    self.built = True

  def call(self, inputs, state):
    """IndRNN: output = new_state = act(W * input + u (*) state + b),

    where * is the matrix multiplication and (*) is the Hadamard product.
    """
    gate_inputs = math_ops.matmul(inputs, self._input_kernel)
    gate_inputs = math_ops.add(gate_inputs,
                               math_ops.multiply(state, self._recurrent_kernel))
    gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
    output = self._activation(gate_inputs)
    return output, output
