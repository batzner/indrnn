"""Module implementing the IndRNN cell"""

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.layers import base as base_layer

try:
  # TF 1.7+
  from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell
except ImportError:
  from tensorflow.python.ops.rnn_cell_impl import _LayerRNNCell as LayerRNNCell


class IndRNNCell(LayerRNNCell):
  """Independently RNN Cell. Adapted from `rnn_cell_impl.BasicRNNCell`.

  Each unit has a single recurrent weight connected to its last hidden state.

  The implementation is based on:

    https://arxiv.org/abs/1803.04831

  Shuai Li, Wanqing Li, Chris Cook, Ce Zhu, Yanbo Gao
  "Independently Recurrent Neural Network (IndRNN): Building A Longer and
  Deeper RNN"

  The default initialization values for recurrent weights, input weights and
  biases are taken from:

    https://arxiv.org/abs/1504.00941

  Quoc V. Le, Navdeep Jaitly, Geoffrey E. Hinton
  "A Simple Way to Initialize Recurrent Networks of Rectified Linear Units"

  Args:
    num_units: int, The number of units in the RNN cell.
    recurrent_min_abs: float, minimum absolute value of each recurrent weight.
    recurrent_max_abs: (optional) float, maximum absolute value of each
      recurrent weight. For `relu` activation, `pow(2, 1/timesteps)` is
      recommended. If None, recurrent weights will not be clipped.
      Default: None.
    recurrent_kernel_initializer: (optional) The initializer to use for the
      recurrent weights. If None, every recurrent weight is initially set to 1.
      Default: None.
    input_kernel_initializer: (optional) The initializer to use for the input
      weights. If None, the input weights are initialized from a random normal
      distribution with `mean=0` and `stddev=0.001`. Default: None.
    activation: Nonlinearity to use.  Default: `relu`.
    reuse: (optional) Python boolean describing whether to reuse variables
      in an existing scope.  If not `True`, and the existing scope already has
      the given variables, an error is raised.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
  """

  def __init__(self,
               num_units,
               recurrent_min_abs=0,
               recurrent_max_abs=None,
               recurrent_kernel_initializer=None,
               input_kernel_initializer=None,
               activation=None,
               reuse=None,
               name=None):
    super(IndRNNCell, self).__init__(_reuse=reuse, name=name)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._recurrent_min_abs = recurrent_min_abs
    self._recurrent_max_abs = recurrent_max_abs
    self._recurrent_initializer = recurrent_kernel_initializer
    self._input_initializer = input_kernel_initializer
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
    if self._input_initializer is None:
      self._input_initializer = init_ops.random_normal_initializer(mean=0.0,
                                                                   stddev=0.001)
    self._input_kernel = self.add_variable(
        "input_kernel",
        shape=[input_depth, self._num_units],
        initializer=self._input_initializer)

    if self._recurrent_initializer is None:
      self._recurrent_initializer = init_ops.constant_initializer(1.)
    self._recurrent_kernel = self.add_variable(
        "recurrent_kernel",
        shape=[self._num_units],
        initializer=self._recurrent_initializer)

    # Clip the absolute values of the recurrent weights to the specified minimum
    if self._recurrent_min_abs:
      abs_kernel = math_ops.abs(self._recurrent_kernel)
      min_abs_kernel = math_ops.maximum(abs_kernel, self._recurrent_min_abs)
      self._recurrent_kernel = math_ops.multiply(
          math_ops.sign(self._recurrent_kernel),
          min_abs_kernel
      )

    # Clip the absolute values of the recurrent weights to the specified maximum
    if self._recurrent_max_abs:
      self._recurrent_kernel = clip_ops.clip_by_value(self._recurrent_kernel,
                                                      -self._recurrent_max_abs,
                                                      self._recurrent_max_abs)

    self._bias = self.add_variable(
        "bias",
        shape=[self._num_units],
        initializer=init_ops.zeros_initializer(dtype=self.dtype))

    self.built = True

  def call(self, inputs, state):
    """Run one time step of the IndRNN.

    Calculates the output and new hidden state using the IndRNN equation

      `output = new_state = act(W * input + u (*) state + b)`

    where `*` is the matrix multiplication and `(*)` is the Hadamard product.

    Args:
      inputs: Tensor, 2-D tensor of shape `[batch, num_units]`.
      state: Tensor, 2-D tensor of shape `[batch, num_units]` containing the
        previous hidden state.

    Returns:
      A tuple containing the output and new hidden state. Both are the same
        2-D tensor of shape `[batch, num_units]`.
    """
    gate_inputs = math_ops.matmul(inputs, self._input_kernel)
    recurrent_update = math_ops.multiply(state, self._recurrent_kernel)
    gate_inputs = math_ops.add(gate_inputs, recurrent_update)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
    output = self._activation(gate_inputs)
    return output, output
