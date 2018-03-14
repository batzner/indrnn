from tensorflow.python.ops import math_ops, init_ops, array_ops, nn_ops
from tensorflow.python.ops.rnn_cell_impl import _LayerRNNCell, \
  _WEIGHTS_VARIABLE_NAME, _BIAS_VARIABLE_NAME
from tensorflow.python.layers import base as base_layer


class BasicRNNCell(_LayerRNNCell):
  """The most basic RNN cell.
  Args:
    num_units: int, The number of units in the RNN cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
  """

  def __init__(self, num_units, activation=None, reuse=None, name=None):
    super(BasicRNNCell, self).__init__(_reuse=reuse, name=name)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._activation = activation or math_ops.tanh

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
    self._kernel = self.add_variable(
      _WEIGHTS_VARIABLE_NAME,
      shape=[input_depth + self._num_units, self._num_units])
    self._bias = self.add_variable(
      _BIAS_VARIABLE_NAME,
      shape=[self._num_units],
      initializer=init_ops.zeros_initializer(dtype=self.dtype))

    self.built = True

  def call(self, inputs, state):
    """Most basic RNN: output = new_state = act(W * input + U * state + B)."""

    gate_inputs = math_ops.matmul(
      array_ops.concat([inputs, state], 1), self._kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
    output = self._activation(gate_inputs)
    return output, output
