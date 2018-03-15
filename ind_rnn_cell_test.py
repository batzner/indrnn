import numpy as np

from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test

from ind_rnn_cell import IndRNNCell


class RNNCellTest(test.TestCase):
  def testIndRNNCell(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(1.)):
        x = array_ops.zeros([1, 4])
        m = array_ops.zeros([1, 4])
        recurrent_init = init_ops.constant_initializer([-5., -2., 0.1, 5.])
        cell = IndRNNCell(4, recurrent_min_abs=1., recurrent_max_abs=3.,
                          recurrent_initializer=recurrent_init)
        output, _ = cell(x, m)
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run([output], {x.name: np.array([[1., 1., 1., 1.]]),
                                  m.name: np.array([[2., 2., 2., 2.]])})
        # Recurrent Weights u should be -3, -2, 1, 3
        # Pre-activations (4 + 2*u) should be -2, 0, 6, 10
        self.assertAllEqual(res[0], [[0., 0., 6., 10.]])
