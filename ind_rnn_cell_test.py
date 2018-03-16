"""Tests for the IndRNN cell."""

import numpy as np

from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

from ind_rnn_cell import IndRNNCell


class IndRNNCellTest(test.TestCase):
  def testIndRNNCell(self):
    """Tests basic cell functionality"""

    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(1.)):
        x = array_ops.zeros([1, 4])
        m = array_ops.zeros([1, 4])

        # Create the cell with input weights = 1 and constant recurrent weights
        recurrent_init = init_ops.constant_initializer([-3., -2., 1., 3.])
        cell = IndRNNCell(4,
                          recurrent_initializer=recurrent_init,
                          activation=array_ops.identity)
        output, _ = cell(x, m)

        sess.run([variables.global_variables_initializer()])
        res = sess.run([output],
                       {x.name: np.array([[1., 0., 0., 0.]]),
                         m.name: np.array([[2., 2., 2., 2.]])})
        # (Pre)activations (1*1 + 2*rec_weight) should be -5, -3, 3, 7
        self.assertAllEqual(res[0], [[-5., -3., 3., 7.]])

  def testIndRNNCellBounds(self):
    """Tests cell with recurrent weights exceeding the bounds."""

    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(1.)):
        x = array_ops.zeros([1, 4])
        m = array_ops.zeros([1, 4])

        # Create the cell with input weights = 1 and constant recurrent weights
        recurrent_init = init_ops.constant_initializer([-5., -2., 0.1, 5.])
        cell = IndRNNCell(4,
                          recurrent_min_abs=1.,
                          recurrent_max_abs=3.,
                          recurrent_initializer=recurrent_init,
                          activation=array_ops.identity)
        output, _ = cell(x, m)

        sess.run([variables.global_variables_initializer()])
        res = sess.run([output],
                       {x.name: np.array([[1., 0., 0., 0.]]),
                         m.name: np.array([[2., 2., 2., 2.]])})
        # Recurrent weights should be clipped to -3, -2, 1, 3
        # (Pre)activations (1*1 + 2*rec_weight) should be -5, -3, 3, 7
        self.assertAllEqual(res[0], [[-5., -3., 3., 7.]])
