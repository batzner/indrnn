# Independently Recurrent Neural Networks

Simple TensorFlow implementation of [Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN](https://arxiv.org/pdf/1803.04831.pdf) by Shuai Li et al.

## Summary

In IndRNNs, neurons in recurrent layers are independent from each other. The basic RNN calculates the hidden state `h` with `h = act(W * input + U * state + b)`. IndRNNs use an element-wise vector multiplication `u * state` meaning each neuron has a single recurrent weight connected to its last hidden state. 

The IndRNN 
- can be used efficiently with ReLU activation functions making it easier to stack multiple recurrent layers without saturating gradients
- allows for better interpretability, as neurons in the same layer are independent from each other
- prevents vanishing and exploding gradients by regulating each neuron's recurrent weight

## Usage

Copy [ind_rnn_cell.py](https://github.com/batzner/indrnn/blob/master/ind_rnn_cell.py) into your project.

```python
from ind_rnn_cell import IndRNNCell

# Regulate each neuron's recurrent weight as recommended in the paper
recurrent_max = pow(2, 1 / TIME_STEPS)

cell = MultiRNNCell([IndRNNCell(128, recurrent_max_abs=recurrent_max),
                     IndRNNCell(128, recurrent_max_abs=recurrent_max)])
output, state = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
...
```
## Experiments in the paper

See [examples/addition_rnn.py](https://github.com/batzner/indrnn/blob/master/examples/addition_rnn.py) for a script reconstructing the "Adding Problem" from the paper. More experiments, such as Sequential MNIST, will follow in the next days. 

## Requirements
- Python 3.4+
- TensorFlow 1.5+
