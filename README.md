# Independently Recurrent Neural Networks

TensorFlow implementation of (Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN)[https://arxiv.org/pdf/1803.04831.pdf] by Shuai Li et al.

## Summary

In IndRNNs, neurons in recurrent layers are independent from each other. The basic RNN calculates the hidden state `h` with `h = act(W @ input + U @ state + b)`, where `@` is the matrix multiplication. IndRNNs use an element-wise vector multiplication `u * state`. Each neuron's hidden state is independent from the other neurons. 

The IndRNN 
- makes it easier to stack multiple recurrent layers without saturating gradients
- allows for better interpretability, as the neurons in one layer are independent from each other
- prevents vanishing and exploding gradients by regulating each neuron's recurrent weight

## Usage

    from ind_rnn_cell import IndRNNCell
    
    # Regulate each neuron's recurrent weight
    recurrent_max = pow(2, 1 / TIME_STEPS)
    
    cell = MultiRNNCell([IndRNNCell(128, recurrent_max),
                       IndRNNCell(128, recurrent_max)])
    output, state = tf.nn.dynamic_rnn(cell, inputs_ph, dtype=tf.float32)
    ...
    
## Experiments in the paper

