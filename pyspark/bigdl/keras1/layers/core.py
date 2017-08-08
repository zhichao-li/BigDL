# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import numpy as np
import bigdl.nn.layer as bigdl_layer
from bigdl.keras1.engine.topology import *
import bigdl.keras1.activations as activations

class Input(Layer):
    def __init__(self, **kwargs):
        super(Input, self).__init__(**kwargs)

    def build(self, x):
        self.B = bigdl_layer.Input()
        self.output_shape = x.output_shape if x is not None else self.input_shape

class Dense(Layer):
    """Just your regular densely-connected NN layer.

    # Example

    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_dim=16))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

        # this is equivalent to the above:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))

        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```

    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer). This argument
            (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.

    # Input shape
        nD tensor with shape: `(nb_samples, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(nb_samples, input_dim)`.

    # Output shape
        nD tensor with shape: `(nb_samples, ..., output_dim)`.
        For instance, for a 2D input with shape `(nb_samples, input_dim)`,
        the output would have shape `(nb_samples, output_dim)`.
    """

    def __init__(self, output_dim, init='glorot_uniform',
                 activation=None, weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, input_dim=None, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.activation = Activation(activation) if activation else None
        if self.input_dim:
            self.input_shape = [input_dim]
        super(Dense, self).__init__(**kwargs)

    def build(self, x):
        last_output_shape = x.output_shape  # TODO: where to find the input?????
        input_dim = self.input_dim if self.input_dim else last_output_shape[0]
        self.B = bigdl_layer.Linear(input_dim, self.output_dim)
        self.output_shape = [self.output_dim]
        super(Dense, self).build(x)


    def get_output_shape(self, input_shape=None):
        return [self.output_dim] # TODO: add assert that dense only accept 1D tensor

    # TODO: need to add tostring to every layer
    def to_string(self):
        pass

# we are suppose every layer should have a field named activation
class Activation(Layer):

    # def __init__(self, activation, **kwargs):
    #     super(Activation, self).__init__(**kwargs)
    #     self.activation = activations.get(activation)

    def __init__(self, activation_name, **kwargs):
        super(Activation, self).__init__(**kwargs)
        self.activation_name = activation_name

    def build(self, x):
        self.B = activations.get(self.activation_name)
        self.output_shape = x.output_shape
    #
    # def call(self, x, mask=None):
    #     return self.B()

class Dropout(Layer):
    """Applies Dropout to the input.

    Dropout consists in randomly setting
    a fraction `p` of input units to 0 at each update during training time,
    which helps prevent overfitting.

    # Arguments
        p: float between 0 and 1. Fraction of the input units to drop.
        noise_shape: 1D integer tensor representing the shape of the
            binary dropout mask that will be multiplied with the input.
            For instance, if your inputs ahve shape
            `(batch_size, timesteps, features)` and
            you want the dropout mask to be the same for all timesteps,
            you can use `noise_shape=(batch_size, 1, features)`.
        seed: A Python integer to use as random seed.

    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    """
    # TODO: noise_shape seed ??
    def __init__(self, p, noise_shape=None, seed=None, **kwargs):
        self.p = p
        super(Dropout, self).__init__(**kwargs)

    def _get_noise_shape(self, _):
        raise Exception("Unsupported !!")
        #return self.noise_shape

    def build(self, x):
        self.B = bigdl_layer.Dropout(init_p = self.p, inplace = False, scale = True)
        self.output_shape = x.output_shape

