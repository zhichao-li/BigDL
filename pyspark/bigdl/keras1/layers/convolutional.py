# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import numpy as np
import bigdl.nn.layer as bigdl_layer
from bigdl.keras1.engine.topology import *
import bigdl.keras1.activations as activations
from bigdl.keras1.utility import *

# https://stackoverflow.com/questions/37659538/custom-padding-for-convolutions-in-tensorflow
class Convolution2D(Layer):
    """Convolution operator for filtering windows of two-dimensional inputs.

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures.

    # Examples

    ```python
        # apply a 3x3 convolution with 64 output filters on a 256x256 image:
        model = Sequential()
        model.add(Convolution2D(64, 3, 3,
                                border_mode='same',
                                input_shape=(3, 256, 256)))
        # now model.output_shape == (None, 64, 256, 256)

        # add a 3x3 convolution on top, with 32 output filters:
        model.add(Convolution2D(32, 3, 3, border_mode='same'))
        # now model.output_shape == (None, 32, 256, 256)
    ```

    # Arguments
        nb_filter: Number of convolution filters to use.
        nb_row: Number of rows in the convolution kernel.
        nb_col: Number of columns in the convolution kernel.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)), or alternatively,
            Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass
            a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        border_mode: 'valid', 'same' or 'full'
            ('full' requires the Theano backend).
        subsample: tuple of length 2. Factor by which to subsample output.
            Also called strides elsewhere.
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
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(samples, nb_filter, new_rows, new_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, nb_filter)` if dim_ordering='tf'.
        `rows` and `cols` values might have changed due to padding.
    """
    # TODO: we only support NCHW, so we should remove dim_ordering
    # TODO: how to support mode?
    def __init__(self, nb_filter, nb_row, nb_col,
                 init='glorot_uniform', activation=None, weights=None,
                 border_mode='valid', subsample=(1, 1), dim_ordering='default',
                 W_regularizer=None, b_regularizer=None,
                 activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        if border_mode not in {'valid', 'same'}:
            raise ValueError('Invalid border mode for Convolution2D:', border_mode)
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        # self.init = initializations.get(init)
        self.b_activation = activations.get(activation)
        self.border_mode = border_mode
        self.subsample = tuple(subsample)

        self.dim_ordering = dim_ordering

        # self.W_regularizer = regularizers.get(W_regularizer)
        # self.b_regularizer = regularizers.get(b_regularizer)
        # self.activity_regularizer = regularizers.get(activity_regularizer)

        # self.W_constraint = constraints.get(W_constraint)
        # self.b_constraint = constraints.get(b_constraint)

        # self.bias = bias
        # self.initial_weights = weights
        super(Convolution2D, self).__init__(**kwargs)

    def call(self, x, mask=None):
        """This is where the layer's logic lives.

        # Arguments
            x: input tensor, or list/tuple of input tensors.
            mask: a masking tensor (or list of tensors). Used mainly in RNNs.

        # Returns:
            A tensor or list/tuple of tensors.
        """
        bigdl_tensor = self.B(bigdl_util.to_bigdl(x))
        if self.b_activation:
            bigdl_tensor = self.b_activation(bigdl_tensor)
        return Tensor(bigdl_tensor, self)

    def build(self, input_shape):
        assert(len(input_shape) == 4, "The input shape should be NCHW") # we only accept NCHW
        stack_size = input_shape[1]
        # TODO: Check order
        # if self.dim_ordering == 'th':
        #     stack_size = input_shape[1]
        # else:
        #     raise Exception("lOnly support NCHW for now, which means you should use th ordering")
        (self.pad_w, self.pad_h) = get_padding(self.nb_col, self.nb_row, self.border_mode)
        self.B = bigdl_layer.SpatialConvolution(n_input_plane=stack_size,
                 n_output_plane = self.nb_filter,
                 kernel_w=self.nb_col, # Additional zeros added to the input plane data on both sides of width axis. Default is 0. (kW-1)/2 is often used here.
                 kernel_h=self.nb_row,
                 stride_w=self.subsample[0], #TODO: the first one is col? or they always equal?
                 stride_h=self.subsample[1],
                 pad_w=self.pad_w,
                 pad_h=self.pad_h,
                 n_group=1,
                 propagate_back=True,
                 wRegularizer=None,
                 bRegularizer=None,
                 init_weight=None,
                 init_bias=None,
                 init_grad_weight=None,
                 init_grad_bias=None,
                 with_bias=True,
                 bigdl_type="float")
        self.built = True

    def get_output_shape_for(self, input_shape): #NCHW
        rows = input_shape[2]
        cols = input_shape[3]

        rows = conv_output_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])

        return (input_shape[0], self.nb_filter, rows, cols)
