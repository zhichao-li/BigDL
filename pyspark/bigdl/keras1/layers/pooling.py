# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import numpy as np
import bigdl.nn.layer as bigdl_layer
from bigdl.keras1.engine.topology import *
import bigdl.keras1.activations as activations
from bigdl.keras1.utility import *

class MaxPooling2D(Layer):
    """Max pooling operation for spatial data.

    # Arguments
        pool_size: tuple of 2 integers,
            factors by which to downscale (vertical, horizontal).
            (2, 2) will halve the image in each dimension.
        strides: tuple of 2 integers, or None. Strides values.
            If None, it will default to `pool_size`.
        border_mode: 'valid' or 'same'.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(nb_samples, channels, pooled_rows, pooled_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, pooled_rows, pooled_cols, channels)` if dim_ordering='tf'.
    """

    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='default', **kwargs):
        self.pool_size = pool_size
        self.border_mode = border_mode
        if strides is None:
            self.strides = self.pool_size
        super(MaxPooling2D, self).__init__(**kwargs)

    def build(self, input_shape):
        kw = self.pool_size[0]
        kh = self.pool_size[1]
        pad_w, pad_h = get_padding(kw, kh, self.border_mode)
        self.B = bigdl_layer.SpatialMaxPooling(kw,
                 kh,
                 dw=self.strides[0],  #stride width, stride height
                 dh=self.strides[1],
                 pad_w=pad_w,
                 pad_h=pad_h,
                 to_ceil=False,
                 bigdl_type="float")

    def get_output_shape_for(self, input_shape): # TODO: add unittest to check if the logic is correct. NCHW
        rows = input_shape[2]
        cols = input_shape[3]
        nb_row, nb_col = self.pool_size

        rows = conv_output_length(rows, nb_row,
                                  self.border_mode, self.strides[0])
        cols = conv_output_length(cols, nb_col,
                                  self.border_mode, self.strides[1])

        return (input_shape[0], input_shape[1], rows, cols)