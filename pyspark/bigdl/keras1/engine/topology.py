

import bigdl.nn.layer as bigdl_layer
import bigdl.nn.criterion as bigdl_criterion
import bigdl.util.common as bigdl_common

class Layer(object):

    def __init__(self, **kwargs):
        self.activation = None
        self.built = False

        if "input_shape" in kwargs:
            self.input_shape = kwargs["input_shape"]

    # x is a BigDLHolder
    def build(self, x):
        if self.activation:
            self.activation()

    # this shape doesn't include sample number as the first dimension.
    def get_output_shape(self, input_shape=None):
        if input_shape is None and "input_shape" in dir(self) and self.input_shape:
            return self.input_shape
        return bigdl_common.to_list(input_shape)

    # x is a BigDLHolder in BigDL
    # return a bigdlholder
    def call(self, x=None, mask=None):
        if not self.built:
            self.build(x)
            built = True
        if x is None: # only Input would go into this condition.
            return self.B
        temp_node = self.B(x.B)
        # assert input is a Node
        if self.activation: # activate is a keras's Activation object, need to invoke call() to init it.
            temp_node = self.activation.B(temp_node)
        return temp_node

    def __call__(self, x=None, mask=None):
       return BigDLHolder(self.call(x, mask), self)

# TODO: does it need to inherit from Layer?
class BigDLHolder(Layer):
    def __init__(self, bigdl_layer, keras_layer=None):
        self.K = keras_layer
        self.B = bigdl_layer
        self.output_shape = self.K.output_shape

    def call(self, x=None, mask=None):
        return self(x)

    # seems like we can delete this and inference not require it.
    # we should maitaine a field for shape inference(this field would be updated each time we add a new input), and an array for multiple input
    def get_output_shape(self, input_shape=None):
        return self.K.get_output_shape(input_shape) if self.K else input_shape


class Container(Layer):
    @staticmethod
    def _to_bigdl(x):
        from bigdl.util.common import to_list
        return [i.B for i in to_list(x)]


