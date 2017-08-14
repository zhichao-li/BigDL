

import bigdl.nn.layer as bigdl_layer
import bigdl.nn.criterion as bigdl_criterion
import bigdl.util.common as bigdl_common

class KerasNode(object):

    def __init__(self, bigdl_node, keras_layer):
        self.B = bigdl_node
        self.keras_layer = keras_layer
        self.input_shape = keras_layer.input_shape
        if keras_layer.input_shapes:
            self.input_shape = keras_layer.input_shapes
        self.output_shape = keras_layer.output_shape

    def name(self):
        return self.B.name()

    def inbound_nodes(self):
        from .training import Model
        return [Model.name_to_node[node] for node in self.B.inbound_nodes()]

    def outbound_nodes(self):
        from .training import Model
        return [Model.name_to_node[node] for node in self.B.outbound_nodes()]

    def forward(self, input):
        return self.B.element().forward(input)

# we don't support output_shape, output_mask, arguments
#inputs are KerasNodes.
# return KerasNode
def merge(inputs, mode='sum', concat_axis=-1, dot_axes=-1, name=None):
    if mode == 'concat':
        from ..layers.core import Concat
        return Concat(concat_axis=concat_axis, name = name)(inputs)
    else:
        raise Exception("unsupported " + mode)

class Layer(object):

    def __init__(self, **kwargs):
        self.activation = None
        self.built = False
        if "input_shape" in kwargs:
            self.input_shape = kwargs["input_shape"]
        self.input_shapes = None

    def name(self):
        return self.B.name()

    def build(self):
        pass

    # input_shape can be a tuple or list of tuple
    def get_output_shape(self, input_shape=None):
        return input_shape

    # x is a BigDLHolder in BigDL
    # return a bigdlholder
    # after calling thie a new field BNode is added
    #input_shape output_shpape should always go in here instead of the subclass
    def __call__(self, x=None, mask=None):
        if x is None:
            raise Exception("x should not be None")
        if not self.built:
            self.input_shapes = [i.output_shape for i in bigdl_common.to_list(x)]
            self.input_shape = self.input_shapes[0]
            self.output_shape = self.get_output_shape(self.input_shapes if len(self.input_shapes) > 1 else self.input_shape)
            self.build()
            self.built = True
        from .training import Model
        if isinstance(x, list):
            b_input = [i.B for i in x]
        else:
            b_input = x.B
        temp_node = self.B(b_input)
        # assert input is a Node
        if self.activation:  # activate is a keras's Activation object, need to invoke call() to init it.
            temp_node = self.activation.B(temp_node)
        kerasNode = KerasNode(temp_node, self)
        Model.name_to_node[temp_node.name()] = kerasNode
        return kerasNode

# TODO: does it need to inherit from Layer?
# class BigDLHolder(Layer):
#     def __init__(self, bigdl_layer, keras_layer=None):
#         self.K = keras_layer
#         self.B = bigdl_layer
#         self.output_shape = self.K.output_shape

    # def call(self, x=None, mask=None):
    #     return self(x)

    # seems like we can delete this and inference not require it.
    # we should maitaine a field for shape inference(this field would be updated each time we add a new input), and an array for multiple input
    # def get_output_shape(self, input_shape=None):
    #     return self.K.get_output_shape(input_shape) if self.K else input_shape


class Container(Layer):
    @staticmethod
    def _to_bigdl(x):
        from bigdl.util.common import to_list
        return [i.B for i in to_list(x)]


