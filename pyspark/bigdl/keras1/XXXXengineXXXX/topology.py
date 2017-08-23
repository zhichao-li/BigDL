

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
        self.inbound_keras_nodes = []
        self.outbound_keras_nodes = []

    @property
    def name(self):
        return self.B.name()

    def inbound_nodes(self):
        # from .training import Model
        # return [Model.name_to_node[node] for node in self.B.inbound_nodes()]
        return self.inbound_keras_nodes

    def outbound_nodes(self):
        # from .training import Model
        # return [Model.name_to_node[node] for node in self.B.outbound_nodes()]
        return self.outbound_keras_nodes

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
        if "name" in kwargs:
            self._name = kwargs["name"]
        else:
            self._name = None

        self.inbound_keras_nodes = []
        self.outbound_keras_nodes = []

    @property
    def weights(self): # in keras this would return list of Variable
        return self.get_weights()

    def get_weights(self): #in keras this would return list of ndarray
        assert self.built == True, "we should build this layer first before getting weights"
        return self.B.get_weights()


    def set_weights(self, weights):
        """Sets the weights of the layer, from Numpy arrays.

        # Arguments
            weights: a list of Numpy arrays. The number
                of arrays and their shape must match
                number of the dimensions of the weights
                of the layer (i.e. it should match the
                output of `get_weights`).[weights, bias]
        """
        assert self.built == True, "we should build this layer first before setting weights"
        self.B.set_weights(weights)

    @property
    def name(self):
        return self.B.name()

    def build(self):
        pass

    # input_shape can be a tuple or list of tuple
    def get_output_shape_for(self, input_shape=None):
        return input_shape

    # x is a KerasNode
    # return a KerasNode
    # after calling thie a new field BNode is added
    #input_shape output_shpape should always go in here instead of the subclass
    #This method should not be override by subclass
    def __call__(self, x=None, mask=None):
        if x is None:
            raise Exception("x should not be None")
        if not self.built:
            self.input_shapes = [i.output_shape for i in bigdl_common.to_list(x)]
            self.input_shape = self.input_shapes[0]
            self.output_shape = self.get_output_shape_for(self.input_shapes if len(self.input_shapes) > 1 else self.input_shape)
            self.build()
            self.built = True
            if self._name: # need to be called after built.
                self.B.set_name(self._name)
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
        Model.name_to_node[kerasNode.name] = kerasNode
        # Maintain graph for keras nodes.
        kerasNode.inbound_keras_nodes += bigdl_common.to_list(x)
        [x_inode.outbound_keras_nodes.append(kerasNode) for x_inode in bigdl_common.to_list(x)]
        self.inbound_keras_nodes += bigdl_common.to_list(x)
        self.outbound_keras_nodes.append(kerasNode)
        return kerasNode

def Input(shape, dtype=None, **kwargs):
    return InputLayer(shape, dtype, **kwargs)()

class InputLayer(Layer):

    def __init__(self, shape, dtype=None, **kwargs):
        super(InputLayer, self).__init__(**kwargs)
        self.input_shape = (None,) + tuple(shape)
        self.output_shape = (None,) + tuple(shape)
        self.inbound_keras_nodes = []
        self.outbound_keras_nodes = []

    def inbound_nodes(self):
        return []

    def __call__(self):
        from bigdl.keras1.engine.training import Model
        self.B = bigdl_layer.Input()  # special here, B in the other keras layer is bigdl layer instead of Node.
        kerasNode = KerasNode(self.B, self)
        Model.name_to_node[self.name] = kerasNode
        self.outbound_keras_nodes.append(kerasNode)
        return kerasNode

class Container(Layer):
    @staticmethod
    def _to_bigdl(x):
        from bigdl.util.common import to_list
        return [i.B for i in to_list(x)]

    def get_weights(self):
        """Returns the weights of the model,
        as a flat list of Numpy arrays.
        """
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights

    def set_weights(self, weights):
        """Sets the weights of the model.
        The `weights` argument should be a list
        of Numpy arrays with shapes and types matching
        the output of `model.get_weights()`.
        """
        tuples = []
        for layer in self.layers:
            nb_param = len(layer.weights)
            layer_weights = weights[:nb_param]
            layer.keras_layer.set_weights(layer_weights)
            weights = weights[nb_param:]


