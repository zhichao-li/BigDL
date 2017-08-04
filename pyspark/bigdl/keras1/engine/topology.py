

import bigdl.nn.layer as bigdl_layer
import bigdl.nn.criterion as bigdl_criterion

class Layer(object):

    def __init__(self, **kwargs):
        self.activation = None
        self.built = False

    def build(self):
        pass

    # x is a BigDLHolder in BigDL
    def call(self, x=None, mask=None):
        if not self.built:
            self.build()
        temp_node = self.B(x.B)
        # assert input is a Node
        if self.activation: # activate is a string or object?
            temp_node = self.activation.B(temp_node)
        return temp_node

    def __call__(self, x=None, mask=None):
       return BigDLHolder(self.call(x, mask))


class BigDLHolder(Layer):
    def __init__(self, bigdl_layer):
        self.B = bigdl_layer

    def call(self, x=None, mask=None):
        return self(x)


class Container(Layer):
    @staticmethod
    def _to_bigdl(x):
        from bigdl.util.common import to_list
        return [i.B for i in to_list(x)]


