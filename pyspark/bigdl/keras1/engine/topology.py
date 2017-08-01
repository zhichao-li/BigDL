

import bigdl.nn.layer as bigdl_layer
import bigdl.nn.criterion as bigdl_criterion



class Layer(object):
    pass


class Container(Layer):
    @staticmethod
    def _to_bigdl(x):
        from bigdl.util.common import to_list
        return [i.value for i in to_list(x)]



class Input():
    pass #self.value = bigdl_layer.Input()

