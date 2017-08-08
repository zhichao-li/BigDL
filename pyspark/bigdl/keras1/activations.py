
import bigdl.nn.layer as bigdl_layer
import six
from bigdl.keras1.utility import *
from bigdl.keras1.layers.core import *

# TODO: we haven't support alpha and max_value yet
def relu(alpha=0., max_value=None):
    return bigdl_layer.ReLU()

def softmax():
    return bigdl_layer.SoftMax()

def linear():
    return bigdl_layer.Input()

def get(identifier):
    """
    :param identifier: name of activation
    :return: activation layer
    >>> a = get("relu")
    >>> assert(a, ReLU)
    """
    if identifier is None:
        return linear
    return get_from_module(identifier, globals(), 'activations', True)


def _test():
    from pyspark import SparkContext
    from bigdl.util.common import init_engine
    from bigdl.util.common import create_spark_conf
    sc = SparkContext(master="local[4]", appName="test layer",
                      conf=create_spark_conf())
    init_engine()
    a = get("relu")
    print a

if __name__ == "__main__":
    _test()