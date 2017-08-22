import bigdl.optim.optimizer as bigdl_optimizer
import bigdl.nn.criterion as bigdl_criterion
from bigdl.keras1.engine.topology import *
from bigdl.keras1.utility import *
from bigdl.keras1.layers.core import *
from bigdl.keras1 import optimizers
from bigdl.keras1 import activations
from bigdl.keras1 import objectives
from bigdl.util.common import get_spark_context


class Model(Container):
    """The `Model` class adds training & evaluation routines to a `Container`.
    """
    name_to_node = {}

    @property
    def layers(self):
        return self.topological_sort()

    # This method is for visualize or debug only
    # Get a topological sorted layers in the current graph
    def topological_sort(self):
        stack = []
        visited = set()
        def dfs(layer, visited, stack):
            if layer not in visited:
                visited.add(layer)
                print visited
                for node in layer.inbound_keras_nodes:
                    dfs(node.keras_layer, visited, stack)
            stack.append(layer)

        for output_i in bigdl_common.to_list(self.output):
            if output_i.keras_layer not in visited:
                dfs(output_i.keras_layer, visited, stack)
        return stack


    def nodes(self):
        return [Model.name_to_node[e] for e in self.B.executions()]

    # output is a kerasNode or list of kerasNode
    def __init__(self, input, output, name=None):
        self.output = output
        self.B = bigdl_layer.Model(Container._to_bigdl(input), Container._to_bigdl(output))

    def _to_bigdl_criterion(self, loss):
        #return objectives.get(loss)
        return  bigdl_criterion.CrossEntropyCriterion()

    def _to_bigdl_optim_method(self, optimizer):
        return bigdl_optimizer.Adagrad()
        #return  optimizers.get(optimizer)

    def _to_bigdl_metrics(self, metrics):
        if not metrics:
            return []
        return [bigdl_optimizer.Top1Accuracy() if m == "accuracy" or m == "acc" else None for m in metrics] # TODO: fix return None


    def compile(self, optimizer, loss, metrics=None, loss_weights=None,
                sample_weight_mode=None, **kwargs):
        """Configures the model for training.
        # Arguments
            optimizer: str (name of optimizer) or optimizer object.
                See [optimizers](/optimizers).
            loss: str (name of objective function) or objective function.
                See [losses](/losses).
                If the model has multiple outputs, you can use a different loss
                on each output by passing a dictionary or a list of losses.
                The loss value that will be minimized by the model
                will then be the sum of all individual losses.
            metrics: list of metrics to be evaluated by the model
                during training and testing.
                Typically you will use `metrics=['accuracy']`.
                To specify different metrics for different outputs of a
                multi-output model, you could also pass a dictionary,
                such as `metrics={'output_a': 'accuracy'}`.
            loss_weights: Optional list or dictionary specifying scalar
                coefficients (Python floats) to weight the loss contributions
                of different model outputs.
                The loss value that will be minimized by the model
                will then be the *weighted sum* of all individual losses,
                weighted by the `loss_weights` coefficients.
                If a list, it is expected to have a 1:1 mapping
                to the model's outputs. If a tensor, it is expected to map
                output names (strings) to scalar coefficients.
            sample_weight_mode: if you need to do timestep-wise
                sample weighting (2D weights), set this to `"temporal"`.
                `None` defaults to sample-wise weights (1D).
                If the model has multiple outputs, you can use a different
                `sample_weight_mode` on each output by passing a
                dictionary or a list of modes.
            **kwargs: when using the Theano/CNTK backends, these arguments
                are passed into K.function. When using the TensorFlow backend,
                these arguments are passed into `tf.Session.run`.
        # Raises
            ValueError: In case of invalid arguments for
                `optimizer`, `loss`, `metrics` or `sample_weight_mode`.
        """

        self.optim_method = optimizers.get(optimizer)
        self.criterion = objectives.get(loss)
        self.metrics = self._to_bigdl_metrics(metrics)

    def fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=None,
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=0):
        sc = get_spark_context()
        result = Model._to_sample_rdd(sc, x, y).collect()

        bigdl_optimizer.Optimizer(
            model = self.B,
            training_rdd = Model._to_sample_rdd(sc, x, y),
            criterion = self.criterion,
            end_trigger = bigdl_optimizer.MaxEpoch(nb_epoch),
            batch_size = batch_size,
            optim_method = self.optim_method).optimize()

    # TODO: maybe we don't need batch_size, verbose and sample_weight
    def evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None):
        sc = get_spark_context()
        sample_rdd = Model._to_sample_rdd(sc, x, y)
        return [r.result for r in self.B.test(sample_rdd, batch_size, self.metrics)]

    def predict(self, x, batch_size=32, verbose=0):
        """Generates output predictions for the input samples,
        processing the samples in a batched way.

        # Arguments
            x: the input data, as a Numpy array
                (or list of Numpy arrays if the model has multiple outputs).
            batch_size: integer.
            verbose: verbosity mode, 0 or 1.

        # Returns
            A Numpy array of predictions.
        """
        sc = get_spark_context()
        x_rdd = sc.parallelize(x).map(
            lambda i: bigdl_common.Sample.from_ndarray(i, np.zeros((1))))
        return np.asarray(self.B.predict(x_rdd).collect())

    @staticmethod
    def _to_sample_rdd(sc, x, y):
        from bigdl.util.common import Sample
        x_rdd = sc.parallelize(x)
        y_rdd = sc.parallelize(y)
        return x_rdd.zip(y_rdd).map(lambda item: Sample.from_ndarray(item[0], item[1]))

# we should at least get the graph structure from python side.
# class Sequential(Model):
#
#     def __init__(self, name=None):
#         self.layers = []
#         super(Sequential, self).__init__(input=[], output=[], name=name)
#
#     def __call__(self, input, output):
#         self.B = Model(input=input, output=output).B
#         return self
#
#     def add(self, layer):
#         node = None
#         if len(self.layers) == 0:
#             if "input_shape" not in dir(layer) or layer.input_shape is None:
#                 raise Exception("you should specify input_shape for first layer")
#             input = Input(input_shape=layer.input_shape)()
#             self.layers.append(input)
#             node = layer(input)
#         else:
#             node = layer(self.layers[-1])
#         self.layers.append(node)
#         return self(input=[self.layers[0]], output=[self.layers[-1]])

class Sequential(Model):

    def __init__(self, name=None):
        self.added_nodes = []
        self.B = bigdl_layer.Sequential()

    # TODO: we can remove this if implement Sequential base on Graph
    def nodes(self):
        return self.added_nodes

    def __call__(self, input, output):
        self.B = Model(input=input, output=output).B
        return self

    def add(self, layer):
        node = None
        if len(self.added_nodes) == 0:
            if "input_shape" not in dir(layer) or layer.input_shape is None:
                raise Exception("you should specify input_shape for first layer")
            input = Input(shape=layer.input_shape)
            node = layer(input)
        else:
            node = layer(self.added_nodes[-1])
        self.added_nodes.append(node)
        self.B.add(node.B.element())
        return self



def _test():
    from pyspark import SparkContext
    from bigdl.keras1.layers.core import Dense
    from bigdl.util.common import init_engine
    from bigdl.util.common import create_spark_conf
    sc = SparkContext(master="local[4]", appName="test layer",
                      conf=create_spark_conf())
    init_engine()


if __name__ == "__main__":
    _test()

