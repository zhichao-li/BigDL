import bigdl.optim.optimizer as bigdl_optimizer
import bigdl.nn.criterion as bigdl_criterion
from bigdl.keras1.engine.topology import *
from bigdl.keras1.utility import *
from bigdl.keras1.layers.core import *
from bigdl.keras1 import optimizers
from bigdl.keras1 import activations
from bigdl.keras1 import objectives
from bigdl.util.common import get_spark_context
import bigdl.keras1.utils.bigdl_util as bigdl_util


class Model(Container):
    """The `Model` class adds training & evaluation routines to a `Container`.
    """
    name_to_node = {}

    # @property
    # def layers(self):
    #     return self.topological_sort()

    # This method is for visualize or debug only
    # Get a topological sorted layers in the current graph
    # def topological_sort(self):
    #     stack = []
    #     visited = set()
    #     def dfs(layer, visited, stack):
    #         if layer not in visited:
    #             visited.add(layer)
    #             print visited
    #             for node in layer.inbound_keras_nodes:
    #                 dfs(node.keras_layer, visited, stack)
    #         stack.append(layer)
    #
    #     for output_i in bigdl_common.to_list(self.outputs):
    #         if output_i.keras_layer not in visited:
    #             dfs(output_i.keras_layer, visited, stack)
    #     return stack


    def nodes(self):
        return self.B.executions()

    # output is a kerasNode or list of kerasNode
    def __init__(self, input, output, name=None):
        self.inputs = input
        self.outputs = output
        self.B = bigdl_layer.Model(bigdl_util.to_bigdl(self.inputs), bigdl_util.to_bigdl(self.outputs))
        super(Model, self).__init__(input, output, name)

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
        # create the underlying model
        self.build()
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

# class Sequential(Model):
#
#     def __init__(self, name=None):
#         self.added_nodes = []
#         self.B = bigdl_layer.Sequential()
#
#     # TODO: we can remove this if implement Sequential base on Graph
#     def nodes(self):
#         return self.added_nodes
#
#     def add(self, layer):
#         node = None
#         if len(self.added_nodes) == 0:
#             if not hasattr(layer, "batch_input_shape"):
#                 raise Exception("you should specify input_shape for first layer")
#             input = Input(shape=layer.batch_input_shape[1:])
#             node = layer(input)
#         else:
#             node = layer(self.added_nodes[-1])
#         self.added_nodes.append(node)
#         self.B.add(node.B.element())
#         return self


class Sequential(Model):
    """Linear stack of layers.

    # Arguments
        layers: list of layers to add to the model.

    # Note
        The first layer passed to a Sequential model
        should have a defined input shape. What that
        means is that it should have received an `input_shape`
        or `batch_input_shape` argument,
        or for some type of layers (recurrent, Dense...)
        an `input_dim` argument.

    # Example

        ```python
            model = Sequential()
            # first layer must have a defined input shape
            model.add(Dense(32, input_dim=500))
            # afterwards, Keras does automatic shape inference
            model.add(Dense(32))

            # also possible (equivalent to the above):
            model = Sequential()
            model.add(Dense(32, input_shape=(500,)))
            model.add(Dense(32))

            # also possible (equivalent to the above):
            model = Sequential()
            # here the batch dimension is None,
            # which means any batch size will be accepted by the model.
            model.add(Dense(32, batch_input_shape=(None, 500)))
            model.add(Dense(32))
        ```
    """

    def __init__(self, layers=None, name=None):
        self.layers = []  # stack of layers
        self.model = None  # internal Model instance
        self.inputs = []  # tensors
        self.outputs = []  # tensors (length 1)
        self._trainable = True

        # model attributes
        self.inbound_nodes = []
        self.outbound_nodes = []
        self.built = False
        self._flattened_layers = None

        if not name:
            prefix = 'sequential_'
            # name = prefix + str(K.get_uid(prefix))
            name = prefix + bigdl_util.get_uid()
        self.name = name

        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        """Adds a layer instance on top of the layer stack.

        # Arguments
            layer: layer instance.
        """
        if not isinstance(layer, Layer):
            raise TypeError('The added layer must be '
                            'an instance of class Layer. '
                            'Found: ' + str(layer))
        if not self.outputs:
            # first layer in model: check that it is an input layer
            if len(layer.inbound_nodes) == 0:
                # create an input layer
                if not hasattr(layer, 'batch_input_shape'):
                    raise ValueError('The first layer in a '
                                     'Sequential model must '
                                     'get an `input_shape` or '
                                     '`batch_input_shape` argument.')
                batch_input_shape = layer.batch_input_shape
                if hasattr(layer, 'input_dtype'):
                    input_dtype = layer.input_dtype
                else:
                    input_dtype = None
                layer.create_input_layer(batch_input_shape, input_dtype)

            if len(layer.inbound_nodes) != 1:
                raise ValueError('A layer added to a Sequential model must '
                                 'not already be connected somewhere else. '
                                 'Model received layer ' + layer.name +
                                 ' which has ' +
                                 str(len(layer.inbound_nodes)) +
                                 ' pre-existing inbound connections.')

            if len(layer.inbound_nodes[0].output_tensors) != 1:
                raise ValueError('All layers in a Sequential model '
                                 'should have a single output tensor. '
                                 'For multi-output layers, '
                                 'use the functional API.')

            self.outputs = [layer.inbound_nodes[0].output_tensors[0]]
            self.inputs = get_source_inputs(self.outputs[0])

            # We create an input node, which we will keep updated
            # as we add more layers
            Node(outbound_layer=self,
                 inbound_layers=[],
                 node_indices=[],
                 tensor_indices=[],
                 input_tensors=self.inputs,
                 output_tensors=self.outputs,
                 # no model-level masking for now
                 input_masks=[None for _ in self.inputs],
                 output_masks=[None],
                 input_shapes=[x._keras_shape for x in self.inputs],
                 output_shapes=[self.outputs[0]._keras_shape])
        else:
            output_tensor = layer(self.outputs[0])
            if isinstance(output_tensor, list):
                raise TypeError('All layers in a Sequential model '
                                'should have a single output tensor. '
                                'For multi-output layers, '
                                'use the functional API.')
            self.outputs = [output_tensor]
            # update self.inbound_nodes
            self.inbound_nodes[0].output_tensors = self.outputs
            self.inbound_nodes[0].output_shapes = [self.outputs[0]._keras_shape]

        self.layers.append(layer)
        self.built = False
        self._flattened_layers = None

    def get_layer(self, name=None, index=None):
        """Returns a layer based on either its name (unique)
        or its index in the graph. Indices are based on
        order of horizontal graph traversal (bottom-up).

        # Arguments
            name: string, name of layer.
            index: integer, index of layer.

        # Returns
            A layer instance.
        """
        if not self.built:
            self.build()
        return self.model.get_layer(name, index)

    def call(self, x, mask=None):
        # if not self.built:
        #     self.build()
        # return self.model.call(x, mask)
        raise Exception("Haven't supported yet")

    def build(self, input_shape=None):
        if not self.inputs or not self.outputs:
            raise TypeError('Sequential model cannot be built: model is empty.'
                            ' Add some layers first.')
        # actually create the model
        self.model = Model(self.inputs, self.outputs[0],
                           name=self.name + '_model')
        self.model.trainable = self.trainable

        # mirror model attributes
        self.supports_masking = self.model.supports_masking
        self._output_mask_cache = self.model._output_mask_cache
        self._output_tensor_cache = self.model._output_tensor_cache
        self._output_shape_cache = self.model._output_shape_cache
        self.input_layers = self.model.input_layers
        self.input_layers_node_indices = self.model.input_layers_node_indices
        self.input_layers_tensor_indices = self.model.input_layers_tensor_indices
        self.output_layers = self.model.output_layers
        self.output_layers_node_indices = self.model.output_layers_node_indices
        self.output_layers_tensor_indices = self.model.output_layers_tensor_indices
        self.nodes_by_depth = self.model.nodes_by_depth
        self.container_nodes = self.model.container_nodes
        self.output_names = self.model.output_names
        self.input_names = self.model.input_names

        # make sure child model callbacks will call the parent Sequential model:
        # self.model.callback_model = self
        self.B = bigdl_layer.Sequential()
        for layer in self.layers:
            self.B.add(layer.B)
        self.built = True

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        if self.model:
            self.model.trainable = value
        self._trainable = value

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

