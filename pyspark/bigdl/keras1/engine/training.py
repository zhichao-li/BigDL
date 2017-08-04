import bigdl.optim.optimizer as bigdl_optimizer
import bigdl.nn.criterion as bigdl_criterion
from bigdl.keras1.engine.topology import *
from bigdl.keras1.utility import *
from bigdl.keras1.layers.core import *
from bigdl.keras1 import optimizers
from bigdl.keras1 import activations
from bigdl.keras1 import objectives

class Model(Container):
    """The `Model` class adds training & evaluation routines to a `Container`.
    """
    def __init__(self, input, output, name=None):
        self.B = bigdl_layer.Model(Container._to_bigdl(input), Container._to_bigdl(output))
        # Handle name argument.
        if not name:
            self.name = self.B.name()
        else:
            self.name = name

    def _to_bigdl_criterion(self, loss):
        #return objectives.get(loss)
        return  bigdl_criterion.CrossEntropyCriterion()

    def _to_bigdl_optim_method(self, optimizer):
        return bigdl_optimizer.Adagrad()
        #return  optimizers.get(optimizer)

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

    def fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=None,
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=0):
        from bigdl.util.common import get_spark_context
        sc = get_spark_context()
        result = Model._to_sample_rdd(sc, x, y).collect()

        bigdl_optimizer.Optimizer(
            model = self.B,
            training_rdd = Model._to_sample_rdd(sc, x, y),
            criterion = self.criterion,
            end_trigger = bigdl_optimizer.MaxEpoch(nb_epoch),
            batch_size = batch_size,
            optim_method = self.optim_method).optimize()

    @staticmethod
    def _to_sample_rdd(sc, x, y):
        from bigdl.util.common import Sample
        x_rdd = sc.parallelize(x)
        y_rdd = sc.parallelize(y)
        return x_rdd.zip(y_rdd).map(lambda item: Sample.from_ndarray(item[0], item[1]))


class Sequential(Model):

    def __init__(self):
        self.B = bigdl_layer.Sequential()

    def add(self, layer):
        self.B.add(layer.B)
        if layer.activation:
            a = activations.get(layer.activation)
            self.B.add(a.B)
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

