import bigdl.optim.optimizer as bigdl_optimizer
import bigdl.nn.criterion as bigdl_criterion
from bigdl.keras1.engine.topology import *

class Model(Container):
    """The `Model` class adds training & evaluation routines to a `Container`.
    """
    def __init__(self, input, output, name=None):
        self.value = bigdl_layer.Model(Container._to_bigdl(input), Container._to_bigdl(output))
        # Handle name argument.
        if not name:
            self.name = self.value.name()

    def _to_bigdl_criterion(self, loss):
        return bigdl_criterion.ClassNLLCriterion()
        #return  bigdl_criterion.CrossEntropyCriterion()
    def _to_bigdl_optim_method(self, optimizer):
        return  bigdl_optimizer.Adagrad(learningrate=0.01, learningrate_decay=0.0002)

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

        self.optim_method = self._to_bigdl_optim_method(optimizer)
        self.criterion = self._to_bigdl_criterion(loss)

    def fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=None,
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=0):
        from bigdl.util.common import get_spark_context
        sc = get_spark_context()
        for sample in Model._to_sample_rdd(sc, x, y).take(10):
            print "features", sample.features.shape
            print "label", sample.label.shape
        bigdl_optimizer.Optimizer(
            model = self.value,
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
    pass

def _test():
    from pyspark import SparkContext
    from bigdl.keras1.layers.core import Dense
    from bigdl.util.common import init_engine
    from bigdl.util.common import create_spark_conf
    sc = SparkContext(master="local[4]", appName="test layer",
                      conf=create_spark_conf())
    init_engine()
    dense1 = Dense(4, input_dim = 20)()
    dense2 = Dense(2, input_dim = 4)(dense1)
    model = Model(input = [dense1], output= [dense2])
    model.compile(optimizer="SGD", loss="classnll")

    # generate dummy data
    import numpy as np
    data = np.random.random((100, 20))
    labels = np.random.randint(low=1, high=2, size=(100, 2))

    model.fit(x=data, y=labels, batch_size=32, nb_epoch=10)

if __name__ == "__main__":
    _test()