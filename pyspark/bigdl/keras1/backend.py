# import new
# import types # TODO: support python3
import tempfile

from bigdl.util.common import get_spark_context
from bigdl.nn.layer import Model as BModel
import bigdl.nn.layer as BLayer
import bigdl.optim.optimizer as boptimizer
import bigdl.nn.criterion as bcriterion
import bigdl.util.common as bcommon
import keras.optimizers as koptimizers
from keras.models import model_from_json


# > import types
# > x.method = types.MethodType(method, x)
from keras.models import Sequential, Model
import numpy as np


def install_bigdl_backend(kmodel):
    def bevaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None):
        sc = get_spark_context()
        sample_rdd = Model._to_sample_rdd(sc, x, y)
        return [r.result for r in self.B.test(sample_rdd, batch_size, self.metrics)]

    def bpredict(self, x, batch_size=32, verbose=0):
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
            lambda i: bcommon.Sample.from_ndarray(i, np.zeros((1))))
        return np.asarray(self.B.predict(x_rdd).collect())


    def bfit(x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=None,
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=0):
        sc = get_spark_context()
        # result = ModelLoader.to_sample_rdd(sc, x, y).collect()

        boptimizer.Optimizer(
            model=bmodel,
            training_rdd=ModelLoader.to_sample_rdd(sc, x, y),
            criterion=ModelLoader.to_bigdl_criterion(kmodel.loss),
            end_trigger=boptimizer.MaxEpoch(nb_epoch),
            batch_size=batch_size,
            optim_method=ModelLoader.to_bigdl_optim_method(kmodel.optimizer)
        ).optimize()

        # TODO: maybe we don't need batch_size, verbose and sample_weight
    bmodel = ModelLoader.to_bigdl_definition(kmodel)
    kmodel.__old_fit = kmodel.fit
    kmodel.fit = bfit
    kmodel.predict = bpredict
    kmodel.evaluate = bevaluate
    kmodel.get_weights = bmodel.get_weights

