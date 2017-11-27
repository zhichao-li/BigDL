#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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
from bigdl.keras.converter import *
from bigdl.keras.optimization import *

import numpy as np


def to_sample_rdd(sc, x, y):
    from bigdl.util.common import Sample
    x_rdd = sc.parallelize(x)
    y_rdd = sc.parallelize(y)
    return x_rdd.zip(y_rdd).map(lambda item: Sample.from_ndarray(item[0], item[1]))


class KerasModelWrapper():
    def __init__(self, kmodel):
        self.bmodel = DefinitionLoader.from_kmodel(kmodel)
        WeightLoader.load_weights_from_kmodel(self.bmodel, kmodel)  # share the same weight.
        self.criterion = OptimConverter.to_bigdl_criterion(kmodel.loss)
        self.optim_method = OptimConverter.to_bigdl_optim_method(kmodel.optimizer)
        self.metrics = OptimConverter.to_bigdl_metrics(kmodel.metrics) if kmodel.metrics else None

    def evaluate_distributed(self, x, y, batch_size=32, verbose=1, sample_weight=None):
        if self.metrics:
            sc = get_spark_context()
            sample_rdd = to_sample_rdd(sc, x, y)
            return [r.result for r in self.bmodel.evaluate(sample_rdd, batch_size, self.metrics)]
        else:
            raise Exception("No Metrics found.")

    def evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None):
        raise Exception("Haven't be implemented yet")

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
        return self.bmodel.predict_local(x)

    # TODO: Support batch_size??
    def predict_distributed(self, x, batch_size=32, verbose=0):
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
        return np.asarray(self.bmodel.predict(x_rdd).collect())

    def fit(self, X, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=None,
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=0):
        if callbacks:
            raise Exception("We don't support callbacks in fit for now")
        if class_weight:
            unsupport_exp("class_weight")
        if sample_weight:
            unsupport_exp("sample_weight")
        if initial_epoch != 0:
            unsupport_exp("initial_epoch")
        if shuffle != True:
            unsupport_exp("shuffle")
        if validation_split != 0.:
            unsupport_exp("validation_split")

        sc = get_spark_context()
        bopt = boptimizer.LocalOptimizer(
            X=X,
            y=y,
            model=self.bmodel,
            criterion=self.criterion,
            end_trigger=boptimizer.MaxEpoch(nb_epoch),
            batch_size=batch_size,
            optim_method=self.optim_method,
            cores=None
        )
        # TODO: enable validation for local optimizer.
        bopt.optimize()


    def fit_distributed(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=None,
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=0):
        if callbacks:
            raise Exception("We don't support callbacks in fit for now")
        if class_weight:
            unsupport_exp("class_weight")
        if sample_weight:
            unsupport_exp("sample_weight")
        if initial_epoch != 0:
            unsupport_exp("initial_epoch")
        if shuffle != True:
            unsupport_exp("shuffle")
        if validation_split != 0.:
            unsupport_exp("validation_split")

        sc = get_spark_context()
        bopt = boptimizer.Optimizer(
            model=self.bmodel,
            training_rdd=to_sample_rdd(sc, x, y),
            criterion=self.criterion,
            end_trigger=boptimizer.MaxEpoch(nb_epoch),
            batch_size=batch_size,
            optim_method=self.optim_method
        )
        if validation_data:
            bopt.set_validation(batch_size,
                                val_rdd=to_sample_rdd(sc, *validation_data ),
                                # TODO: check if keras use the same strategy
                                trigger=boptimizer.EveryEpoch(),
                                val_method=self.metrics)
        bopt.optimize()


def with_bigdl_backend(kmodel):
        # TODO: maybe we don't need batch_size, verbose and sample_weight
    bcommon.init_engine()
    return KerasModelWrapper(kmodel)

