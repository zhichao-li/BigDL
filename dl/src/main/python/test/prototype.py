#
# Licensed to Intel Corporation under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# Intel Corporation licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Still in experimental stage!

from nn.layer import *
from optim.optimizer import *
from util.common import *
import numpy as np

if __name__ == "__main__":
    sparkConf = calc_spark_conf(1, 4)
    sc = SparkContext(master="local[*]", appName="test model", conf=sparkConf)
    conf = initEngine(1, 4)
    FEATURES_DIM = 2

    def gen_rand_sample():
        features = np.random.uniform(0, 1, (FEATURES_DIM))
        label = (2*features).sum() + 0.4
        return PySample.from_ndarray(features, label)

    trainingData = sc.parallelize(range(0, 100)).map(
        lambda i: gen_rand_sample())

    model = Sequential()
    model.add(Linear(FEATURES_DIM, 1, "Xavier"))
    state = {"learningRate": 0.01,
             "learningRateDecay": 0.0002}
    optimizer = Optimizer(
        model=model,
        training_rdd=trainingData,
        criterion=MSECriterion(),
        optim_method="Adagrad",
        state=state,
        end_trigger=MaxEpoch(5),
        batch_size=32)
    optimizer.setvalidation(
        batch_size=32,
        val_rdd=trainingData,
        trigger=EveryEpoch(),
        val_method=["top1"]
    )
    optimizer.setcheckpoint(SeveralIteration(1), "/tmp/prototype/")

    trained_model = optimizer.optimize()
    parameters = trained_model.parameters()
    print("parameters %s" % parameters["weights"])
    predict_result = trained_model.predict(trainingData)
    p = predict_result.take(5)
    print("predict predict: \n")
    for i in p:
        print(str(i) + "\n")
    print(len(p))
    sc.stop()
