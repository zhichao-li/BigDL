/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.keras.nn

import com.intel.analytics.bigdl.example.loadmodel.AlexNet_OWT
import com.intel.analytics.bigdl.nn.abstractnn.InvalidLayer
import com.intel.analytics.bigdl.nn.keras.{Activation, Dense, Input, InputLayer, Model, Sequential => KSequential}
import com.intel.analytics.bigdl.nn.{Sequential => TSequential, _}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{BigDLSpecHelper, Shape}


class KerasStyleSpec extends BigDLSpecHelper {

  "Graph: Dense" should "work correctly" in {
    val input = Input[Float](inputShape = Shape(10))
    val d = Dense[Float](20, activation = "relu").setName("dense1").inputs(input)
    val d2 = Dense[Float](5).setName("dense2").inputs(d)
    val model = Model[Float](input, d2)
    val inputData = Tensor[Float](Array(20, 10)).rand()
    val output = model.forward(inputData)
    require(model.getOutputShape().toSingle().sameElements(Array(-1, 5)))
    require(model.getInputShape().toSingle().sameElements(Array(-1, 10)))
  }

  "Sequential: Dense" should "work correctly" in {
    val seq = KSequential[Float]()
    val d1 = Dense[Float](20, inputShape = Shape(10)).setName("dense1")
    val d2 = Dense[Float](5).setName("dense2")
    val d3 = Dense[Float](6).setName("dense4")
    seq.add(d1)
    seq.add(d2)
    seq.add(d3)
    val inputData = Tensor[Float](Array(20, 10)).rand()
    val output = seq.forward(inputData)
    require(d3.getOutputShape().toSingle().sameElements(Array(-1, 6)))
    require(d3.getInputShape().toSingle().sameElements(Array(-1, 5)))
  }

  "Frozen sequential" should "be tested" in {
    intercept[RuntimeException] {
      val seq = KSequential[Float]()
      val seq1 = KSequential[Float]()
      seq.add(seq1)
      seq1.add(Dense[Float](20, inputShape = Shape(10)))
    }
  }

  "Sequential: shared relu" should "work correctly" in {
    val sharedRelu = ReLU[Float]()
    val seq1 = KSequential[Float]()
    seq1.add(Dense[Float](20, inputShape = Shape(10)))
    seq1.add(sharedRelu)
    require(seq1.getOutputShape().toSingle().sameElements(Array(-1, 20)))

    val seq2 = KSequential[Float]()
    seq2.add(Dense[Float](5, inputShape = Shape(20)))
    seq2.add(sharedRelu)
    require(seq2.getOutputShape().toSingle().sameElements(Array(-1, 5)))

    val seq = KSequential[Float]()
    seq.add(seq1)
    seq.add(seq2)

    val inputData = Tensor[Float](Array(20, 10)).rand()
    val output = seq.forward(inputData)
    require(seq.getInputShape().toSingle().sameElements(Array(-1, 10)))
    require(seq.getOutputShape().toSingle().sameElements(Array(-1, 5)))
  }

  "TSequential" should "work with alex" in {
    val model = AlexNet_OWT(1000, false, true)
    TSequential[Float].add(model)
  }

  "TSequential" should "not work with dense" in {
    intercept[InvalidLayer] {
      val seq = TSequential[Float]()
      val d1 = Dense[Float](20, inputShape = Shape(10)).setName("dense1")
      seq.add(d1)
    }
  }

  "Incompatible inputShape" should "not work" in {
    intercept[RuntimeException] {
      val seq = KSequential[Float]()
      val d1 = Dense[Float](20, inputShape = Shape(10)).setName("dense1")
      seq.add(InputLayer(inputShape = Shape(5)))
      seq.add(d1)
    }
  }

  "TGraph" should "not work with dense" in {
    intercept[InvalidLayer] {
      val d1 = Dense[Float](20).setName("dense1").inputs(Input(inputShape = Shape(10)))
      val l1 = Linear(2, 3).inputs(d1)
    }
  }


  "KGraph" should "not work with linear and seq" in {
    intercept[InvalidLayer] {
      val input = Input(inputShape = Shape(10))
      val l1 = Linear(10, 3).inputs(input)
      val seq = TSequential[Float]().inputs(l1)
      val l2 = Linear(3, 4).inputs(seq)
      Model(input, l2)
    }
  }

  "TSequential" should "not works with container containing Dense" in {
    val seq = TSequential[Float]()
    intercept[InvalidLayer] {
      val parallelTable = ParallelTable[Float]()
      val d1 = Dense[Float](20, inputShape = Shape(10)).setName("dense1")
      parallelTable.add(d1)
      seq.add(parallelTable)
    }
  }

  "TSequential" should "not work with container with dense" in {
    intercept[InvalidLayer] {
      val seq = TSequential[Float]()
      val seq2 = TSequential[Float]()
      val d1 = Dense[Float](20, inputShape = Shape(10)).setName("dense1")
      seq2.add(d1)
      seq.add(seq2)
    }
  }

  "save and reload model" should "work correctly" in {
    val input = Input[Float](inputShape = Shape(10))
    val d = Dense[Float](20).setName("dense1").inputs(input)
    val d2 = Dense[Float](5).setName("dense2").inputs(d)
    val model = Model[Float](input, d2)
    val tmpFile = createTmpFile()
    val absPath = tmpFile.getAbsolutePath
    tmpFile.delete()
    model.saveModule(absPath)
    val reloadedModel = Module.loadModule(absPath)
    val inputData = Tensor[Float](Array(20, 10)).rand()
    val output = reloadedModel.forward(inputData)
  }

  "save and reload sequential" should "work correctly" in {
    val kseq = KSequential[Float]()
    val d1 = Dense[Float](20, inputShape = Shape(10)).setName("dense1")
    val d2 = Dense[Float](5).setName("dense2")
    kseq.add(d1)
    kseq.add(d2)
    val tmpFile = createTmpFile()
    val absPath = tmpFile.getAbsolutePath
    tmpFile.delete()
    kseq.saveModule(absPath)
    val reloadedModel = Module.loadModule(absPath)
    val inputData = Tensor[Float](Array(20, 10)).rand()
    val output = reloadedModel.forward(inputData)
  }

  "multiple outputs with index" should "be test" in {
    val input = Input[Float](inputShape = Shape(10))
    val d1 = Dense[Float](20).setName("dense1").inputs(input)
    val d2 = Dense[Float](5).setName("dense2").inputs(input)
    val multiOutput = Model[Float](input, Array(d1, d2)).inputs(input)

    val relu1 = Activation[Float]("relu").inputs(multiOutput(1))
    val model = Model[Float](input, relu1)
    model.forward(Tensor[Float](Array(2, 10)).rand())
    assert(model.getOutputShape().toSingle().sameElements(Array(-1, 20)))

    val relu2 = Activation[Float]("relu").inputs(multiOutput(2))
    val model2 = Model[Float](input, relu2)
    model2.forward(Tensor[Float](Array(2, 10)).rand())
    assert(model2.getOutputShape().toSingle().sameElements(Array(-1, 5)))

  }
}
