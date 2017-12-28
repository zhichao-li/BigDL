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

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.keras.Dense
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}


class KerasStyleSpec extends FlatSpec with Matchers {

  "Graph: Dense + Linear" should "works correctly" in {
    val input = Input[Float](inputShape = Array(10))
    val d = new Dense[Float](20).setName("dense1").inputs(input)
    val d2 = new Dense[Float](5).setName("dense2").inputs(d)
    // mix with the old layer
    val d3 = new Linear[Float](inputSize = 5, outputSize = 30).setName("dense3").inputs(d2)
    val d4 = new Dense[Float](6).setName("dense4").inputs(d3)
    val graph = Graph[Float](input, d4)
    val inputData = Tensor[Float](Array(20, 10)).rand()
    val output = graph.forward(inputData)
  }


  "Sequential: Dense + Linear" should "works correctly" in {
    val seq = Sequential[Float]()
    val d1 = new Dense[Float](20, inputShape = Array(10)).setName("dense1")
    val d2 = new Dense[Float](5).setName("dense2")
    // mix with the old layer
    val d3 = new Linear[Float](inputSize = 5, outputSize = 30).setName("dense3")
    val d4 = new Dense[Float](6).setName("dense4")
    seq.add(d1)
    seq.add(d2)
    seq.add(d3)
    seq.add(d4)
    val inputData = Tensor[Float](Array(20, 10)).rand()
    val output = seq.forward(inputData)
    require(d3.getOutputShape().toTensor[Int].toArray().sameElements(Array(30)))
    require(d3.getInputShape().toTensor[Int].toArray().sameElements(Array(5)))
  }

  "Sequential: Linear + Dense" should "works correctly" in {
    val seq = Sequential[Float]()
    val d0 = InputLayer[Float](inputShape = Array(5))
    val d1 = new Linear[Float](inputSize = 5, outputSize = 30).setName("dense1")
    val d2 = new Dense[Float](20).setName("dense2")
    seq.add(d0)
    seq.add(d1)
    seq.add(d2)
    val inputData = Tensor[Float](Array(20, 5)).rand()
    val output = seq.forward(inputData)
    require(d2.getOutputShape().toTensor[Int].toArray().sameElements(Array(20)))
    require(d2.getInputShape().toTensor[Int].toArray().sameElements(Array(30)))
    require(seq.getOutputShape().toTensor[Int].toArray().sameElements(Array(20)))
    require(seq.getInputShape().toTensor[Int].toArray().sameElements(Array(5)))
  }

  "Sequential: pure old style" should "works correctly" in {
    val seq = Sequential[Float]()
    val d1 = new Linear[Float](inputSize = 5, outputSize = 6).setName("dense1")
    val d2 = new Linear[Float](inputSize = 6, outputSize = 7).setName("dense2")
    seq.add(d1)
    seq.add(d2)
    val inputData = Tensor[Float](Array(2, 5)).rand()
    val output = seq.forward(inputData)
  }

  // This can be removed as there are tons of such unitests
  "Graph: pure old style without compile" should "works correctly" in {
    val input = Input[Float]()
    val d = new Linear[Float](inputSize = 5, outputSize = 30).inputs(input)
    val graph = Graph[Float](input, d)
    val inputData = Tensor[Float](Array(20, 5)).rand()
    val output = graph.forward(inputData)
  }

  "Graph: pure old style with compile" should "works correctly" in {
    val input = Input[Float](inputShape = Array(5))
    val d1 = new Linear[Float](inputSize = 5, outputSize = 6).inputs(input)
    val d2 = new Linear[Float](inputSize = 6, outputSize = 7).inputs(d1)
    val graph = Graph[Float](input, d2)
    val inputData = Tensor[Float](Array(20, 5)).rand()
    val output = graph.forward(inputData)
    require(d2.element.getOutputShape().toTensor[Int].toArray().sameElements(Array(7)))
    require(d2.element.getInputShape().toTensor[Int].toArray().sameElements(Array(6)))
    require(graph.getOutputShape().toTensor[Int].toArray().sameElements(Array(7)))
    require(graph.getInputShape().toTensor[Int].toArray().sameElements(Array(5)))
  }
}
