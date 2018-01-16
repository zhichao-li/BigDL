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

package com.intel.analytics.bigdl.nn.keras

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class Activation[T: ClassTag](var activation: String,
                              var inputShape: Array[Int] = null)
(implicit ev: TensorNumeric[T]) extends KerasModule[Tensor[T], Tensor[T], T](inputShape) {

  override def doBuild(inputShape: Activity): AbstractModule[Tensor[T], Tensor[T], T] = {
    activation = activation.toLowerCase()
    var layer: TensorModule[T] = null
    if (activation == "tanh") {
      layer = Tanh()
    }
    else if (activation == "sigmoid") {
      layer = Sigmoid()
    }
    else if (activation == "relu") {
      layer = ReLU()
    }
      // TODO: softmax 3D input is different from Keras
    else if (activation == "softmax") {
      layer = SoftMax()
    }
    else if (activation == "softplus") {
      layer = SoftPlus()
    }
    else if (activation == "softsign") {
      layer = SoftSign()
    }
    else if (activation == "hard_sigmoid") {
      layer = HardSigmoid()
    }
    else {
      throw new IllegalArgumentException("Only simple activations can be constructed using string")
    }
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Activation {
  def apply[@specialized(Float, Double) T: ClassTag](
    activation: String,
    inputShape: Array[Int] = null)(implicit ev: TensorNumeric[T]) : Activation[T] = {
      new Activation[T](activation, inputShape)
  }
}