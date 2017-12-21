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

import com.intel.analytics.bigdl.nn.{Graph, Input, Linear}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, Initializable, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag


@SerialVersionUID( 359656776803598944L)
class Dense[T: ClassTag](val outputSize: Int,
                           val withBias: Boolean = true,
                           var wRegularizer: Regularizer[T] = null,
                           var bRegularizer: Regularizer[T] = null,
                           var inputShape: Array[Int] = null,
                           private val initWeight: Tensor[T] = null,
                           private val initBias: Tensor[T] = null,
                           private val initGradWeight: Tensor[T] = null,
                           private val initGradBias: Tensor[T] = null
      )(implicit ev: TensorNumeric[T]) extends NewModule[Tensor[T], Tensor[T], T] {
  if (inputShape != null) {
    setInputShape(Tensor(data = inputShape, shape = Array(inputShape.length)))
  }

  override def doBuild(inputShape: Activity): AbstractModule[Tensor[T], Tensor[T], T] = {
    Linear(
      inputSize = inputShape.toTensor[Int].toArray()(0),
      outputSize,
      withBias,
      wRegularizer,
      bRegularizer,
      initWeight,
      initBias,
      initGradWeight,
      initGradBias
    ).asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }

//  override def toString(): String = {
//    s"${getPrintName}($inputSize -> $outputSize)"
//  }
}

object Dense extends App {

}
