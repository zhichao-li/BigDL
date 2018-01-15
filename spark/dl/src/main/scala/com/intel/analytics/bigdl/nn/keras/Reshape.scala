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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class Reshape[T: ClassTag](val targetShape: Array[Int],
                           var inputShape: Array[Int] = null)
(implicit ev: TensorNumeric[T]) extends KerasModule[Tensor[T], Tensor[T], T](inputShape) {
  override def doBuild(inputShape: Activity): AbstractModule[Tensor[T], Tensor[T], T] = {
    val layer = com.intel.analytics.bigdl.nn.Reshape(targetShape)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Reshape {
  def apply[@specialized(Float, Double) T: ClassTag]
  (targetShape: Array[Int], inputShape: Array[Int] = null)
  (implicit ev: TensorNumeric[T]): Reshape[T] = {
    new Reshape[T](targetShape, inputShape)
  }
}
