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

import com.intel.analytics.bigdl.nn.abstractnn.{IModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag


abstract class NewModule[A <: Activity: ClassTag, B <: Activity: ClassTag,
T: ClassTag](implicit ev: TensorNumeric[T]) extends IModuleAdapter[A, B, T] {

  private var inputShapeValue: Activity = null

  override def setInputShape(inputShape: Activity): Unit = {
    this.inputShapeValue = inputShape
  }

  override def getInputShape(): Activity = inputShapeValue

  override def build(inputShape: Activity): Unit = {
    labor = doBuild(inputShape)
    labor.build(inputShape)
  }

  def doBuild(inputShape: Activity): IModule[A, B, T]

  // This method would only be called after `doBuilt`
  override def doComputeOutputShape(inputShape: Activity): Activity = {
    this.labor.doComputeOutputShape(inputShape)
  }
}
