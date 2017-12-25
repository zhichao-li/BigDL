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
T: ClassTag](implicit ev: TensorNumeric[T]) extends IModule[A, B, T] {

  override def build(inputShape: Activity): Unit = {
    labor = doBuild(inputShape)
    labor.build(inputShape)
//    this.output = labor.output
//    this.gradInput = labor.gradInput
//    this.forwardTime = labor.forwardTime
//    this.backwardTime = labor.backwardTime
//    this.line = labor.line
//    this.scaleB = labor.scaleB
//    this.scaleBCache = labor.scaleBCache  //TODO: add this back and override all other methods
//    this.scaleW = labor.scaleW
//    this.scaleWCache = labor.scaleWCache
//    this.train = labor.train
  }

  def doBuild(inputShape: Activity): IModule[A, B, T]

  // This method would only be called after `doBuilt`
  override def doComputeOutputShape(inputShape: Activity): Activity = {
    this.labor.doComputeOutputShape(inputShape)
  }

  override def accGradParameters(input: A, gradOutput: B): Unit = {
    labor.accGradParameters(input, gradOutput)
  }

  /**
   * Computes the output using the current parameter set of the class and input. This function
   * returns the result which is stored in the output field.
   *
   * @param input
   * @return
   */
  override def updateOutput(input: A): B = {
    if (!this.isBuilt) {
            throw new RuntimeException("The model haven't been built")
    }
    this.labor.updateOutput(input)
  }

  /**
   * Computing the gradient of the module with respect to its own input. This is returned in
   * gradInput. Also, the gradInput state variable is updated accordingly.
   *
   * @param input
   * @param gradOutput
   * @return
   */
  override def updateGradInput(input: A, gradOutput: B): A
  = {
    if (!this.isBuilt) {
      throw new RuntimeException("The model haven't been built")
    }
    this.labor.updateGradInput(input, gradOutput)
  }
}
