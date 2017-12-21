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

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Node

import scala.reflect.ClassTag

/**
 * Input layer do nothing to the input tensors, just pass them. It should be used as input node
 * when the first layer of your module accepts multiple tensors as inputs.
 *
 * Each input node of the graph container should accept one tensor as input. If you want a module
 * accepting multiple tensors as input, you should add some Input module before it and connect
 * the outputs of the Input nodes to it.
 *
 * Please note that the return is not a layer but a Node containing input layer.
 *
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */
@SerialVersionUID(- 8525406230282608924L)
class NewInput[T: ClassTag](inputShape: Array[Int])(implicit ev: TensorNumeric[T])
  extends AbstractModule[Activity, Activity, T] {

  setInputShape(Tensor(data = inputShape, shape = Array(inputShape.length)))

  override def updateOutput(input: Activity): Activity = {
    output = input
    output
  }
  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput = gradOutput
    gradInput
  }
  override def equals(other: Any): Boolean = {
    if (!other.isInstanceOf[NewInput[_]]) return false
    this.eq(other.asInstanceOf[NewInput[_]])
  }

  override def getOutputShape(): Activity = {
    Tensor(data = inputShape, shape = Array(inputShape.size))
  }


  override def hashCode(): Int = System.identityHashCode(this)
}

object NewInput {
  def apply[T: ClassTag](inputShape: Array[Int],
    name : String = null)(implicit ev: TensorNumeric[T]): ModuleNode[T] = {
    val module = new NewInput(inputShape)
    if (name != null) {
      module.setName(name)
    }
    new Node(module.asInstanceOf[AbstractModule[Activity, Activity, T]])
  }
}
