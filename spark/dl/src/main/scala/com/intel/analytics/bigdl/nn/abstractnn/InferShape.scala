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

package com.intel.analytics.bigdl.nn.abstractnn

import com.intel.analytics.bigdl.nn.keras.{Input => KInput}
import com.intel.analytics.bigdl.nn.{Input => TInput}
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

class InvalidLayer(msg: String) extends RuntimeException(msg)

trait InferShape {

  private[bigdl] var isUsed = false

  private[bigdl] var _inputShapeValue: Shape = null

  private[bigdl] var _outputShapeValue: Shape = null

  private[bigdl] def inputShapeValue: Shape = _inputShapeValue

  private[bigdl] def outputShapeValue: Shape = _outputShapeValue

  // scalastyle:off
  private[bigdl] def inputShapeValue_=(value: Shape): Unit = {
    _inputShapeValue = value
  }

  private[bigdl] def outputShapeValue_=(value: Shape): Unit = {
    _outputShapeValue = value
  }
  // scalastyle:on

  /**
   * We suppose the first dim is batch
   */
  private[bigdl] final def getInputShape(): Shape = {
    _inputShapeValue
  }

  /**
   * We suppose the first dim is batch
   */
  private[bigdl] final def getOutputShape(): Shape = {
    outputShapeValue
  }

  private[bigdl] def inferShape(calcInputShape: Shape): Shape = {
    val outputShape = computeOutputShape(calcInputShape)
    this.outputShapeValue = outputShape
    this.inputShapeValue = calcInputShape
    outputShape
  }

  /**
   * Execute building logic and return the outputShape for the given inputShape.
   * NB: the first dim of inputShape is batch
   */
  private[bigdl] def build(inputShape: Shape): Shape = {
    inferShape(inputShape)
  }

  private[bigdl] def isBuilt(): Boolean = outputShapeValue != null

  private[bigdl] def isKerasStyle(): Boolean = false

  /**
   * We suppose the first dim is batch
   */
  private[bigdl] def computeOutputShape(inputShape: Shape): Shape = {
    throw new RuntimeException("Haven't been implemented yet. Do not use it with Keras Layer")
  }

  private def ensureNotShared(): Unit = {
    if (isUsed == true && !this.isInstanceOf[TInput[_]]
    && !this.isInstanceOf[KInput[_]]) {
      throw new RuntimeException(s"Reuse module is not allowed: $this")
    }
    isUsed = true
  }

  private def ensureNotShared[T: ClassTag](modules : Seq[AbstractModule[_, _, T]]): Unit = {
    modules.map{_.ensureNotShared()}
  }

  private def excludeInvalidLayers[T: ClassTag]
  (modules : Seq[AbstractModule[_, _, T]]): Unit = {
    val invalidNodes = if (this.isKerasStyle()) {
      modules.filter{!_.isKerasStyle()}
    } else {
      modules.filter{_.isKerasStyle()}
    }
    if (invalidNodes.length > 0) {
      throw new InvalidLayer(s"Do not mix with Layer: ${invalidNodes.mkString(",")}")
    }
  }

  private[bigdl] def validateInput[T: ClassTag](modules : Seq[AbstractModule[_, _, T]]): Unit = {
    if (this.isKerasStyle()) {
      require(modules != null && !modules.isEmpty, "Empty input is not allow")
      ensureNotShared(modules)
    }
    excludeInvalidLayers(modules)
  }
}

