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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat, TensorModule}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

@SerialVersionUID( 4656141099678250116L)
class Convolution2D[T: ClassTag](val nbFilter: Int,
                                 val nbRow: Int,
                                 val nbCol: Int,
                                 val init: InitializationMethod = Xavier,
                                 val activation: TensorModule[T] = null,
                                 val borderMode: String = "valid",
                                 val subsample: (Int, Int) = (1, 1),
                                 var wRegularizer: Regularizer[T] = null,
                                 var bRegularizer: Regularizer[T] = null,
                                 val format: DataFormat = DataFormat.NCHW,
                                 val bias: Boolean = true,
                                 var inputShape: Array[Int] = null) (implicit ev: TensorNumeric[T])
  extends KerasModule[Tensor[T], Tensor[T], T](inputShape) {

  override def doBuild(inputShape: Activity): AbstractModule[Tensor[T], Tensor[T], T] = {
    val pads = KerasUtils.getPadsFromBorderMode(borderMode)
    val layer = SpatialConvolution(
      nInputPlane = inputShape.toTensor[Int].toArray()(format.getHWCDims(3)._3 - 1),
      nOutputPlane = nbFilter,
      kernelW = nbCol,
      kernelH = nbRow,
      strideW = subsample._2,
      strideH = subsample._1,
      padW = pads._2,
      padH = pads._1,
      wRegularizer = wRegularizer,
      bRegularizer = bRegularizer,
      withBias = bias,
      format = format
    )
    layer.setInitMethod(weightInitMethod = init, biasInitMethod = Zeros)
    KerasModule.fuse(layer,
      activation,
      inputShape).asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}


object Convolution2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    nbFilter: Int,
    nbRow: Int,
    nbCol: Int,
    init: InitializationMethod = Xavier,
    activation: TensorModule[T] = null,
    borderMode: String = "valid",
    subsample: (Int, Int) = (1, 1),
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    format: DataFormat = DataFormat.NCHW,
    bias: Boolean = true,
    inputShape: Array[Int] = null)(implicit ev: TensorNumeric[T]) : Convolution2D[T] = {
    new Convolution2D[T](nbFilter, nbRow, nbCol, init, activation, borderMode,
      subsample, wRegularizer, bRegularizer, format, bias, inputShape)
  }
}
