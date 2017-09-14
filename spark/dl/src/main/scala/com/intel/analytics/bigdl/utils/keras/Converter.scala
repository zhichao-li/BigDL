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

package com.intel.analytics.bigdl.utils.keras

import com.google.protobuf.GeneratedMessage
import com.intel.analytics.bigdl.dataset.DataSet._
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.log4j.Logger
import play.api.libs.json.{JsArray, JsNull, JsPath, JsValue}

import scala.collection.mutable
import scala.reflect.ClassTag



abstract class Converter[T: ClassTag](kerasJson: KerasJson)(implicit ev: TensorNumeric[T]) {
  protected val logger = Logger.getLogger(getClass)

  private val kerasToBigDLCreator = new mutable.HashMap[String,
    (Layer) => AbstractModule[Activity, Activity, T]]()

  private val nodeIdToKerasLayer = new mutable.HashMap[String, Layer]()
  private val nodeIdToNodeInstance = new mutable.HashMap[String, ModuleNode[T]]()

  init()

  def init(): Unit = {
    kerasToBigDLCreator("InputLayer") = createInput
    kerasToBigDLCreator("Dense") = createDense
    kerasToBigDLCreator("Dropout") = createDropout
    kerasToBigDLCreator("Activation") = createActivation

    // Create name to keras layer mapping
    kerasJson.config.layers.foreach {layer =>
      if (nodeIdToKerasLayer.contains(layer.name)) {
        throw new RuntimeException(s"Duplicate node id: ${layer.name}")
      }
      nodeIdToKerasLayer(layer.name) = layer
    }
  }

  private def convertInOrOutForModel(boundNodes: Seq[JsArray]): Array[ModuleNode[T]] = {
      boundNodes.map { node =>
        val nodeName = node.value(0).toString().replaceAll("^\"|\"$", "")
        // TODO: parse nodeID and tensorID
        this.nodeIdToNodeInstance(nodeName)
      }.toArray
  }

  def createGraph(kerasJson: KerasJson): Graph[T] = {
    // ensure each node instances is created
    kerasJson.config.layers.foreach { layer =>
      if (!this.nodeIdToNodeInstance.contains(layer.name)) {
        doCreateNode(layer)
      }
    }

    val input = convertInOrOutForModel(kerasJson.config.inputLayers)
    val output = convertInOrOutForModel(kerasJson.config.outputLayers)
    Graph[T](input = input, output = output)
  }

  def doCreateNode(layer: Layer): ModuleNode[T] = {
     if (layer.className == "InputLayer") {
       val input = Input[T]() // input cannot set name
       this.nodeIdToNodeInstance(layer.name) = input
       return input
     }
    val inNodes = layer.inboundNodes.map { node =>
      val nodeName = node(0)(0).get.toString().replaceAll("^\"|\"$", "") // TODO why always o here?
      // todo: parse nodeindex or tensorindex
      if (!this.nodeIdToNodeInstance.contains(nodeName)) {
        val node = doCreateNode(this.nodeIdToKerasLayer(nodeName))
        logger.info(s"Creating: $nodeName")
      }
      this.nodeIdToNodeInstance(nodeName)
    }
    val bigDLLayer = this.kerasToBigDLCreator(layer.className)(layer)
    val newNode = bigDLLayer.inputs(inNodes : _*)
    this.nodeIdToNodeInstance(layer.name) = newNode
    newNode
  }

  def createInput(layer: Layer): AbstractModule[Activity, Activity, T]

  def createEmbedding(layer: Layer): AbstractModule[Activity, Activity, T]

  def createFlatten(layer: Layer): AbstractModule[Activity, Activity, T]

  def createMerge(layer: Layer): AbstractModule[Activity, Activity, T]

  def createDense(layer: Layer): AbstractModule[Activity, Activity, T]

  def createDropout(layer: Layer): AbstractModule[Activity, Activity, T]

  def createActivation(layer: Layer): AbstractModule[Activity, Activity, T]

}

object InitMethodHelper {
  def toBigDL[T](initName: String): InitializationMethod = {
    initName match {
      case "glorot_uniform" => RandomUniform  // case object cannot use isinstance of
      case "one" => Ones
      case i: String => throw new RuntimeException(s"not supported yet $i")
    }
  }
}

object RegularizerHelper {
  def toBigDL[T](reg: JsValue): Regularizer[T] = {
    reg match {
      case JsNull => null  // case object cannot use isinstance of
      case _ => throw new RuntimeException("not supported yet")
    }
  }
}

object ActivationHelper {
  def toBigDL[T: ClassTag](activationName: String,
                 layerName: String)
                          (implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {
    activationName match {
      case "relu" => ReLU[T]().setName(layerName)
      case "softmax" => LogSoftMax[T]().setName(layerName)
      case _ => throw new IllegalArgumentException(
        s"unsupported type: ${activationName}")
    }
  }

  def fuse[T: ClassTag](srcLayer: AbstractModule[Activity, Activity, T],
              activationName: String,
              name: String)
                       (implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {
    // "linear" meaning do nothing
    if (activationName != "linear") {
      val seq = Sequential[T]()
      seq.add(srcLayer)
      seq.add(toBigDL(activationName, name))
      seq.setName(srcLayer.getName())
    } else {
      srcLayer
    }
  }
}

class Keras1Converter[T: ClassTag](kerasJson: KerasJson)(implicit ev: TensorNumeric[T])
  extends Converter[T](kerasJson) {

  override def createInput(layer: Layer): AbstractModule[Activity, Activity, T] = {
    // place holder , dummpy
    return null
  }

  override def createEmbedding(layer: Layer): AbstractModule[Activity, Activity, T] = {
    return null
  }

  override def createFlatten(layer: Layer): AbstractModule[Activity, Activity, T] = {
    return null
  }

  override def createMerge(layer: Layer): AbstractModule[Activity, Activity, T] = {
    return null
  }

  override def createDense(layer: Layer): AbstractModule[Activity, Activity, T] = {
    val layerConfig = new DenseConfig(layer.config)
    if (layerConfig.wConstraint != JsNull || layerConfig.bConstraint != JsNull ) {
      throw new IllegalArgumentException("Haven't support constraint yet")
    }

    val linear = Linear[T](
      inputSize = layerConfig.inputDim,
      outputSize = layerConfig.outputDim,
      withBias = layerConfig.bias,
      wRegularizer = RegularizerHelper.toBigDL(layerConfig.wRegularizer),
      bRegularizer = RegularizerHelper.toBigDL(layerConfig.wRegularizer)
    ).setName(layer.name)
    val initMethod = InitMethodHelper.toBigDL(layerConfig.initMethod)
    linear.setInitMethod(initMethod, Zeros) // Keras always set this to be Zero.

    ActivationHelper.fuse[T](linear,
      layerConfig.activation,
      s"${layer.name}_${layerConfig.activation}")
  }

  def createDropout(layer: Layer): AbstractModule[Activity, Activity, T] = {
    val dropoutConfig = new DropoutConfig(layer.config)
    Dropout[T](dropoutConfig.p).setName(layer.name)
  }

  def createActivation(layer: Layer): AbstractModule[Activity, Activity, T] = {
    val inboundNodes = layer.inboundNodes
    val instanceName = layer.name
    val layerConfig = new ActivationConfig(layer.config)
    ActivationHelper.toBigDL(layerConfig.activation, layer.name)
  }

}
