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
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.{Graph, Input, Linear}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import play.api.libs.json.{JsNull, JsPath, JsValue}

import scala.collection.mutable

class BaseLayerConfig(val name: String,
                      val trainable: Boolean,
                      val batchInputShape: Option[JsValue],
                      val inputDtype: Option[String]) {
  def this(config: JsValue) = {
    this(
      (JsPath \ "name").read[String].reads(config).get,
      (JsPath \ "trainable").read[Boolean].reads(config).get,
      (JsPath \ "batch_input_shape").readNullable[JsValue].reads(config).get,
      (JsPath \ "input_dtype").readNullable[String].reads(config).get
    )
  }
}
//object BaseLayerConfig {
//  def apply(config: JsValue) = {
//
//  }
//  def parse(config: JsValue): BaseLayerConfig = {
//    new BaseLayerConfig(
//      (JsPath \ "name").read[String].reads(config).get,
//      (JsPath \ "trainable").read[Boolean].reads(config).get,
//      (JsPath \ "batch_input_shape").readNullable[JsValue].reads(config).get,
//      (JsPath \ "input_dtype").readNullable[String].reads(config).get
//    )
//  }
//}

class InputConfig(config: JsValue) extends BaseLayerConfig(config) {
  val sparse: Boolean = (JsPath \ "sparse").read[Boolean].reads(config).get
}

class FlattenConfig(config: JsValue) extends BaseLayerConfig(config)

class DenseConfig(config: JsValue) extends BaseLayerConfig(config) {
  val outputDim = (JsPath \ "output_dim").read[Int].reads(config).get
  val initMethod = (JsPath \ "init").read[String].reads(config).get
  val activation = (JsPath \ "activation").read[String].reads(config).get
  val wRegularizer = (JsPath \ "W_regularizer").read[JsValue].reads(config).get
  val wConstraint = (JsPath \ "W_constraint").read[JsValue].reads(config).get
  val bConstraint = (JsPath \ "b_constraint").read[JsValue].reads(config).get
  val bias = (JsPath \ "bias").read[Boolean].reads(config).get
  val inputDim = (JsPath \ "input_dim").read[Int].reads(config).get
}

object RegularizerHelper {
  def toBigDL[T](reg: JsValue): Regularizer[T] = {
    reg match {
      case JsNull => null  // case object cannot use isinstance of
      case _ => throw new RuntimeException("not supported yet")
    }
  }
}

class ActivationConfig(config: JsValue) extends BaseLayerConfig(config) {
  val activation = (JsPath \ "activation").read[String].reads(config).get
}

class DropoutConfig(config: JsValue) extends BaseLayerConfig(config) {
  val activation = (JsPath \ "p").read[String].reads(config).get
}


abstract class Converter[T](kerasJson: KerasJson) {
  private val kerasToBigDLCreator = new mutable.HashMap[String,
    (Layer) => AbstractModule[Activity, Activity, T]]()

  private val nodeIdToKerasLayer = new mutable.HashMap[String, Layer]()
  private val nodeIdToNodeInstance = new mutable.HashMap[String, ModuleNode[T]]()

  init()

  def init(): Unit = {
    kerasToBigDLCreator("InputLayer") = createInput
    kerasJson.config.layers.foreach {layer =>
      if (nodeIdToKerasLayer.contains(layer.name.get)) {
        throw new RuntimeException(s"Duplicate node id: ${layer.name.get}")
      }
      nodeIdToKerasLayer(layer.name.get) = layer
    }


  }

  def createNode(kerasJson: KerasJson): ModuleNode[T] = {

    kerasJson.config.layers.foreach { layer =>
      val bigdlLayer = kerasToBigDLCreator(layer.name.get)(layer)
      val inNodes = layer.inboundNodes.map { inNodeConfig =>
        val inNodeName = inNodeConfig(0).get.toString()
        // TODO: may need to get the nodeID and tensorID from inboundNodes
        if(!nodeIdToNodeInstance.contains(inNodeName)) {
          val jsLayer = nodeIdToKerasLayer(inNodeName)
          val inNode = doCreateNode(jsLayer)
          nodeIdToNodeInstance(inNodeName) = inNode
        }
        this.nodeIdToNodeInstance(inNodeName)
      }
      val bigDLNode = bigdlLayer.inputs(inNodes : _*)
      bigDLNode
    }

    val input = kerasJson.config.inputLayers { inputLayer =>
      val inputName = inputLayer(0).toString()
      // TODO: parse nodeID and tensorID
      this.nodeIdToNodeInstance(inputName)
    }

    val output = kerasJson.config.outputLayers { outputLayer =>
      val inputName = outputLayer(0).toString()
      // TODO: parse nodeID and tensorID
      this.nodeIdToNodeInstance(inputName)
    }

    Graph(input=input, output=output)
  }

  def doCreateNode(layer: Layer): ModuleNode[T] = {

  }

//  def apply(layer: Layer): ModuleNode[T] = {
//    // TODO: add guard here in case exception
//    kerasToBigDL(layer.className)(layer)
//  }

//  def parseLayerConfig(config: JsValue): Map[String, String] = {
//    val baseConfig = BaseLayerConfig.parse(config)
//    baseConfig.name match {
//      case "Flatten" =>
//        new FlattenConfig(config)
//    }
//  }

  def createInput(layer: Layer): AbstractModule[Activity, Activity, T]

  def createEmbedding(layer: Layer): AbstractModule[Activity, Activity, T]

  def createFlatten(layer: Layer): AbstractModule[Activity, Activity, T]

  def createMerge(layer: Layer): AbstractModule[Activity, Activity, T]

  def createDense(layer: Layer): AbstractModule[Activity, Activity, T]

  def createDropout(layer: Layer): AbstractModule[Activity, Activity, T]

  def createActivation(layer: Layer): AbstractModule[Activity, Activity, T]

}

class Keras1Converter[T] extends Converter[T] {

  override def createInput(layer: Layer): AbstractModule[Activity, Activity, T] = {
            new FlattenConfig(layer.config)

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
    val inboundNodes = layer.inboundNodes
    val instanceName = layer.name
    val layerConfig = new DenseConfig(layer.config)
    if (layerConfig.wConstraint != JsNull || layerConfig.bConstraint != JsNull ) {
      throw new IllegalArgumentException("Haven't support constraint yet")
    }
    Linear[T](
      inputSize = layerConfig.inputDim,
      outputSize = layerConfig.outputDim,
      withBias = layerConfig.bias,
      wRegularizer = RegularizerHelper.toBigDL(layerConfig.wRegularizer),
      bRegularizer = RegularizerHelper.toBigDL(layerConfig.wRegularizer)
    ).asInstanceOf
  }

  def createDropout(layer: Layer): AbstractModule[Activity, Activity, T] = {
null
  }

  def createActivation(layer: Layer): AbstractModule[Activity, Activity, T] = {
null
  }

}
