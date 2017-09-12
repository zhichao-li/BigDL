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

import java.nio.file.{Files, Paths}

import com.amazonaws.services.cloudfront.model.InvalidArgumentException
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.Graph
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.log4j.Logger
import play.api.libs.functional.syntax._
import play.api.libs.json._

import scala.reflect.ClassTag

//case class KerasHistory(layerName: String, nodeIndex: String, tensorIndex: String)


case class Layer(className: String,
                 config: JsValue,
                 inboundNodes: Seq[JsValue],
                 name: Option[String]) // name is layer name
case class ModelConfig(name: String,
                       layers: Seq[Layer],
                       inputLayers: Seq[JsValue],
                       outputLayers: Seq[JsValue]) // name is model name
case class KerasJson(className: String, config: ModelConfig, kerasVersion: String)


class KerasLoader[T: ClassTag]()(implicit ev: TensorNumeric[T]) {
  private val logger = Logger.getLogger(getClass)
  private val nameToBigDLNode = Map[String, ModuleNode[T]]()
  private val nameToBigDLLayer = Map[String, ModuleNode[T]]()
  private val converter = new Keras1Converter[T]()

  def tensorToBigDLNode(tensorName: String): ModuleNode[T] = {
    if (!nameToBigDLNode.contains(tensorName)) {
      throw new InvalidArgumentException("Not support type: " + tensorName)
    }
    this.nameToBigDLNode(tensorName)
  }

  private def createBigDLNodes(kerasLayers: Seq[Layer]): Unit = {
    kerasLayers.map { kerasLayer =>
      val kerasLayerClassName = kerasLayer.className
      converter(kerasLayer)


    }
  }

  def loadModule(kerasJsonPath: String): AbstractModule[Activity, Activity, T] = {
    loadModule(loadKerasJsonFromPath(kerasJsonPath))
  }

  def loadModule(kerasJson: KerasJson): AbstractModule[Activity, Activity, T] = {
    // Map keras layer to bigdl instances
    createBigDLNodes(kerasJson.config.layers)

    val outputs = kerasJson.config.outputLayers.map {tensorJsValue =>
      val tensorName = tensorJsValue(0).get.toString()
      val nodeIndex = tensorJsValue(1)
      val tensorIndex = tensorJsValue(2)
      this.tensorToBigDLNode(tensorName)
    }.toArray

    val inputs = kerasJson.config.inputLayers.map {tensorJsValue =>
      val tensorName = tensorJsValue(0).get.toString()
      val nodeIndex = tensorJsValue(1)
      val tensorIndex = tensorJsValue(2)
      this.tensorToBigDLNode(tensorName)
    }.toArray
    val graph = Graph(inputs, outputs)

    graph

  }
  def loadKerasJsonFromPath(path: String): KerasJson = {
    val jsonStr = readFileToString(path)
    val kerasJson = new JsonParser[KerasJson]().parseKerasJson(jsonStr)
    kerasJson
  }


  private def readFileToString(path: String): String = {
    new String(Files.readAllBytes(Paths.get(path)))
  }

}

object KerasLoader {
  def main(args: Array[String]) {
    val config = new KerasLoader[Float]().loadKerasJsonFromPath(
      "/home/lizhichao/bin/god/recommder/ncf_model.json")
//    new FlattenConfig(config)
    println("Hello, world")
  }

}
