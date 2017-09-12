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

import play.api.libs.json._
import play.api.libs.functional.syntax._

import scala.reflect.ClassTag


class JsonParser[K: ClassTag] {
  implicit val layerReads: Reads[Layer] = (
    (JsPath \ "class_name").read[String] and
      (JsPath \ "config").read[JsValue] and
      (JsPath \ "inbound_nodes").read[Seq[JsValue]] and
      (JsPath \ "name").readNullable[String]
    )(Layer.apply _)

  implicit val modelConfigReads: Reads[ModelConfig] = (
    (JsPath \ "name").read[String] and
      (JsPath \ "layers").read[Seq[Layer]] and
      (JsPath \ "input_layers").read[Seq[JsValue]] and
      (JsPath \ "output_layers").read[Seq[JsValue]]
    )(ModelConfig.apply _)

  implicit val KerasJsonReads: Reads[KerasJson] = (
    (JsPath \ "class_name").read[String] and
      (JsPath \ "config").read[ModelConfig] and
      (JsPath \ "keras_version").read[String]
    )(KerasJson.apply _)

  def parseLayer(jsonString: String): Layer = {
    val jsonObject = Json.parse(jsonString)
    val placeResult = jsonObject.validate[Layer]
    placeResult match {
      case JsSuccess(value, path) => value
    }
  }

  def parseKerasJson(jsonString: String): KerasJson = {
    val jsonObject = Json.parse(jsonString)
    val placeResult = jsonObject.validate[KerasJson]
    placeResult match {
      case JsSuccess(value, path) => value
    }
  }

//  def parse(jsonString: String): K = {
//    //  val trainable = kerasJson.config.layers(3).config \ "trainable"
//    val jsonObject = Json.parse(jsonString)
//     val placeResult = jsonObject.validate[K]
//        val result = placeResult match {
//          case JsSuccess(value, path) => value
//        }
//        result
//  }
}

//object JsonParser {
//  def apply[T](): JsonParser[T] = {
//    JsonParser[T]()
//  }
//}