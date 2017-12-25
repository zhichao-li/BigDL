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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, IModule}
import com.intel.analytics.bigdl.nn.keras.NewModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Node, T, Table}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * [[Container]] is an abstract [[IModule]] class which
 * declares methods defined in all containers. A container usually
 * contain some other modules in the `modules` variable. It overrides
 * many module methods such that calls are propogated to the contained
 * modules.
 *
 * @tparam A Input data type
 * @tparam B Output data type
 * @tparam T Numeric type. Only support float/double now
 */
@SerialVersionUID(- 2120105647780417237L)
abstract class Container[A <: Activity : ClassTag,
    B <: Activity : ClassTag, T: ClassTag](
  implicit ev: TensorNumeric[T]) extends AbstractModule[A, B, T] {

//  private var train: Boolean = true

//  protected var forwardTime = 0L
//
//  protected var backwardTime = 0L
//
//  /**
//   * The cached output. So we don't compute it again when need it
//   */
//  var output: B = Activity.allocate[B, T]()
//
//  /**
//   * The cached gradient of activities. So we don't compute it again when need it
//   */
//  var gradInput: A = Activity.allocate[A, T]()

  // list of sub modules
  val modules: ArrayBuffer[IModule[Activity, Activity, T]]
  = ArrayBuffer[IModule[Activity, Activity, T]]()

  def executionNodes(): List[Node[IModule[Activity, Activity, T]]] = {
    val nodes = modules.map(Node(_))
    var i = 0
    var j = 1
    while (i < nodes.length && j < nodes.length) {
      nodes(i).add(nodes(i + 1))
      i += 1
      j += 1
    }
    return nodes.toList
  }

  def compile(): Unit = {
    def gatherFinalResult(values: List[Activity]): Activity = {
      if (values.isEmpty) {
        return null
      }
      if (values.length == 1) {
        return values(0)
      } else {
        val t = new Table()
        values.foreach {v =>
          t.insert(v)
        }
        return t
      }
    }
    var i = 0
    val executionNodes = this.executionNodes()
    while (i < executionNodes.length) {
      val node = executionNodes(i)
//      if (!node.element.isBuilt()) {
        val preNodes = node.prevNodes
        val inputShapes = if (preNodes.isEmpty) {
          if (node.element.getInputShape() == null) {
            throw new RuntimeException("The first layer should explicitly declare inputShape")
          } else {
            List(node.element.getInputShape())
          }
        } else {
          preNodes.map{_.element.getOutputShape()}.toList
        }
        val n = if (node.element.isInstanceOf[NewModule[A, B, T]]) {
          node.element.asInstanceOf[NewModule[A, B, T]]
        } else {
          node.element
        }
        n.build(gatherFinalResult(inputShapes))
//      }
      i += 1
    }
  }

  /**
   * Add a sub-module to the contained `modules`
   *
   * @param module module to be add
   * @return this container
   */
  def add(module: IModule[_ <: Activity, _ <: Activity, T]): this.type = {
    modules += module.asInstanceOf[IModule[Activity, Activity, T]]
    this
  }

  override def zeroGradParameters(): Unit = {
    modules.foreach(_.zeroGradParameters())
  }

  override def updateParameters(learningRate: T): Unit = {
    modules.foreach(_.updateParameters(learningRate))
  }

  override def reset(): Unit = {
    modules.foreach(_.reset())
  }

  override def training(): this.type = {
    train = true
    modules.foreach(_.training())
    this
  }

  override def evaluate(): this.type = {
    train = false
    modules.foreach(_.evaluate())
    this
  }

  override def checkEngineType(): this.type = {
    modules.foreach(_.checkEngineType())
    this
  }

  override def getTimes():
    Array[(IModule[_ <: Activity, _ <: Activity, T], Long, Long)] = {
    this.modules.flatMap(_.getTimes()).toArray
  }

  override def resetTimes(): Unit = {
    modules.foreach(_.resetTimes())
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    val weights = new ArrayBuffer[Tensor[T]]()
    val gradWeights = new ArrayBuffer[Tensor[T]]()
    modules.foreach(m => {
      val params = m.parameters()
      if (params != null) {
        params._1.foreach(weights += _)
        params._2.foreach(gradWeights += _)
      }
    })
    (weights.toArray, gradWeights.toArray)
  }

  override def getExtraParameter(): Array[Tensor[T]] = {
    val extraParam = new ArrayBuffer[Tensor[T]]()
    modules.foreach(m => {
      val state = m.getExtraParameter()
      if (state != null) {
        extraParam ++= state
      }
    })
    extraParam.toArray
  }

  override def getParametersTable(): Table = {
    val pt = T()
    modules.foreach(m => {
      val params = m.getParametersTable()
      if (params != null) {
        params.keySet.foreach(key => pt(key) = params(key))
      }
    })
    pt
  }


  def findModules(moduleType: String): ArrayBuffer[IModule[_, _, T]] = {
    def getName = (x: IModule[_, _, T]) =>
      x.getClass.getName.split("\\.").last

    val nodes = ArrayBuffer[IModule[_, _, T]]()
    if (getName(this) == moduleType) {
      nodes.append(this)
    }
    modules.foreach {
      case container: Container[_, _, T] =>
        nodes ++= container.findModules(moduleType)
      case m =>
        if (getName(m) == moduleType) nodes.append(m)
    }

    nodes
  }

  override def clearState() : this.type = {
    super.clearState()
    modules.foreach(_.clearState())
    this
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[Container[A, B, T]]

  override def equals(other: Any): Boolean = other match {
    case that: Container[A, B, T] =>
      super.equals(that) &&
        (that canEqual this) &&
        modules == that.modules
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), modules)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def setScaleW(w: Double): this.type = {
    modules.foreach(_.setScaleW(w))
    this
  }

  override def setScaleB(b: Double): this.type = {
    modules.foreach(_.setScaleB(b))
    this
  }

  override def freeze(names: String*): this.type = {
    if (names.isEmpty) {
      modules.foreach(_.freeze())
    } else {
      names.foreach(name => {
        this (name) match {
          case Some(x) => x.freeze()
          case _ => throw new Exception(s"cannot match module named $name")
        }
      })
    }
    this
  }

  override def unFreeze(names: String*): this.type = {
    if (names.isEmpty) {
      modules.foreach(_.unFreeze())
    } else {
      names.foreach(name => {
        this (name) match {
          case Some(x) => x.unFreeze()
          case _ => throw new Exception(s"cannot match module named $name")
        }
      })
    }
    this
  }

  override def apply(name : String): Option[IModule[Activity, Activity, T]] = {
    if (this.getName() == name) {
      Some(this)
    } else {
      val find = this.modules.map(m => {
        val get = m(name)
        if (get.isDefined) {
          get
        } else {
          None
        }
      }).filter(_.isDefined)
      require(find.length <= 1, "find multiple modules with same name")
      if (find.length == 1) {
        find(0)
      } else {
        None
      }
    }
  }
}
