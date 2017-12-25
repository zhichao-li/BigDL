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

import java.nio.ByteOrder

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.tensor.{Tensor, TensorDataType}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.nn.{Module, _}
import com.intel.analytics.bigdl.utils.TorchObject.TYPE_MODULE
import org.apache.commons.lang3.SerializationUtils
import org.apache.spark.rdd.RDD
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch, Sample}
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.quantized.Quantization
import com.intel.analytics.bigdl.transform.vision.image.{DistributedImageFrame, ImageFeature, ImageFrame, LocalImageFrame}
import com.intel.analytics.bigdl.utils.caffe.CaffePersister
import com.intel.analytics.bigdl.utils.serializer.ModulePersister
import com.intel.analytics.bigdl.utils.tf.{TensorflowDataFormat, TensorflowSaver}

import scala.reflect.ClassTag

/**
 * Module is the basic component of a neural network. It forward activities and backward gradients.
 * Modules can connect to others to construct a complex neural network.
 *
 * @tparam A Input data type
 * @tparam B Output data type
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now
 */
abstract class IModule[A <: Activity: ClassTag, B <: Activity: ClassTag, T: ClassTag]
  extends Serializable {

  def getOutput: B

  def getGradInput: A

  def setGradInput(gradInput: A): Unit

  def getNamePostfix : String = Integer.toHexString(java.util.UUID.randomUUID().hashCode())

  def setNamePostfix(namePostfix : String) : Unit

  def setInputShape(inputShape: Activity): Unit

  def getInputShape(): Activity

  def getOutputShape(): Activity

  def isBuilt(): Boolean

  def build(inputShape: Activity): Unit

  final def computeOutputShape(inputShape: Activity): Activity = {
    if (! isBuilt) {
      throw new RuntimeException("The model haven't been built")
    }
    doComputeOutputShape(inputShape)
  }

  def doComputeOutputShape(inputShape: Activity): Activity = inputShape

  /**
   * Get the scale of gradientWeight
   */
  def getScaleW(): Double

  /**
   * Get the scale of gradientBias
   */
  def getScaleB(): Double

  /**
   * Set the scale of gradientWeight
   *
   * @param w the value of the scale of gradientWeight
   * @return this
   */
  def setScaleW(w: Double): IModule[A, B, T]

  /**
   * Set the scale of gradientBias
   *
   * @param b the value of the scale of gradientBias
   * @return this
   */
  def setScaleB(b: Double): IModule[A, B, T]

  /**
   * Clear cached activities to save storage space or network bandwidth. Note that we use
   * Tensor.set to keep some information like tensor share
   *
   * The subclass should override this method if it allocate some extra resource, and call the
   * super.clearState in the override method
   *
   * @return
   */
  def clearState() : IModule[A, B, T]


  def hasName: Boolean
  /**
   * Set the module name
   *
   * @param name
   * @return
   */
  def setName(name : String) : IModule[A, B, T]

  /**
   * Get the module name, default name is className@namePostfix
   *
   * @return
   */
  def getName() : String

  def getPrintName(): String

  override def toString(): String = getPrintName


  def getTimes(): Array[(IModule[_ <: Activity, _ <: Activity, T], Long, Long)]

  def resetTimes(): Unit

  /**
   * freeze the module,
   * i.e. their parameters(weight/bias, if exists) are not changed in training process
   * if names is not empty,
   * set an array of layers that match the given ```names``` to be "freezed",
   *
   * @param names an array of layer names
   * @return current graph model
   */
  def freeze(names: String*): IModule[A, B, T]

  /**
   * "unfreeze" module, i.e. make the module parameters(weight/bias, if exists)
   * to be trained(updated) in training process
   * if names is not empty, unfreeze layers that match given names
   *
   * @param names array of module names to unFreeze
   */
  def unFreeze(names: String*): IModule[A, B, T]
  /**
   * Takes an input object, and computes the corresponding output of the module. After a forward,
   * the output state variable should have been updated to the new value.
   *
   * @param input input data
   * @return output data
   */
  def forward(input: A): B

  /**
   * Performs a back-propagation step through the module, with respect to the given input. In
   * general this method makes the assumption forward(input) has been called before, with the same
   * input. This is necessary for optimization reasons. If you do not respect this rule, backward()
   * will compute incorrect gradients.
   *
   * @param input input data
   * @param gradOutput gradient of next layer
   * @return gradient corresponding to input data
   */
  def backward(input: A, gradOutput: B): A

  /**
   * Computes the output using the current parameter set of the class and input. This function
   * returns the result which is stored in the output field.
   *
   * @param input
   * @return
   */
  def updateOutput(input: A): B

  /**
   * Computing the gradient of the module with respect to its own input. This is returned in
   * gradInput. Also, the gradInput state variable is updated accordingly.
   *
   * @param input
   * @param gradOutput
   * @return
   */
  def updateGradInput(input: A, gradOutput: B): A

  /**
   * Computing the gradient of the module with respect to its own parameters. Many modules do not
   * perform this step as they do not have any parameters. The state variable name for the
   * parameters is module dependent. The module is expected to accumulate the gradients with
   * respect to the parameters in some variable.
   *
   * @param input
   * @param gradOutput
   */
  def accGradParameters(input: A, gradOutput: B): Unit

  /**
   * If the module has parameters, this will zero the accumulation of the gradients with respect
   * to these parameters. Otherwise, it does nothing.
   */
  def zeroGradParameters(): Unit

  def updateParameters(learningRate: T): Unit

  /**
   * This method compact all parameters and gradients of the model into two tensors. So it's easier
   * to use optim method
   *
   * @return
   */
  def getParameters(): (Tensor[T], Tensor[T])

  /**
   * This function returns two arrays. One for the weights and the other the gradients
   * Custom modules should override this function if they have parameters
   *
   * @return (Array of weights, Array of grad)
   */
  def parameters(): (Array[Tensor[T]], Array[Tensor[T]])

  /**
   * Get extra parameter in this module.
   * Extra parameter means the trainable parameters beside weight and bias. Such as runningMean
   * and runningVar in BatchNormalization.
   *
   * The subclass should override this method if it has some parameters besides weight and bias.
   *
   * @return an array of tensor
   */
  def getExtraParameter(): Array[Tensor[T]]

  /**
   * Set extra parameter to this module.
   * Extra parameter means the trainable parameters beside weight and bias. Such as runningMean
   * and runningVar in BatchNormalization.
   *
   * @return this
   */
  def setExtraParameter(extraParam: Array[Tensor[T]]): IModule[A, B, T]

  /**
   * This function returns a table contains ModuleName, the parameter names and parameter value
   * in this module.
   * The result table is a structure of Table(ModuleName -> Table(ParameterName -> ParameterValue)),
   * and the type is Table[String, Table[String, Tensor[T]]].
   *
   * For example, get the weight of a module named conv1:
   *   table[Table]("conv1")[Tensor[T]]("weight").
   *
   * Custom modules should override this function if they have parameters.
   *
   * @return Table
   */
  def getParametersTable(): Table

  def training(): IModule[A, B, T]

  def evaluate(): IModule[A, B, T]

  def isTraining(): Boolean

  def reset(): Unit

  def setLine(line: String): IModule[A, B, T]

  /**
   * get execution engine type
   */
  def checkEngineType(): IModule[A, B, T]

  def cloneModule(): IModule[A, B, T]

  def canEqual(other: Any): Boolean = other.isInstanceOf[IModule[A, B, T]]

  /**
   * Save this module to path.
   * @param path path to save module, local file system, HDFS and Amazon S3 is supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx"
   *             Amazon S3 path should be like "s3a://bucket/xxx"
   * @param overWrite if overwrite
   * @return self
   */
  @deprecated("please use recommended saveModule(path, overWrite)")
  def save(path : String, overWrite: Boolean = false) : IModule[A, B, T]

  /**
   * Save this module to path with protobuf format
   * @param path path to save module, local file system, HDFS and Amazon S3 is supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx"
   *             Amazon S3 path should be like "s3a://bucket/xxx"
   * @param weightPath where to store weight
   * @param overWrite if overwrite
   * @return self
   */
  def saveModule(path : String, weightPath : String = null,
                 overWrite: Boolean = false) : IModule[A, B, T]

  /**
   * Save this module definition to path.
   * @param path path to save module, local file system, HDFS and Amazon S3 is supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx"
   *             Amazon S3 path should be like "s3a://bucket/xxx"
   * @param overWrite if overwrite
   * @return self
   */
  def saveDefinition(path : String, overWrite: Boolean = false) : IModule[A, B, T]

  def saveTorch(path : String, overWrite: Boolean = false) : IModule[A, B, T]

  def saveCaffe(prototxtPath: String, modelPath: String,
                useV2 : Boolean = true, overwrite : Boolean = false) : IModule[A, B, T]

  def saveTF(
              inputs : Seq[(String, Seq[Int])],
              path: String,
              byteOrder: ByteOrder = ByteOrder.LITTLE_ENDIAN,
              dataFormat: TensorflowDataFormat = TensorflowDataFormat.NHWC): IModule[A, B, T]

  /**
   * module predict, return the probability distribution
   * @param dataset dataset for prediction
   * @param batchSize total batchSize for all partitions.
   *                  if -1, default is 4 * partitionNumber of datatset
   * @param shareBuffer whether to share same memory for each batch predict results
   */
  def predict(dataset: RDD[Sample[T]],
              batchSize: Int = -1,
              shareBuffer: Boolean = false): RDD[Activity]
  /**
   * module predict, return the predict label
   * @param dataset dataset for prediction
   * @param batchSize total batchSize for all partitions.
   *                  if -1, default is 4 * partitionNumber of dataset
   */
  def predictClass(dataset: RDD[Sample[T]], batchSize: Int = -1): RDD[Int]

  /**
   * model predict images, return imageFrame with predicted tensor
   * @param imageFrame imageFrame that contains images
   * @param outputLayer if outputLayer is not null, the output of layer that matches
   *                      outputLayer will be used as predicted output
   * @param shareBuffer whether to share same memory for each batch predict results
   * @param batchPerPartition batch size per partition, default is 4
   * @param predictKey key to store predicted result
   * @return
   */
  def predictImage(imageFrame: ImageFrame,
                   outputLayer: String = null,
                   shareBuffer: Boolean = false,
                   batchPerPartition: Int = 4,
                   predictKey: String = ImageFeature.predict): ImageFrame

  /**
   * Set weight and bias for the module
   * @param newWeights array of weights and bias
   * @return
   */
  def setWeightsBias(newWeights: Array[Tensor[T]]): IModule[A, B, T]

  /**
   * Get weight and bias for the module
   * @return array of weights and bias
   *
   */
  def getWeightsBias(): Array[Tensor[T]]

  /**
   * save weights and bias to file
   * @param path file to save
   * @param overWrite whether to overwrite or not
   */
  def saveWeights(path: String, overWrite: Boolean): Unit
  /**
   * load pretrained weights and bias to current module
   * @param weightPath file to store weights and bias
   * @param matchAll whether to match all layers' weights and bias,
   *                 if not, only load existing pretrained weights and bias
   * @return current module
   */
  def loadWeights(weightPath: String, matchAll: Boolean = true): IModule[A, B, T]

  /**
   * copy weights from another model, mapping by layer name
   * @param srcModel model to copy from
   * @param matchAll whether to match all layers' weights and bias,
   * @return current module
   */
  def loadModelWeights(srcModel: Module[Float], matchAll: Boolean = true): IModule[A, B, T]

  /**
   * Build graph: some other modules point to current module
   * @param nodes upstream module nodes
   * @return node containing current module
   */
  def inputs(nodes : ModuleNode[T]*): ModuleNode[T]

  /**
   * Build graph: some other modules point to current module
   * @param nodes upstream module nodes in an array
   * @return node containing current module
   */
  def inputs(nodes : Array[ModuleNode[T]]): ModuleNode[T]

  /**
   * Build graph: some other modules point to current module
   * @param first distinguish from another inputs when input parameter list is empty
   * @param nodesWithIndex upstream module nodes and the output tensor index. The start index is 1.
   * @return node containing current module
   */
  def inputs(first: (ModuleNode[T], Int), nodesWithIndex : (ModuleNode[T], Int)*): ModuleNode[T]

  /**
   * Find a module with given name. If there is no module with given name, it will return None. If
   * there are multiple modules with the given name, an exception will be thrown.
   * @param name
   * @return
   */
  def apply(name : String): Option[IModule[Activity, Activity, T]]

  /**
   * use ValidationMethod to evaluate module
   * @param dataset dataset for test
   * @param vMethods validation methods
   * @param batchSize total batchsize of all partitions,
   *                  optional param and default 4 * partitionNum of dataset
   * @return
   */
  def evaluate(dataset: RDD[Sample[T]],
               vMethods: Array[ValidationMethod[T]],
               batchSize: Option[Int] = None): Array[(ValidationResult, ValidationMethod[T])]

  def evaluate(dataSet: LocalDataSet[MiniBatch[T]],
               vMethods: Array[ValidationMethod[T]]
              ): Array[(ValidationResult, ValidationMethod[T])]

  def quantize(): Module[T]


  /**
   * Generate graph module with start nodes
   * @param startNodes
   * @return
   */
  def toGraph(startNodes: ModuleNode[T]*): Graph[T]


  def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]])

  def getLine(): String

  def getNumericType(): TensorDataType

  private[bigdl] def getEndNodes(startNodes: Array[ModuleNode[T]]): Array[ModuleNode[T]]

  private[nn] def allocateAs(dest: Activity): Activity

}


