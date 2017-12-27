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

import java.nio.ByteOrder

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch, Sample}
import com.intel.analytics.bigdl.nn.Graph
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, IModule}
import com.intel.analytics.bigdl.optim.{ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.tensor.{Tensor, TensorDataType}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, ImageFrame}
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.utils.tf.TensorflowDataFormat
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

class IModuleAdapter[A <: Activity: ClassTag, B <: Activity: ClassTag, T: ClassTag]
(implicit ev: TensorNumeric[T]) extends IModule[A, B, T]{

  protected  var labor: IModule[A, B, T] = null

  def getOutput: B = labor.getOutput

  def getGradInput: A = labor.getGradInput

  def setGradInput(gradInput: A): Unit = labor.setGradInput(gradInput)

  def setOutputShape(outputShape: Activity): Unit = labor.setOutputShape(outputShape)

  def setInputShape(inputShape: Activity): Unit = labor.setGradInput(inputShape)

  def getInputShape(): Activity = labor.getInputShape()

  def getOutputShape(): Activity = labor.getOutputShape()

  def isBuilt(): Boolean = labor.isBuilt()

  def build(inputShape: Activity): Unit = labor.build(inputShape)

  override def computeOutputShape(inputShape: Activity): Activity = inputShape

  def forward(input: A): B = labor.forward(input)

  def backward(input: A, gradOutput: B): A = labor.backward(input, gradOutput)

  /**
   * Get the scale of gradientWeight
   */
  def getScaleW(): Double = labor.getScaleW()

  /**
   * Get the scale of gradientBias
   */
  def getScaleB(): Double = labor.getScaleB()

  /**
   * Set the scale of gradientWeight
   *
   * @param w the value of the scale of gradientWeight
   * @return this
   */
  def setScaleW(w: Double): IModule[A, B, T] = labor.setScaleW(w)

  /**
   * Set the scale of gradientBias
   *
   * @param b the value of the scale of gradientBias
   * @return this
   */
  def setScaleB(b: Double): IModule[A, B, T] = labor.setScaleB(b)

  /**
   * Clear cached activities to save storage space or network bandwidth. Note that we use
   * Tensor.set to keep some information like tensor share
   *
   * The subclass should override this method if it allocate some extra resource, and call the
   * super.clearState in the override method
   *
   * @return
   */
  def clearState() : IModule[A, B, T] = labor.clearState()

  override def toString(): String = getPrintName


  def getTimes(): Array[(IModule[_ <: Activity, _ <: Activity, T], Long, Long)] = labor.getTimes()

  def resetTimes(): Unit = labor.resetTimes()

  /**
   * freeze the module,
   * i.e. their parameters(weight/bias, if exists) are not changed in training process
   * if names is not empty,
   * set an array of layers that match the given ```names``` to be "freezed",
   *
   * @param names an array of layer names
   * @return current graph model
   */
  def freeze(names: String*): IModule[A, B, T] = labor.freeze(names : _*)

  /**
   * "unfreeze" module, i.e. make the module parameters(weight/bias, if exists)
   * to be trained(updated) in training process
   * if names is not empty, unfreeze layers that match given names
   *
   * @param names array of module names to unFreeze
   */
  def unFreeze(names: String*): IModule[A, B, T] = labor.unFreeze(names : _*)


  /**
   * Computes the output using the current parameter set of the class and input. This function
   * returns the result which is stored in the output field.
   *
   * @param input
   * @return
   */
  def updateOutput(input: A): B = labor.updateOutput(input)

  /**
   * Computing the gradient of the module with respect to its own input. This is returned in
   * gradInput. Also, the gradInput state variable is updated accordingly.
   *
   * @param input
   * @param gradOutput
   * @return
   */
  def updateGradInput(input: A, gradOutput: B): A = labor.updateGradInput(input, gradOutput)

  /**
   * Computing the gradient of the module with respect to its own parameters. Many modules do not
   * perform this step as they do not have any parameters. The state variable name for the
   * parameters is module dependent. The module is expected to accumulate the gradients with
   * respect to the parameters in some variable.
   *
   * @param input
   * @param gradOutput
   */
  def accGradParameters(input: A, gradOutput: B): Unit = labor.accGradParameters(input, gradOutput)

  /**
   * If the module has parameters, this will zero the accumulation of the gradients with respect
   * to these parameters. Otherwise, it does nothing.
   */
  def zeroGradParameters(): Unit = labor.zeroGradParameters()

  def updateParameters(learningRate: T): Unit = labor.updateParameters(learningRate)

  /**
   * This method compact all parameters and gradients of the model into two tensors. So it's easier
   * to use optim method
   *
   * @return
   */
  def getParameters(): (Tensor[T], Tensor[T]) = labor.getParameters()

  /**
   * This function returns two arrays. One for the weights and the other the gradients
   * Custom modules should override this function if they have parameters
   *
   * @return (Array of weights, Array of grad)
   */
  def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = labor.parameters()

  /**
   * Get extra parameter in this module.
   * Extra parameter means the trainable parameters beside weight and bias. Such as runningMean
   * and runningVar in BatchNormalization.
   *
   * The subclass should override this method if it has some parameters besides weight and bias.
   *
   * @return an array of tensor
   */
  def getExtraParameter(): Array[Tensor[T]] = labor.getExtraParameter()

  /**
   * Set extra parameter to this module.
   * Extra parameter means the trainable parameters beside weight and bias. Such as runningMean
   * and runningVar in BatchNormalization.
   *
   * @return this
   */
  def setExtraParameter(extraParam: Array[Tensor[T]]): IModule[A, B, T] =
  labor.setExtraParameter(extraParam)

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
  def getParametersTable(): Table = labor.getParametersTable()

  def training(): IModule[A, B, T] = labor.training()

  def evaluate(): IModule[A, B, T] = labor.evaluate()

  def isTraining(): Boolean = labor.isTraining()

  def reset(): Unit = labor.reset()

  def setLine(line: String): IModule[A, B, T] = labor.setLine(line)

  /**
   * get execution engine type
   */
  def checkEngineType(): IModule[A, B, T] = labor.checkEngineType()

  def cloneModule(): IModule[A, B, T] = labor.cloneModule()  // TODO: clone itself??

  /**
   * Save this module to path.
   * @param path path to save module, local file system, HDFS and Amazon S3 is supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx"
   *             Amazon S3 path should be like "s3a://bucket/xxx"
   * @param overWrite if overwrite
   * @return self
   */
  @deprecated("please use recommended saveModule(path, overWrite)")
  def save(path : String, overWrite: Boolean = false) : IModule[A, B, T] = {
    labor.save(path, overWrite)
  }

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
                 overWrite: Boolean = false) : IModule[A, B, T] = {
    labor.saveModule(path, weightPath, overWrite)
  }

  /**
   * Save this module definition to path.
   * @param path path to save module, local file system, HDFS and Amazon S3 is supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx"
   *             Amazon S3 path should be like "s3a://bucket/xxx"
   * @param overWrite if overwrite
   * @return self
   */
  def saveDefinition(path : String, overWrite: Boolean = false) : IModule[A, B, T] = {
    labor.saveDefinition(path, overWrite)
  }

  def saveTorch(path : String, overWrite: Boolean = false) : IModule[A, B, T] = {
    labor.saveTorch(path, overWrite)
  }

  def saveCaffe(prototxtPath: String, modelPath: String,
                useV2 : Boolean = true, overwrite : Boolean = false) : IModule[A, B, T] = {
    labor.saveCaffe(prototxtPath, modelPath, useV2, overwrite)
  }

  def saveTF(
              inputs : Seq[(String, Seq[Int])],
              path: String,
              byteOrder: ByteOrder = ByteOrder.LITTLE_ENDIAN,
              dataFormat: TensorflowDataFormat = TensorflowDataFormat.NHWC): IModule[A, B, T] = {
    labor.saveTF(inputs, path, byteOrder, dataFormat)
  }

  /**
   * module predict, return the probability distribution
   * @param dataset dataset for prediction
   * @param batchSize total batchSize for all partitions.
   *                  if -1, default is 4 * partitionNumber of datatset
   * @param shareBuffer whether to share same memory for each batch predict results
   */
  def predict(dataset: RDD[Sample[T]],
              batchSize: Int = -1,
              shareBuffer: Boolean = false): RDD[Activity] = {
    labor.predict(dataset, batchSize, shareBuffer)
  }
  /**
   * module predict, return the predict label
   * @param dataset dataset for prediction
   * @param batchSize total batchSize for all partitions.
   *                  if -1, default is 4 * partitionNumber of dataset
   */
  def predictClass(dataset: RDD[Sample[T]], batchSize: Int = -1): RDD[Int] = {
    labor.predictClass(dataset, batchSize)
  }

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
                   predictKey: String = ImageFeature.predict): ImageFrame = {
    labor.predictImage(imageFrame, outputLayer, shareBuffer, batchPerPartition, predictKey)
  }

  /**
   * Set weight and bias for the module
   * @param newWeights array of weights and bias
   * @return
   */
  def setWeightsBias(newWeights: Array[Tensor[T]]): IModule[A, B, T] =
   labor.setWeightsBias(newWeights)

  /**
   * Get weight and bias for the module
   * @return array of weights and bias
   *
   */
  def getWeightsBias(): Array[Tensor[T]] = labor.getWeightsBias()

  /**
   * save weights and bias to file
   * @param path file to save
   * @param overWrite whether to overwrite or not
   */
  def saveWeights(path: String, overWrite: Boolean): Unit = labor.saveWeights(path, overWrite)
  /**
   * load pretrained weights and bias to current module
   * @param weightPath file to store weights and bias
   * @param matchAll whether to match all layers' weights and bias,
   *                 if not, only load existing pretrained weights and bias
   * @return current module
   */
  def loadWeights(weightPath: String, matchAll: Boolean = true): IModule[A, B, T] = {
    labor.loadWeights(weightPath, matchAll)
  }

  /**
   * copy weights from another model, mapping by layer name
   * @param srcModel model to copy from
   * @param matchAll whether to match all layers' weights and bias,
   * @return current module
   */
  def loadModelWeights(srcModel: Module[Float], matchAll: Boolean = true): IModule[A, B, T] =
    labor.loadModelWeights(srcModel = srcModel, matchAll = matchAll)


  /**
   * Find a module with given name. If there is no module with given name, it will return None. If
   * there are multiple modules with the given name, an exception will be thrown.
   * @param name
   * @return
   */
  def apply(name : String): Option[IModule[Activity, Activity, T]] = labor.apply(name)

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
               batchSize: Option[Int] = None): Array[(ValidationResult, ValidationMethod[T])] =
  labor.evaluate(dataset, vMethods, batchSize)

  def evaluate(dataSet: LocalDataSet[MiniBatch[T]],
               vMethods: Array[ValidationMethod[T]]
              ): Array[(ValidationResult, ValidationMethod[T])] = labor.evaluate(dataSet, vMethods)

  def quantize(): Module[T] = labor.quantize()


  /**
   * Generate graph module with start nodes
   * @param startNodes
   * @return
   */
  def toGraph(startNodes: ModuleNode[T]*): Graph[T] = labor.toGraph(startNodes : _*)


  def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) =
    labor.getClassTagNumerics()

  def getLine(): String = labor.getLine()

  def getNumericType(): TensorDataType = labor.getNumericType()

  private[bigdl] def getEndNodes(startNodes: Array[ModuleNode[T]]): Array[ModuleNode[T]] = {
    labor.getEndNodes(startNodes)
  }

  private[nn] def allocateAs(dest: Activity): Activity = labor.allocateAs(dest)

}
