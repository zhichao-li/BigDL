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

package com.intel.analytics.bigdl.example.keras

import com.intel.analytics.bigdl.nn.keras._
import com.intel.analytics.bigdl.utils.Shape

object LeNet {
  def apply(classNum: Int): Sequential[Float] = {
    import com.intel.analytics.bigdl.nn.keras._
    import com.intel.analytics.bigdl.utils.Shape

    val model = Sequential[Float]()
    model.add(Reshape(Array(1, 28, 28), inputShape = Shape(28, 28, 1)))
    model.add(Convolution2D[Float](6, 5, 5, activation = "tanh").setName("conv1_5x5"))
    model.add(MaxPooling2D())
    model.add(Convolution2D[Float](12, 5, 5, activation = "tanh").setName("conv2_5x5"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense[Float](100, activation = "tanh").setName("fc1"))
    model.add(Dense[Float](classNum).setName("fc2"))
    model.add(Activation("softmax"))
    return model
  }
}
