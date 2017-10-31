
import bigdl.nn.initialization_method as BInit
import numpy as np
import bigdl.nn.layer as BLayer
import bigdl.optim.optimizer as boptimizer
import bigdl.nn.criterion as bcriterion
import bigdl.util.common as bcommon
import keras.optimizers as koptimizers
from keras.models import model_from_json
from keras.models import Sequential, Model
from bigdl.util.common import callBigDlFuncWithoutMappingReturn
import keras

def unsupport_exp(name):
    raise Exception("We don't support %s for now" % name)


class OptimConverter:

    @staticmethod
    def to_bigdl_metrics(metrics):
        metrics = bcommon.to_list(metrics)
        bmetrics = []
        for metric in metrics:
            if metric == "accuracy":
                bmetrics.append(boptimizer.Top1Accuracy())
            else:
                unsupport_exp(metric)
        return bmetrics

    @staticmethod
    def to_bigdl_criterion(kloss):
        # TODO: it may pass in an object and with parameters
        if kloss == "categorical_crossentropy":
            return bcriterion.ClassNLLCriterion()
        elif kloss == "mse":
            return bcriterion.MSECriterion()
        elif kloss == "binary_crossentropy":
            return bcriterion.BCECriterion()
        else:
            raise Exception("Not supported type: %s" % kloss)

    @staticmethod
    def to_bigdl_optim_method(koptim_method):
        # This is always be an object
        if isinstance(koptim_method, koptimizers.Adagrad):
            return boptimizer.Adagrad()
        elif isinstance(koptim_method, koptimizers.SGD):
            return boptimizer.SGD(learningrate=0.01)  # TODO: enrich parameters, sgd.lr return a variable!!!not float
        elif isinstance(koptim_method, koptimizers.Adam):
            return boptimizer.Adam()
        else:
            unsupport_exp(koptim_method)

class ModelLoader:

    @staticmethod
    def load_def_from_kmodel(kmodel):
        keras_model_path = bcommon.create_tmp_path()
        keras_model_path_json = keras_model_path + ".json"
        #keras_model_path_hdf5 = keras_model_path + ".hdf5"

        with open(keras_model_path_json, "w") as json_file:
            json_file.write(kmodel.to_json())
        #kmodel.save(keras_model_path_hdf5)

        # load bigdl model from file. TODO: no need to save json to file first
        return ModelLoader.load_def_from_json(keras_model_path_json)

    @staticmethod
    def load_def_from_json(keras_model_json_path, bigdl_type="float"):
        if bigdl_type != "float":
            raise Exception("we only support float32, not %s" % bigdl_type)
        return DefinitionLoader.from_json(keras_model_json_path).to_bigdl()

    # this code is from keras
    # model is a BModel
    @staticmethod
    def load_weights(bmodel, kmodel, filepath, by_name=False):
        '''Loads all layer weights from a HDF5 save file.

        If `by_name` is False (default) weights are loaded
        based on the network's topology, meaning the architecture
        should be the same as when the weights were saved.
        Note that layers that don't have weights are not taken
        into account in the topological ordering, so adding or
        removing layers is fine as long as they don't have weights.

        If `by_name` is True, weights are loaded into layers
        only if they share the same name. This is useful
        for fine-tuning or transfer-learning models where
        some of the layers have changed.
        '''
        import h5py
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']
        else:
            raise Exception("not supported format")
        if by_name:
            pass  # load_weights_from_hdf5_group_by_name(f)
        else:
            ModelLoader.load_weights_from_hdf5_strict(f, bmodel, kmodel)
        if hasattr(f, 'close'):
            f.close()

    @staticmethod
    def __keras_name_to_Layer(kmodel):
        layers = kmodel.layers
        return dict([(layer.name, layer) for layer in layers])

    @staticmethod
    def load_weights_from_hdf5_strict(f, bmodel, kmodel):
        '''Weight loading is based on layer order in a list
        (matching model.flattened_layers for Sequential models,
        and model.layers for Model class instances), not
        on layer names.
        Layers that have no weights are skipped.
        '''
        keras_name_to_layer = ModelLoader.__keras_name_to_Layer(kmodel)

        if 'nb_layers' in f.attrs:
            raise Exception("not supported format of old version")
        else:
            blayers = bmodel.executions()
            # TODO: get_weights is slow,
            # we should add another method to check if layer containing weigths or not
            bigdl_layers_wb = [layer for layer in blayers if layer.get_weights()]

            layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
            filtered_layer_names = []
            for name in layer_names:
                g = f[name]
                weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
                if len(weight_names):
                    filtered_layer_names.append(name)
            if len(filtered_layer_names) != len(bigdl_layers_wb):
                raise Exception('You are trying to load a weight file '
                                'containing ' + str(len(layer_names)) +
                                ' layers into a model with ' +
                                str(len(bigdl_layers_wb)) + ' layers.')
            # ensure the layers between bigdl and keras are identical
            bigdl_layers_wb = sorted(bigdl_layers_wb, key=lambda layer: layer.name())
            filtered_layer_names = sorted(filtered_layer_names)
            for bwlayer, klayer in zip():
                if bwlayer != klayer:
                    raise Exception(
                        "Found bigdl layer %s, keras layer: %s" % (bwlayer.name(), klayer))

            for k, name in enumerate(filtered_layer_names):
                g = f[name]
                weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
                # TODO: performance?
                weight_values = [np.array(g[weight_name]) for weight_name in weight_names]
                bigdl_layer = bigdl_layers_wb[k]
                # TODO: Check if the len of weights is equal
                # symbolic_weights = layer.weights
                # if len(weight_values) != len(symbolic_weights):
                #     raise Exception('Layer #' + str(k) +
                #                     ' (named "' + layer.name +
                #                     '" in the current model) was found to '
                #                     'correspond to layer ' + name +
                #                     ' in the save file. '
                #                     'However the new layer ' + layer.name +
                #                     ' expects ' + str(len(symbolic_weights)) +
                #                     ' weights, but the saved weights have ' +
                #                     str(len(weight_values)) +
                #                     ' elements.')
                if layer.__class__.__name__ == 'Convolution1D':
                    # This is for backwards compatibility with
                    # the old Conv1D weights format.
                    raise Exception("we don't support old format for now")
                bigdl_weights = WeightsConverter.to_bigdl_weights(
                        keras_name_to_layer[name].__class__.__name__,
                                                                   weight_values)
                bigdl_layer.set_weights(bigdl_weights)


class WeightsConverter:
    """
    Convert keras weights to bigdl weights
    The shape of weights would be changed if using different backend,
    so we only test against TensorFlow backend.
    TODO: Support th backend as well.
    """

    @staticmethod
    def get_converter(class_name):
        function_name = "convert_" + class_name.lower()
        if not hasattr(WeightsConverter, function_name):
            raise Exception("We don't support layer: %s for now" % class_name)
        converter = getattr(WeightsConverter, function_name)
        return converter


    @staticmethod
    # weights is a list of ndarray to a ndarray
    # convert keras weights per layer to bigdl format
    def to_bigdl_weights(class_name, weights):
        return WeightsConverter.get_converter(class_name)(weights)

    @staticmethod
    def get_bigdl_weigths_from_keras(k):
        if isinstance(k, keras.engine.Model):
            return WeightsConverter.get_weights_from_kmodel(k)
        elif isinstance(k, keras.engine.Layer):
            return WeightsConverter.get_bigdl_weights_from_klayer(k)
        else:
            raise Exception("Unsupport type: %s", k)

    @staticmethod
    def get_bigdl_weights_from_klayer(klayer):
        # not klayer.weights
        return WeightsConverter.to_bigdl_weights(klayer.__class__.__name__, klayer.get_weights())

    @staticmethod
    def get_weights_from_kmodel(kmodel):
        """
        Convert kmodel's weights to bigdl format.
        We are supposing the order is the same as the execution order.
        :param kmodel: keras model
        :return: list of ndarray
        """
        layers_with_weights = [layer for layer in kmodel.layers if layer.weights]
        bweights = []
        for klayer in layers_with_weights:
            # bws would be [weiths, bias] or [weights]
            bws = WeightsConverter.get_bigdl_weights_from_klayer(klayer)
            for w in bws:
                bweights.append(w)
        return bweights



    @staticmethod
    def convert_dense(weights):
        return [np.transpose(weights[0]), weights[1]]

    @staticmethod
    def convert_batchnormalization(weights):
        gamma = weights[0]
        beta = weights[1]
        return [gamma, beta]

    @staticmethod
    def convert_convolution2d(weights):
        # return weights
        weight = np.expand_dims(weights[0], 0) # bigdl has a leading dim with value 1
        if len(weights) > 1:
            return [weight, weights[1]]
        else:
            return [weight]
    @staticmethod
    def convert_convolution1d(weights):
        return WeightsConverter.convert_convolution2d(weights)

    @staticmethod
    def convert_embedding(weights):
        return weights

class DefinitionLoader:

    def __init__(self, kmodel):
        self.node_id_to_instance = {}
        self.node_id_to_layer = {}
        self.node_id_to_config_layer = {}
        self.kmodel = kmodel
        self.kconfig = self.kmodel.get_config()

        for layer in self.kmodel.layers:
            self.node_id_to_layer[layer.name] = layer

        if isinstance(self.kmodel, Sequential):
            for layer_config in self.kmodel.get_config():
                layer_name = layer_config["config"]["name"]
                self.node_id_to_config_layer[layer_name] = layer_config
        else:
            for layerConfig in self.kconfig["layers"]:
                self.node_id_to_config_layer[layerConfig["name"]] = layerConfig
        # Miss inbound_nodes in the layer config of Sequential comparing against functional api
        # "inbound_nodes": [
        #   [
        #     [
        #       "mixed3",
        #       0,
        #       0
        #     ]
        #   ]
        # ],


    @classmethod
    def from_kmodel(cls, kmodel):
        return cls(kmodel)

    @classmethod
    def from_json(cls, json_path):
        with open(json_path, "r") as jp:
            kmodel = model_from_json(jp.read())
            return DefinitionLoader.from_kmodel(kmodel)

    def _do_create_node(self, layer, clayer):
        if clayer["class_name"] == "InputLayer":
            input = BLayer.Input()
            input.element().set_name(layer.name) # cannot set name for node?
            self.node_id_to_instance[layer.name] = input
            return input
        bigdl_in_nodes = []
        for node in clayer["inbound_nodes"]:
            for out in node:
                out_name = out[0]
                out_index = out[1]
                out_tensor_index = out[2]
                if out_name not in self.node_id_to_instance:
                    self._do_create_node(self.node_id_to_layer[out_name],
                                         self.node_id_to_config_layer[out_name])
                bigdl_in_nodes.append(self.node_id_to_instance[out_name])

        blayer = LayerConverter().create(layer, clayer)
        new_bnode = blayer(bigdl_in_nodes)
        self.node_id_to_instance[layer.name] = new_bnode
        return new_bnode

    def _construct_bigdl_model(self):
        for clayer in self.kconfig["layers"]:
            if clayer["name"] not in self.node_id_to_instance:

                self._do_create_node(self.node_id_to_layer[clayer["name"]],
                                     clayer)
        ins = []
        for input_layer in self.kconfig["input_layers"]:
            name = input_layer[0]
            ins.append(self.node_id_to_instance[name])
        outs = []
        for output_layer in self.kconfig["output_layers"]:
            name = output_layer[0]
            outs.append(self.node_id_to_instance[name])
        return BLayer.Model(inputs=ins, outputs=outs)

    def _construct_bigdl_sequence(self):
        bseq = BLayer.Sequential()
        layerConverter = LayerConverter()
        for layer in self.kmodel.layers:
            blayer = layerConverter.create(layer, self.node_id_to_config_layer[layer.name])
            bseq.add(blayer)
        return bseq


    def to_bigdl(self):
        if isinstance(self.kmodel, Sequential):
            bmodel = self._construct_bigdl_sequence()
        elif isinstance(self.kmodel, Model):
            bmodel = self._construct_bigdl_model()
        return bmodel


class LayerConverter:

    def __check_is_share_weights(self, kclayer):
        # For Merge layer len(kclayer["inbound_nodes"]) is equal to 1
        # "inbound_nodes": [
        #                      [
        #                          [
        #                              "batchnormalization_194",
        #                              0,
        #                              0
        #                          ],
        #                          [
        #                              "batchnormalization_196",
        #                              0,
        #                              0
        #                          ],
        #                          [
        #                              "batchnormalization_199",
        #                              0,
        #                              0
        #                          ],
        #                          [
        #                              "batchnormalization_200",
        #                              0,
        #                              0
        #                          ]
        #                      ]
        #                  ],
        if "inbound_nodes" in kclayer and len(kclayer["inbound_nodes"]) > 1:
            raise Exception(
                "%s doesn't support multiple inputs with shared weights" % kclayer["class_name"])


    def create(self, klayer, kclayer):
        class_name = kclayer["class_name"]

        self.__check_is_share_weights(kclayer)

        if (hasattr(klayer, "b_constraint") and klayer.b_constraint) or \
           (hasattr(klayer, "W_constraint") and klayer.W_constraint):
            raise Exception("We don't support constraint for now")

        if (hasattr(klayer, "activity_regularizer") and klayer.activity_regularizer):
            raise Exception("We don't support activity_regularizer for now")

        function_name = "create_" + class_name.lower()
        if not hasattr(self, function_name):
            raise Exception("We don't support layer: %s for now" % class_name )

        api = getattr(self, function_name)
        return api(klayer, kclayer).set_name(klayer.name)

    def create_model(self, klayer, kclyer):
        return DefinitionLoader.from_kmodel(klayer).to_bigdl()

    def create_dense(self, klayer, kclayer):
        config = kclayer["config"]
        # Multiple inputs should share the same input_dim for Dense layer
        # We don't need to respect the tensor index for method `get_input_shape_at`
        # which is internel implementation and `get_input_shape_at` has hided that for us,
        # we only need to know the input index, not node index, not tensor index.
        input_shape = klayer.get_input_shape_at(0)
        blayer = BLayer.Linear(
            input_size=input_shape[1],
            output_size=config["output_dim"],
            with_bias=config["bias"],
            wRegularizer=self.to_bigdl_reg(config["W_regularizer"]),
            bRegularizer=self.to_bigdl_reg(config["b_regularizer"])
        )
        return self.combo_parameter_layer(blayer, config)

    def create_embedding(self, klayer, kclayer):
        config = kclayer["config"]
        input_shape = klayer.get_input_shape_at(0) # batch, seq_len
        seq_len = input_shape[1]
        if klayer.input_length and klayer.input_length != seq_len:
            raise Exception(
                "The input_length doesn't match: %s vs %s" % (seq_len, klayer.input_length))

        if (hasattr(klayer, "dropout") and klayer.dropout != 0):
            raise Exception("We don't support dropout for now")

        if (hasattr(klayer, "mask_zero") and klayer.mask_zero != False):
            raise Exception("We don't support mask_zero for now")

        bseq = BLayer.Sequential()
        blayer = BLayer.LookupTable(
                 n_index = klayer.input_dim,
                 n_output = klayer.output_dim,
                 padding_value=0.0,
                 norm_type=2.0,
                 should_scale_grad_by_freq=False,
                 wRegularizer= self.to_bigdl_reg(config["W_regularizer"]),
                 bigdl_type="float")
        bseq.add(BLayer.AddConstant(1.0, inplace=True)) # Add 1 as BigDL is one-based index
        bseq.add(blayer)
        return bseq


    def create_activation(self, klayer, kclayer):
        config = kclayer["config"]
        return self.to_bigdl_activation(config["activation"], klayer.name)

    def create_dropout(self, klayer, kclayer):
        return BLayer.Dropout(klayer.p)

    def create_flatten(self, klayer, kclayer):
        self.__check_is_share_weights(kclayer)
        input_shape = klayer.input_shape
        blayer = BLayer.Reshape([np.prod(input_shape[1:])], None)
        return blayer

    def create_reshape(self, klayer, kclayer):
        self.__check_is_share_weights(kclayer)
        blayer = BLayer.Reshape(klayer.target_shape, None)
        return blayer

    def create_merge(self, klayer, kclayer):
        self.__check_is_share_weights(kclayer)
        input_shape = klayer.get_input_shape_at(0)
        if klayer.mode == "concat":
            blayer = BLayer.JoinTable(
                 dimension = klayer.concat_axis,
                n_input_dims = len(input_shape[0]) - 1,
                 bigdl_type="float")
        elif klayer.mode == "sum":
            blayer = BLayer.CAddTable(
                 inplace=False,
                 bigdl_type="float")
        else:
            # TODO: Add more mode for this
            raise Exception("We don't support % right now" % kclayer.mode)
        return blayer

    def create_batchnormalization(self, klayer, kclayer):
        config = kclayer["config"]

        self.__check_is_share_weights(kclayer)

        if klayer.mode != 0:
            raise Exception(
                "Only support mode = 0 for now, but the current mode is: %s", klayer.mode)
        if config["gamma_regularizer"]:
            raise Exception("We don't support gamma_regularizer for now")

        if config["beta_regularizer"]:
            raise Exception("We don't support beta_regularizer for now")

        input_shape = klayer.get_input_shape_at(0)
        n_input_channel = input_shape[klayer.axis] # default is -1 which is channel-last

        # init gamma and beta
        # TODO: replace this with to_bigdl_init in the future
        gamma = self.get_value_from_init(klayer.gamma_init.func_name, (n_input_channel,))
        beta = self.get_value_from_init(klayer.beta_init.func_name, (n_input_channel,))

        blayer = BLayer.SpatialBatchNormalization(
                 n_output = n_input_channel,
                 eps=klayer.epsilon,
                 momentum=klayer.momentum,
                 affine=True,
                 init_weight=gamma,
                 init_bias=beta,
                 init_grad_weight=None,
                 init_grad_bias=None,
                 bigdl_type="float")
        return blayer

    def to_bigdl_2d_ordering(self, order):
        if order == "tf":
            return "NHWC"
        elif order == "th":
            return "NCHW"
        else:
            raise Exception("Unsupport ordering: %s" % order)

    def to_bigdl_2d_padding(self, border_mode):
        if border_mode == "same":
            return (-1, -1)
        elif border_mode == "valid":
            return (0, 0)
        else:
            raise Exception("Unsupported border mode: %s" % border_mode)

    def to_bigdl_1d_padding(self, border_mode, kernel_w):
        if border_mode == "same":
            raise Exception("We don't support padding for now")
            return int((kernel_w -1) / 2)
        elif border_mode == "valid":
            return 0
        else:
            raise Exception("Unsupported border mode: %s" % border_mode)

################# Layers with weights ############################# noqa

    def create_convolution1d(self, klayer, kclayer):
        config = kclayer["config"]
        input_shape = klayer.get_input_shape_at(0)
        # batch, steps, dim, batch is None here, so you cannot use it directly.
        stack_size = input_shape[2]

        bpadW, bpadH = self.to_bigdl_2d_padding(klayer.border_mode)
        seq = BLayer.Sequential()
        seq.add(BLayer.Reshape([input_shape[1], 1, input_shape[2]], True))
        # seq.add(BLayer.Transpose([]))
        # seq.add(BLayer.View([1, input_shape[1], input_shape[2]], num_input_dims=3))
        blayer = BLayer.SpatialConvolution(
                 n_input_plane = stack_size,
                 n_output_plane = klayer.nb_filter,
            kernel_w = 1,
            kernel_h =  klayer.filter_length,
            stride_w= 1,
            stride_h= klayer.subsample_length,
            #      kernel_w = klayer.filter_length,
            #      kernel_h = 1,
            #      stride_w= klayer.subsample_length,
            #      stride_h= 1,
                 pad_w= bpadW,
                 pad_h= bpadH,
                 n_group=1,
                 propagate_back=True,
                 wRegularizer = self.to_bigdl_reg(config["W_regularizer"]),
                 bRegularizer=self.to_bigdl_reg(config["b_regularizer"]),
                 init_weight=None,
                 init_bias=None,
                 init_grad_weight=None,
                 init_grad_bias=None,
                 with_bias=config["bias"],
                 data_format="NHWC",
                 bigdl_type="float")
        seq.add(blayer)
        seq.add(BLayer.Squeeze(3))

        return seq
        #return self.combo_parameter_layer(seq, config)

    def create_convolution2d(self, klayer, kclayer):
        config = kclayer["config"]
        bigdl_order = self.to_bigdl_2d_ordering(klayer.dim_ordering)
        input_shape = klayer.get_input_shape_at(0)

        if bigdl_order == "NCHW":
            stack_size = input_shape[1]
        elif bigdl_order == "NHWC":
            stack_size = input_shape[3]

        bpadW, bpadH = self.to_bigdl_2d_padding(klayer.border_mode)
        blayer = BLayer.SpatialConvolution(
                 n_input_plane = stack_size,
                 n_output_plane = klayer.nb_filter,
                 kernel_w = klayer.nb_col,
                 kernel_h = klayer.nb_row,
                 stride_w= klayer.subsample[0],
                 stride_h= klayer.subsample[1],
                 pad_w= bpadW,
                 pad_h= bpadH,
                 n_group=1,
                 propagate_back=True,
                 wRegularizer = self.to_bigdl_reg(config["W_regularizer"]),
                 bRegularizer=self.to_bigdl_reg(config["b_regularizer"]),
                 init_weight=None,
                 init_bias=None,
                 init_grad_weight=None,
                 init_grad_bias=None,
                 with_bias=config["bias"],
                 data_format=bigdl_order,
                 bigdl_type="float")

        return self.combo_parameter_layer(blayer, config)



############### Pooling Layers
    def create_maxpooling2d(self, klayer, kclayer):
        bigdl_order = self.to_bigdl_2d_ordering(klayer.dim_ordering)
        bpadW, bpadH = self.to_bigdl_2d_padding(klayer.border_mode)
        # The implementation in BigDL would refer the stack_size base on `format`
        blayer = BLayer.SpatialMaxPooling(
                 kw = klayer.pool_size[0],
                 kh = klayer.pool_size[1],
                 dw = klayer.strides[0],
                 dh = klayer.strides[1],
                 pad_w=bpadW,
                 pad_h=bpadH,
                 to_ceil=False,
                 format=bigdl_order,
                 bigdl_type="float")
        return blayer

    def create_maxpooling1d(self, klayer, kclayer):
        if klayer.border_mode != "valid":
            raise Exception("We don't support padding for now. please set border_mode to be valid")
        blayer = BLayer.TemporalMaxPooling(k_w = klayer.pool_length,
                                           d_w = klayer.stride,
                                           bigdl_type="float")
        return blayer

    def create_globalmaxpooling2d(self, klayer, kclayer):
        bigdl_order = self.to_bigdl_2d_ordering(klayer.dim_ordering)
        input_shape = klayer.get_input_shape_at(0)
        if bigdl_order == "NCHW":
            b_kw = input_shape[2]
            b_kh = input_shape[3]
        else:
            b_kw = input_shape[1]
            b_kh = input_shape[2]

        seq = BLayer.Sequential()
        # The implementation in BigDL would refer the stack_size base on `format`
        blayer = BLayer.SpatialMaxPooling(
            kw=b_kw,
            kh=b_kh,
            dw=0,
            dh=0,
            pad_w=0,
            pad_h=0,
            to_ceil=False,
            format=bigdl_order,
            bigdl_type="float"
        )
        seq.add(blayer)
        seq.add(BLayer.Squeeze(3, num_input_dims=3))
        seq.add(BLayer.Squeeze(2, num_input_dims=2))

        return seq

    def create_globalmaxpooling1d(self, klayer, kclayer):
        input_shape = klayer.get_input_shape_at(0) # batch, step, dim
        b_kw = input_shape[1]
        b_kh = 1

        seq = BLayer.Sequential()
        # The implementation in BigDL would refer the stack_size base on `format`
        blayer = BLayer.SpatialMaxPooling(
            kw=b_kw,
            kh=b_kh,
            dw=0,
            dh=0,
            pad_w=0,
            pad_h=0,
            to_ceil=False,
            format="NHWC",
            bigdl_type="float"
        )
        seq.add(blayer)
        seq.add(BLayer.Squeeze(2, num_input_dims=2))
        seq.add(BLayer.Squeeze(1, num_input_dims=1))

        return seq

    def create_averagepooling2d(self, klayer, kclayer):
        bigdl_order = self.to_bigdl_2d_ordering(klayer.dim_ordering)
        bpadW, bpadH = self.to_bigdl_2d_padding(klayer.border_mode)
        # The implementation in BigDL would refer the stack_size base on `format`
        blayer = BLayer.SpatialAveragePooling(
            kw=klayer.pool_size[0],
            kh=klayer.pool_size[1],
            dw=klayer.strides[0],
            dh=klayer.strides[1],
            pad_w=bpadW,
            pad_h=bpadH,
            global_pooling=False,
            ceil_mode=False,
            count_include_pad=True,
            divide=True,
            format=bigdl_order,
            bigdl_type="float"
        )
        return blayer

    def create_averagepooling1d(self, klayer, kclayer):
        input_shape = klayer.get_input_shape_at(0) # batch, steps, dim
        bpadW, bpadH = self.to_bigdl_2d_padding(klayer.border_mode)

        seq = BLayer.Sequential()
        seq.add(BLayer.View([1, input_shape[1], input_shape[2]], num_input_dims=3))
        # The implementation in BigDL would refer the stack_size base on `format`
        blayer = BLayer.SpatialAveragePooling(
            kw=klayer.pool_length,
            kh=1,
            dw=klayer.stride,
            dh=1,
            pad_w=bpadW,
            pad_h=bpadH,
            global_pooling=False,
            ceil_mode=False,
            count_include_pad=True,
            divide=True,
            format="NHWC",
            bigdl_type="float"
        )
        seq.add(blayer)
        return seq

    def create_globalaveragepooling2d(self, klayer, kclayer):
        bigdl_order = self.to_bigdl_2d_ordering(klayer.dim_ordering)
        input_shape = klayer.get_input_shape_at(0)
        if bigdl_order == "NCHW":
            b_kw = input_shape[2]
            b_kh = input_shape[3]
        else:
            b_kw = input_shape[1]
            b_kh = input_shape[2]

        seq = BLayer.Sequential()
        # The implementation in BigDL would refer the stack_size base on `format`
        blayer = BLayer.SpatialAveragePooling(
            kw=b_kw,
            kh=b_kh,
            dw=0,
            dh=0,
            pad_w=0,
            pad_h=0,
            global_pooling=False,
            ceil_mode=False,
            count_include_pad=True,
            divide=True,
            format=bigdl_order,
            bigdl_type="float"
        )
        seq.add(blayer)
        seq.add(BLayer.Squeeze(3, num_input_dims=3))
        seq.add(BLayer.Squeeze(2, num_input_dims=2))

        return seq

    def create_globalaveragepooling1d(self, klayer, kclayer):
        input_shape = klayer.get_input_shape_at(0) # batch, step, dim
        b_kw = 1
        b_kh = input_shape[1]

        seq = BLayer.Sequential()
        seq.add(BLayer.View([input_shape[1], 1, input_shape[2]], num_input_dims=2))
        # The implementation in BigDL would refer the stack_size base on `format`
        blayer = BLayer.SpatialAveragePooling(
            kw=b_kw,
            kh=b_kh,
            dw=0,
            dh=0,
            pad_w=0,
            pad_h=0,
            global_pooling=False,
            ceil_mode=False,
            count_include_pad=True,
            divide=True,
            format="NHWC",
            bigdl_type="float"
        )
        seq.add(blayer)
        seq.add(BLayer.Squeeze(2, num_input_dims=2)) # the index start from one but without batch
        seq.add(BLayer.Squeeze(1, num_input_dims=1))

        return seq

    def combo_parameter_layer(self, blayer, config):
        if config["W_constraint"] or config["b_constraint"]:
            raise Exception("Haven't support constraint yet")

        blayer.set_name(config["name"])
        if hasattr(blayer, "set_init_method"):
            blayer.set_init_method(self.to_bigdl_init(config["init"]),
                                   BInit.Zeros()) # Keras always set this to be zeros
        # "linear" meaning do nothing
        if config["activation"] != "linear" :
            activation = self.to_bigdl_activation(config["activation"],
                                                  "%s_%s" % (config["name"], config["activation"]))
            return self.fuse(blayer, activation)
        else:
            return blayer

    def to_bigdl_activation(self, activation_name, activation_id):
        activation = None
        if activation_name == "tanh":
            activation = BLayer.Tanh()
        elif activation_name == "sigmoid":
            activation = BLayer.Sigmoid()
        elif activation_name == "relu":
            activation = BLayer.ReLU()
        elif activation_name == "softmax":
            activation = BLayer.SoftMax()
        else:
            raise Exception("Unsupported activation type: %s" % activation_name)
        activation.set_name(activation_id)
        return activation

    def get_value_from_init(self, kinit_method, shape):
        if kinit_method == "zero":
            return np.zeros(shape)
        elif kinit_method == "one":
            return np.ones(shape)
        else:
            raise Exception("We don't support % for now", kinit_method)

    def to_bigdl_init(self, kinit_method): # kinit_method is a string
        init = None
        if kinit_method == "glorot_uniform":
            init = BInit.Xavier()
        elif kinit_method == "one":
            init = BInit.Ones()
        elif kinit_method == "zero":
            init = BInit.Zeros()
        else:
            raise Exception("Unsupported init type: %s" % init)
        return init


    def to_bigdl_reg(self, reg): # reg is a dict
        if reg:
           raise Exception("will support reg very soon")
        else:
            return None

    def fuse(self, src_blayer, activation): # activation is a string
        seq = BLayer.Sequential()
        seq.add(src_blayer)
        seq.add(activation)
        seq.set_name(src_blayer.name())
        return seq