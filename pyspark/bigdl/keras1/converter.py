
import bigdl.nn.layer as BLayer
import bigdl.nn.initialization_method as BInit
class BigDLLayerConverter:

    def create(self, class_name, klayer, kclayer):
        api = getattr(self, "create_" + class_name.lower())
        return api(klayer, kclayer)


    def create_dense(self, klayer, kclayer):
        config = kclayer["config"]
        blayer = BLayer.Linear(
            input_size=config["input_dim"],
            output_size=config["output_dim"],
            with_bias=config["bias"],
            wRegularizer=self.to_bigdl_reg(config["W_regularizer"]),
            bRegularizer=self.to_bigdl_reg(config["b_regularizer"])
        )
        return self.combo_parameter_layer(blayer, config)

    def create_activation(self, klayer, kclayer):
        return self.to_bigdl_activation(klayer.activation, klayer.name)

    def create_dropout(self, klayer, kclayer):
        return BLayer.Dropout(klayer.p).setName(klayer.name)

    def to_bigdl_ordering(self, order):
        if order == "tf":
            return "NHWC"
        elif order == "th":
            return "NCHW"
        else:
            raise Exception("Unsupport ordering: %s" % order)
    def to_bigdl_padding(self, border_mode):
        if border_mode == "same":
            return (-1, -1)
        elif border_mode == "valid":
            return (0, 0)
        else:
            raise Exception("Unsupported border mode: %s" % border_mode)

    def create_convolution2d(self, klayer, kclayer):
        config = kclayer["config"]
        bigdl_order = self.to_bigdl_ordering(klayer.dim_ordering)
        if bigdl_order == "NCHW":
            stack_size = klayer.batch_input_shape[1]
        elif bigdl_order == "NHWC":
            stack_size = klayer.batch_input_shape[3]

        bpadW, bpadH = self.to_bigdl_padding(klayer.border_mode)
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
                 with_bias=True,
                 data_format=bigdl_order,
                 bigdl_type="float")

        return self.combo_parameter_layer(blayer, config)

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