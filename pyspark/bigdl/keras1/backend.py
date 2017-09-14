import new
import types # TODO: support python3
import tempfile
from bigdl.util.common import callBigDlFuncWithoutMappingReturn
from bigdl.util.common import get_spark_context
from bigdl.nn.layer import Model as BModel
# > import types
# > x.method = types.MethodType(method, x)
from keras.models import Sequential, Model
import numpy as np

def create_tmp_path():
    tmp_file = tempfile.NamedTemporaryFile(prefix="bigdl")
    tmp_file.close()
    return tmp_file.name

def new_fit():


def register_bigdl_backend():
    Model.fit = new_fit

# class BigDLBackend:
#     def _new_fit(
#             model,
#             x,
#             y,
#             batch_size=32,
#             nb_epoch=10,
#             verbose=1,
#             callbacks=[],
#             validation_split=0.,
#             validation_data=None,
#             shuffle=True,
#             class_weight=None,
#             sample_weight=None,
#             **kwargs):
#         self.optimizer = model.optimizer
#         self.loss = model.loss
#
#         model_path = create_tmp_path()
#         with open(model_path, "w") as json_file:
#             json_file.write(model.to_json())
#
#     def install_bigdl_backend(model):
#         model.__old_fit = model.fit
#
#         model.fit = new.instancemethod(_new_fit, model, None)
#         # model.load_weights = new.instancemethod(ModelLoader.load_weights, filepath, by_name=False)



class ModelLoader:
    @staticmethod
    def load_definition(model_path, bigdl_type="float"):
        bigdl_graph = callBigDlFuncWithoutMappingReturn(bigdl_type,
                                                        "kerasLoadGraph", model_path)
        return BModel.from_jvalue(bigdl_graph)

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
    def keras_name_to_Layer(kmodel):
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
        keras_name_to_layer = ModelLoader.keras_name_to_Layer(kmodel)

        if 'nb_layers' in f.attrs:
            raise Exception("not supported format of old version")
        else:
            blayers = bmodel.executions()
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
                bigdl_weights = ParameterConverter.convert_weights(
                        keras_name_to_layer[name].__class__.__name__,
                                                                   weight_values)
                bigdl_layer.set_weights(bigdl_weights)


class ParameterConverter:
    @staticmethod
    def convert_weights(layer_name, wb):
        if layer_name == "Dense":
            return [np.transpose(wb[0]), wb[1]]
        else:
            return wb


