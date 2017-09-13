import new
import types # TODO: support python3
import tempfile
from bigdl.util.common import callBigDlFuncWithoutMappingReturn
from bigdl.util.common import get_spark_context
from bigdl.nn.layer import Model as BModel
# > import types
# > x.method = types.MethodType(method, x)
from keras.models import Sequential, Model


def create_tmp_path():
    tmp_file = tempfile.NamedTemporaryFile(prefix="bigdl")
    tmp_file.close()
    return tmp_file.name

def _new_fit(
        self,
        x,
        y,
        batch_size=32,
        nb_epoch=10,
        verbose=1,
        callbacks=[],
        validation_split=0.,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        **kwargs):
    model = self
    model_path = create_tmp_path()
    with open(model_path, "w") as json_file:
        json_file.write(model.to_json())
    # model.save(model_path)
    #     callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
    #                   weight_init_method, bias_init_method)
    # bigdl_graph = callJavaFunc(get_spark_context(), "loadGraph", model_path)

def load_graph(model_path, bigdl_type="float"):
    bigdl_graph = callBigDlFuncWithoutMappingReturn(bigdl_type,
                                                    "kerasLoadGraph", model_path)
    return BModel.of(bigdl_graph)



def install_bigdl_backend(model):
    model.__old_fit = model.fit
    model.fit = new.instancemethod(_new_fit, model, None)
    model.load_weights = new.instancemethod(new_load_weights, filepath, by_name=False)

# this code is from keras
# model is a BModel
def new_load_weights(model, filepath, by_name=False):
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
        pass #load_weights_from_hdf5_group_by_name(f)
    else:
        load_weights_from_hdf5_group(f)
    if hasattr(f, 'close'):
        f.close()

def load_weights_from_hdf5_group(f):
    '''Weight loading is based on layer order in a list
    (matching model.flattened_layers for Sequential models,
    and model.layers for Model class instances), not
    on layer names.
    Layers that have no weights are skipped.
    '''


    if 'nb_layers' in f.attrs:
       raise Exception("not supported format of old version")
    else:
        # New file format.
        filtered_layers = []
        for layer in flattened_layers:
            weights = layer.weights
            if weights:
                filtered_layers.append(layer)
        flattened_layers = filtered_layers

        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
        filtered_layer_names = []
        for name in layer_names:
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            if len(weight_names):
                filtered_layer_names.append(name)
        layer_names = filtered_layer_names
        if len(layer_names) != len(flattened_layers):
            raise Exception('You are trying to load a weight file '
                            'containing ' + str(len(layer_names)) +
                            ' layers into a model with ' +
                            str(len(flattened_layers)) + ' layers.')

        # We batch weight value assignments in a single backend call
        # which provides a speedup in TensorFlow.
        weight_value_tuples = []
        for k, name in enumerate(layer_names):
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            weight_values = [g[weight_name] for weight_name in weight_names]
            layer = flattened_layers[k]
            symbolic_weights = layer.weights
            if len(weight_values) != len(symbolic_weights):
                raise Exception('Layer #' + str(k) +
                                ' (named "' + layer.name +
                                '" in the current model) was found to '
                                'correspond to layer ' + name +
                                ' in the save file. '
                                'However the new layer ' + layer.name +
                                ' expects ' + str(len(symbolic_weights)) +
                                ' weights, but the saved weights have ' +
                                str(len(weight_values)) +
                                ' elements.')
            if layer.__class__.__name__ == 'Convolution1D':
                # This is for backwards compatibility with
                # the old Conv1D weights format.
                w = weight_values[0]
                shape = w.shape
                if shape[:2] != (layer.filter_length, 1) or shape[3] != layer.nb_filter:
                    # Legacy shape: (self.nb_filter, input_dim, self.filter_length, 1)
                    assert shape[0] == layer.nb_filter and shape[2:] == (layer.filter_length, 1)
                    w = np.transpose(w, (2, 3, 1, 0))
                    weight_values[0] = w
            weight_value_tuples += zip(symbolic_weights, weight_values)
        K.batch_set_value(weight_value_tuples)