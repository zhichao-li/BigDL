"""Utilities related to model visualization."""
import os


try:
    # pydot-ng is a fork of pydot that is better maintained.
    import pydot_ng as pydot
except ImportError:
    # Fall back on pydot if necessary.
    import pydot
if not pydot.find_graphviz():
    raise ImportError('Failed to import pydot. You must install pydot'
                      ' and graphviz for `pydotprint` to work.')


def model_to_dot(model, show_shapes=False, show_layer_names=True):
    """Converts a Keras model to dot format.

    # Arguments
        model: A Keras model instance.
        show_shapes: whether to display shape information.
        show_layer_names: whether to display layer names.

    # Returns
        A `pydot.Dot` instance representing the Keras model.
    """
    dot = pydot.Dot()
    dot.set('rankdir', 'TB')
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')

    for layer in model.layers:
        layer_name = layer.name
        class_name = layer.__class__.__name__
        # Create node's label.
        if show_layer_names:
            label = '{}: {}'.format(layer_name, class_name)
        else:
            label = class_name

        # Rebuild the label as a table including input/output shapes.
        if show_shapes:
            try:
                outputlabels = str(layer.output_shape)
            except AttributeError:
                outputlabels = 'multiple'
            if hasattr(layer, 'input_shape'):
                inputlabels = str(layer.input_shape)
            elif hasattr(layer, 'input_shapes'):
                inputlabels = ', '.join(
                    [str(ishape) for ishape in layer.input_shapes])
            else:
                inputlabels = 'multiple'
            label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (label, inputlabels, outputlabels)
        node = pydot.Node(layer.name, label=label)
        dot.add_node(node)

    for layer in model.layers:
        for in_node in layer.inbound_keras_nodes:
            dot.add_edge(pydot.Edge(in_node.name, layer.name))
    return dot


def plot(model, to_file='model.png', show_shapes=False, show_layer_names=True):
    dot = model_to_dot(model, show_shapes, show_layer_names)
    _, extension = os.path.splitext(to_file)
    if not extension:
        extension = 'png'
    else:
        extension = extension[1:]
    dot.write(to_file, format=extension)
