import six


    #import bigdl.nn.criterion as bigdl_criterion
    #print inspect_subclass(bigdl_criterion, bigdl_criterion.Criterion)
def inspect_subclass(module, cls):
    import inspect
    result = inspect.getmembers(module,
                                predicate=lambda o: inspect.isclass(o) and \
                                                    issubclass(o, cls))
    return [i[0] for i in result]


def get_from_module(identifier, module_params, module_name,
                    instantiate=False, kwargs=None):
    """Retrieves a class or function member of a module.

    First checks `_GLOBAL_CUSTOM_OBJECTS` for `module_name`, then checks `module_params`.

    # Arguments
        identifier: the object to retrieve. It could be specified
            by name (as a string), or by dict. In any other case,
            `identifier` itself will be returned without any changes.
        module_params: the members of a module
            (e.g. the output of `globals()`).
        module_name: string; the name of the target module. Only used
            to format error messages.
        instantiate: whether to instantiate the returned object
            (if it's a class).
        kwargs: a dictionary of keyword arguments to pass to the
            class constructor if `instantiate` is `True`.

    # Returns
        The target object.

    # Raises
        ValueError: if the identifier cannot be found.
    """
    if isinstance(identifier, six.string_types):
        res = None
        if not res:
            res = module_params.get(identifier)
        if not res:
            raise ValueError('Invalid ' + str(module_name) + ': ' +
                             str(identifier))
        if instantiate and not kwargs:
            return res()
        elif instantiate and kwargs:
            return res(**kwargs)
        else:
            return res
    elif isinstance(identifier, dict):
        name = identifier.pop('name')
        res = None
        if not res:
            res = module_params.get(name)
        if res:
            return res(**identifier)
        else:
            raise ValueError('Invalid ' + str(module_name) + ': ' +
                             str(identifier))
    return identifier
