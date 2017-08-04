import bigdl.nn.criterion as bigdl_criterion
from bigdl.keras1.utility import *

#loss_name_to_object = {"categorical_crossentropy": "ClassNLLCriterion"}

def categorical_crossentropy():
    return bigdl_criterion.ClassNLLCriterion()

def get(identifier):
    return get_from_module(identifier, globals(), 'objective', True)
    # if name in loss_name_to_object:
    #     return getattr(bigdl_criterion, loss_name_to_object[name])
    # raise Exception("Unsupported loss: %s" % name)