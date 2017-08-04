import bigdl.optim as bigdl_optim
from bigdl.keras1.utility import *

class Optimizer(object):
    pass


class Adagrad(Optimizer):
    """Adagrad optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        lr: float >= 0. Learning rate.
        epsilon: float >= 0.
        decay: float >= 0. Learning rate decay over each update.

    # References
        - [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    """

    def __init__(self, lr=0.01, epsilon=1e-8, decay=0., **kwargs):
        super(Adagrad, self).__init__(**kwargs)
        self.B = bigdl_optim.optimizer.Adagrad(learningrate=lr,
                                               learningrate_decay=epsilon,  #TODO: learningrate_decay is epsilon?
                 weightdecay=decay)

def get(identifier, kwargs=None):
    # Instantiate a Keras optimizer
    return get_from_module(identifier, globals(), 'optimizer',
                           instantiate=True, kwargs=kwargs).B