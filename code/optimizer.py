class LayerFreezing(object):
    """ Optimizer hook that freezes the parameters updates of a certain set of layers by setting the gradients to zero
    Make sure it the latest hook
    """
    name = "LayerFreezing"
    def __init__(self, layers):
        """

        :param layers: Layers to freeze
        """
        self.layers = layers

    def __call__(self, opt):
        for name,params in opt.target.namedparams():
            print name

