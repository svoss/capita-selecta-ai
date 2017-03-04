import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
# This a simple language model, directly copied from the tutorial
class RNNLM(chainer.Chain):
    def __init__(self, n_vocab, n_units, train=True):
        super(RNNLM, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            l1=L.LSTM(n_units, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.Linear(n_units, n_vocab),
        )
        self.n_units = n_units
        self.n_vocab = n_vocab
        # Initialize with uniform distribution, expect for our linear tranformation layer
        # for param in self.params():
        #    param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

        self.train = train

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        h2 = self.l1(F.dropout(h0, train=self.train))
        h3 = self.l2(F.dropout(h2, train=self.train))
        y = self.l3(F.dropout(h3, train=self.train))
        return y


# This network builds a bit different network that add layers that we need to be able to learn our translation
# First we need the translation layer
class TranslationMatrixRNN(chainer.Chain):
    def __init__(self, n_units, n_vocab, train=True):
        super(TranslationMatrixRNN, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            l0=L.Linear(n_units, n_units, nobias=True),
            l1=L.LSTM(n_units, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.Linear(n_units, n_vocab),
        )
        self.n_vocab = n_vocab
        self.n_units = n_units

        # Our linear transformation layer starts with
        for param in self.l0.params():
            param.data[...] = np.eye(n_units)
        self.train = train

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l0(h0)
        h2 = self.l1(F.dropout(h1, train=self.train))
        h3 = self.l2(F.dropout(h2, train=self.train))
        y = self.l3(F.dropout(h3, train=self.train))
        return y


def load_rnn_model(f, n_vocab, n_units):
    """
    Loads a simple pre-trained rnn language model in a classifier object
    Exact training procedure is explained in language model notebook
    """
    model = L.Classifier(RNNLM(n_vocab, n_units))
    model.compute_accuracy = False
    chainer.serializers.load_npz(f, model)
    return model


def transform_to_translation_matrix_model(flow_model, fit_model):
    """
    Loads the weight parameters of two language models into an translation recurrent neural network
    The flow_model is the model of which the weight of recurrent parts will be retrieved, this represent the flow in that language
    The fit model is the language model of the to translate model, the embeded values and output layer will be used
    """
    flow_rnn = flow_model.predictor
    fit_rnn = fit_model.predictor

    TRNN = TranslationMatrixRNN(flow_rnn.n_units, fit_rnn.n_vocab)

    # Embed layer should be copied from the to fit layer
    TRNN.embed.copyparams(fit_rnn.embed)
    TRNN.l3.copyparams(fit_rnn.l3)

    # These are the recurrent parameters
    TRNN.l1.copyparams(flow_rnn.l1)
    TRNN.l2.copyparams(flow_rnn.l2)

    # The l0 layer contains the translation matrix can't be copied

    return TRNN