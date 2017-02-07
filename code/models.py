import chainer
import chainer.links as L
import chainer.functions as F

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