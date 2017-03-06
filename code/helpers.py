from wiki_dataset import get_wiki_dataset
import numpy as np
import chainer
from chainer import training, reporter
import sys
from chainer.training import extension
import chainer.links as L

import chainer.serializer as serializer_module
import chainer.training.trigger as trigger_module


def retrieve_and_split(dump, max_sequence_size=250000,word_th=5):
    """
    From a wikidump file will create a test, validation and training set
    :param dump: full path to file
    :return:
    """
    seq, voc = get_wiki_dataset(dump, max_sequence_size,word_th)
    seq = seq.astype(np.int32)

    val_start = int(len(seq) * .9)
    test_start = int(val_start * .9)
    train = seq[:test_start]
    test = seq[test_start:val_start]
    val = seq[val_start:]

    return train, val, test, voc


def export_dataset(dump, to_file, max_sequence_size=250000,word_th=5):
    """ Export dataset, can be usefull if you want to use the same dataset in other environment
    :param dump:
    :param to_file:
    :param max_sequence_size:
    :param word_th:
    :return:
    """
    seq, voc = get_wiki_dataset(dump, max_sequence_size, word_th)
    np.savez(to_file, seq=seq, voc=voc)


def read_dataset(file):
    """
    Reads dataset that was exported by export_dataset
    :param file:
    :return:
    """
    np.load(file)
    seq,voc = np.load(file)
    seq =  seq.astype(np.int32)
    return seq,voc


def retrieve_and_split(dump, max_sequence_size=250000,word_th=5):
    """
    From a wikidump file will create a test, validation and training set
    :param dump: full path to file
    :return:
    """
    seq, voc = get_wiki_dataset(dump, max_sequence_size,word_th)
    seq = seq.astype(np.int32)

    val_start = int(len(seq) * .9)
    test_start = int(val_start * .9)
    train = seq[:test_start]
    test = seq[test_start:val_start]
    val = seq[val_start:]

    return train, val, test, voc


# Dataset iterator to create a batch of sequences at different positions.
# This iterator returns a pair of current words and the next words. Each
# example is a part of sequences starting from the different offsets
# equally spaced within the whole sequence.
class ParallelSequentialIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size  # batch size
        # Number of completed sweeps over the dataset. In this case, it is
        # incremented if every word is visited at least once after the last
        # increment.
        self.epoch = 0
        # True if the epoch is incremented at the last iteration.
        self.is_new_epoch = False
        self.repeat = repeat
        length = len(dataset)
        # Offsets maintain the position of each sequence in the mini-batch.
        self.offsets = [i * length // batch_size for i in range(batch_size)]
        # NOTE: this is not a count of parameter updates. It is just a count of
        # calls of ``__next__``.
        self.iteration = 0

    def __next__(self):
        # This iterator returns a list representing a mini-batch. Each item
        # indicates a different position in the original sequence. Each item is
        # represented by a pair of two word IDs. The first word is at the
        # "current" position, while the second word at the next position.
        # At each iteration, the iteration count is incremented, which pushes
        # forward the "current" position.
        length = len(self.dataset)
        if not self.repeat and self.iteration * self.batch_size >= length:
            # If not self.repeat, this iterator stops at the end of the first
            # epoch (i.e., when all words are visited once).
            raise StopIteration
        cur_words = self.get_words()
        self.iteration += 1
        next_words = self.get_words()

        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

        return list(zip(cur_words, next_words))

    @property
    def epoch_detail(self):
        # Floating point version of epoch.
        return self.iteration * self.batch_size / len(self.dataset)

    def get_words(self):
        # It returns a list of current words.
        return [self.dataset[(offset + self.iteration) % len(self.dataset)]
                for offset in self.offsets]

    def serialize(self, serializer):
        # It is important to serialize the state to be recovered on resume.
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)


# Custom updater for truncated BackProp Through Time (BPTT)
class BPTTUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, bprop_len, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.bprop_len = bprop_len

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        loss = 0
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Progress the dataset iterator for bprop_len words at each iteration.
        for i in range(self.bprop_len):
            # Get the next batch (a list of tuples of two word IDs)
            batch = train_iter.__next__()

            # Concatenate the word IDs to matrices and send them to the device
            # self.converter does this job
            # (it is chainer.dataset.concat_examples by default)
            x, t = self.converter(batch, self.device)

            # Compute the loss at this time step and accumulate it
            loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))

        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters


# Routine to rewrite the result dictionary of LogReport to add perplexity
# values
def compute_perplexity(result):
    result['perplexity'] = np.exp(result['main/loss'])
    if 'validation/main/loss' in result:
        result['val_perplexity'] = np.exp(result['validation/main/loss'])


def seconds_to_str(t):
    """
    Convert seconds to nicely formatted string
    :param t:
    :return:
    """
    ""
    return "%d:%02d:%02d.%03d" % \
        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
            [(t*1000,),1000,60,60])

def get_config():
    import os
    import ConfigParser
    config = ConfigParser.RawConfigParser()
    config.read(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.ini'))

    return config