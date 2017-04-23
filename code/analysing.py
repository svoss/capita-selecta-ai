"""
This file contains code to analyse trained models
"""
from operator import itemgetter
from heapq import nlargest
from chainer.functions import softmax
import numpy as np


# Shared helping functions
def create_inverse_voc(voc):
    """ Creates inverse vocabulary from word to index
    """
    return dict([(word,idx) for idx,word in enumerate(voc)])


def map_line_to_seq(line,inverse_voc):
    """ Converts a string(sentence) to a sequence of integers
    Will also tokenize the sentence
    """
    return [inverse_voc[w] if w in inverse_voc else inverse_voc['<below_th>'] for w in tokenize(line)]


def fill_till_max(x,filler=-1,n=100):
    """ Will make an array of fixed size n, will use x to fill this array. 
    If len(x) < n will fill the rest with filler. 
    """
    return [x[i] if len(x) > i else filler for i in range(n)]


def map_seq_to_sentence(seq, voc):
    """ Maps seqs back to a readable sentence
    """
    return " ".join([voc[int(w)] for w in seq]).replace(" <eos>",".")


class TextGenerator():
    """ Generates text from language model
    
    """
    TOP_N = 5

    def __init__(self, lm, voc):
        self.lm = lm
        self.voc = voc
        self.inv_voc = create_inverse_voc(voc)

    def generate_text(self, seeds, max_len=100):
        self.lm.reset_state()
        # matrix of sentences in rows, words in columns
        text_idx = np.array([fill_till_max(map_line_to_seq(s, self.inv_voc), n=max_len) for s in seeds], dtype=np.int32)

        # i is the to predict word column
        for i in range(2, max_len):
            # Our input is all words before the one to predict
            before = i - 1

            # calculate probabilty
            x = self.lm(text_idx[:, before])
            d = softmax(x).data
            next_words = []
            d = np.delete(d, 0, 1)
            for r in range(d.shape[0]):
                top = nlargest(TextGenerator.TOP_N, enumerate(d[r, :]), itemgetter(1))
                idx = [x[0] for x in top]
                probs = [x[1] for x in top]
                probs = np.array(probs) / np.sum(probs)
                next_words.append(np.random.choice(idx, p=probs) + 1)

            for si, w in enumerate(next_words):
                if text_idx[si, i] < 0:
                    text_idx[si, i] = w

        return [map_seq_to_sentence(s, self.voc) for s in text_idx]


