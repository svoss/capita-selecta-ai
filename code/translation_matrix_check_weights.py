from __future__ import division
from __future__ import print_function
import sys


import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

from chainer import training
from chainer.training import extensions
import argparse

from models import load_rnn_model, transform_to_translation_matrix_model
from helpers import retrieve_and_split, ParallelSequentialIterator, BPTTUpdater, compute_perplexity,get_config,seconds_to_str
import time
from communication import Communication
import os
import sys


from translation_matrix import check_weights_update

def main():
    # Wraps arg parse functionallity around train function so that it can be provided as arguments
    parser = argparse.ArgumentParser(description='Trains a language model from a wiki dataset')
    parser.add_argument('lm_fit', help='The wiki dump name to train a language model for')
    parser.add_argument('lm_flow', help='Name of the model, used in exported files etc')
    parser.add_argument('fit_dump', help='The wiki dump name to train a language model for')
    parser.add_argument('flow_dump', help='Name of the model, used in exported files etc')
    parser.add_argument('--test-mode', help="makes dataset smaller to see if the script actually runs",
                        action='store_true')
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs to run for")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--gpu', type=int, default=-1, help="Gpu to use")
    parser.add_argument('--out', default='result', help="Folder to put results")
    parser.add_argument('--grad-clip', default=True, help="Clip gradients")
    parser.add_argument('--brpoplen', type=int, default=35)
    parser.add_argument('--resume', default='')
    parser.add_argument('--max-seq-size', default=250000, type=int)
    args = parser.parse_args()
    com = Communication(args.out)


    check_weights_update(com, args.lm_fit, args.lm_flow, args.fit_dump, args.flow_dump, args.test_mode, args.epochs, args.batch_size,
          args.gpu, args.out, args.grad_clip, args.brpoplen, args.resume, args.max_seq_size)


if __name__ == "__main__":
    main()