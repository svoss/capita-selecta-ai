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


from translation_matrix import check_loss

def main():
    config = get_config()
    # Wraps arg parse functionallity around train function so that it can be provided as arguments
    parser = argparse.ArgumentParser(description='Evaluates a TRNN ')
    parser.add_argument('dump', help='The wiki dump name to train a language model for')
    parser.add_argument('lm', help='Path to language model')
    parser.add_argument('--test-mode', help="makes dataset smaller to see if the script actually runs", action='store_true')
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--gpu',type=int, default=-1, help="Gpu to use")
    parser.add_argument('--max-seq-size', default=250000,type=int)
    parser.add_argument('--out', default='result', help="Folder to put results")
    args = parser.parse_args()
    com = Communication(args.out)
    com.add_text("Type", "Translation matrix")

    # keep time
    com.add_text("Start date", time.strftime("%c"))
    start = time.time()

    check_loss(com, args.lm, args.dump, args.gpu, args.batch_size, args.max_seq_size, args.test_mode)
    diff = time.time() - start
    com.add_text('time',seconds_to_str(diff))
    com.send_slack(config.get('slack','channel'),config.get('slack','api_token'))




if __name__ == "__main__":
    main()