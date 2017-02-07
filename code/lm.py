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

from models import RNNLM
from helpers import retrieve_and_split, ParallelSequentialIterator, BPTTUpdater, compute_perplexity



def train(dump, name, test_mode=False, epoch=5, batch_size=128, gpu=-1, out='result', grad_clip=True, brpoplen=35, resume=''):
    """

    """
    train, val, test, voc = retrieve_and_split(dump)
    n_vocab = len(voc)
    # n_vocab=10
    print("Going to run %s" % name)
    print("#training: %d, #val: %d, #test: %d" % (len(train), len(val), len(test)))
    print("#vocabulary: %d" % n_vocab)

    if test_mode:
        print("Running in test mode, cutting test,train and val set to 100 elements each")
        train = train[:100]
        test = test[:100]
        val = val[:100]

    train_iter = ParallelSequentialIterator(train, batch_size)
    val_iter = ParallelSequentialIterator(val, 1, repeat=False)
    test_iter = ParallelSequentialIterator(test, 1, repeat=False)

    # Prepare an RNNLM model
    print("Creating model")
    rnn = RNNLM(n_vocab, 800)
    print("Init model complete")
    model = L.Classifier(rnn)
    model.compute_accuracy = False  # we only want the perplexity
    if gpu >= 0:
        chainer.cuda.get_device(gpu).use()  # make the GPU current
        model.to_gpu()

    # Set up an optimizer
    optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))

    # Set up a trainer
    updater = BPTTUpdater(train_iter, optimizer, brpoplen, gpu)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

    eval_model = model.copy()  # Model with shared params and distinct states
    eval_rnn = eval_model.predictor
    eval_rnn.train = False
    trainer.extend(extensions.Evaluator(
        val_iter, eval_model, device=gpu,
        # Reset the RNN state at the beginning of each evaluation
        eval_hook=lambda _: eval_rnn.reset_state()))

    interval = 10 if test_mode else 500
    trainer.extend(extensions.LogReport(postprocess=compute_perplexity,
                                        trigger=(interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'perplexity', 'val_perplexity']
    ), trigger=(interval, 'iteration'))
    trainer.extend(extensions.ProgressBar(
        update_interval=1 if test_mode else 10))
    trainer.extend(extensions.snapshot())
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'))
    if resume:
        chainer.serializers.load_npz(resume, trainer)

    trainer.run()

    # Evaluate the final model
    print('test')
    eval_rnn.reset_state()
    evaluator = extensions.Evaluator(test_iter, eval_model, device=gpu)
    result = evaluator()
    print('test perplexity:', np.exp(float(result['main/loss'])))


def main():
    # Wraps arg parse functionallity around train function so that it can be provided as arguments
    parser = argparse.ArgumentParser(description='Trains a language model from a wiki dataset')
    parser.add_argument('dump', help='The wiki dump name to train a language model for')
    parser.add_argument('name', help='Name of the model, used in exported files etc')
    parser.add_argument('--test-mode', help="makes dataset smaller to see if the script actually runs", action='store_true')
    parser.add_argument('--epochs',type=int,default=5, help="Number of epochs to run for")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--gpu',type=int, default=-1, help="Gpu to use")
    parser.add_argument('--out',default='result',help="Folder to put results")
    parser.add_argument('--grad-clip', default=True, help="Clip gradients")
    parser.add_argument('--brpoplen', type=int, default=35)
    parser.add_argument('--resume', default='')

    args = parser.parse_args()

    train(args.dump, args.name, args.test_mode, args.epochs, args.batch_size, args.gpu, args.out, args.grad_clip, args.brpoplen, args.resume )


if __name__ == "__main__":
    main()