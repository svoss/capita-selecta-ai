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
from optimizer import LayerFreezing
import os
import sys
import numpy as np
from chainer import cuda

def _load_dataset(com,fit_dump,flow_dump,max_seq_size):

    com.add_text("Fit wiki dump", fit_dump)
    com.add_text("Flow wiki dump", flow_dump)

    # for flow we only need length of voc
    _,_,_, voc = retrieve_and_split(flow_dump, max_seq_size)
    n_vocab_flow = len(voc)
    # for fit we need everything
    train, val, test, voc = retrieve_and_split(fit_dump, max_seq_size)
    n_vocab_fit = len(voc)

    return train, val, test, voc, n_vocab_fit, n_vocab_flow


def _build_trans_model(com, lm_fit, lm_flow, n_vocab_fit, n_vocab_flow):
    com.add_text("Language model flow", lm_flow)
    com.add_text("Language model fit", lm_fit)
    com.add_text("Fit voc size", "%d" % n_vocab_fit)
    com.add_text("Flow voc size", "%d" % n_vocab_flow)

    # Load original models
    print ("Loading flow model %s #voc:%d" % (lm_flow, n_vocab_flow))
    flow_model = load_rnn_model(lm_flow, n_vocab_flow, 800)
    print("Loading fit model %s #voc:%d" % (lm_flow, n_vocab_fit))
    fit_model = load_rnn_model(lm_fit, n_vocab_fit, 800)

    print ("Building new translation matrix model")
    matrix_trans_rnn = transform_to_translation_matrix_model(flow_model, fit_model)
    return matrix_trans_rnn


def _dataset_iterator(com, train, test, val, batch_size,test_mode):
    if test_mode:
        print("Running in test mode: cutting test, train and val set to 100 elements each")
        train = train[:1000]
        test = test[:100]
        val = val[:100]
        if batch_size > 100:
            batch_size = 100
    com.add_text("Batch size", batch_size)
    com.add_text("Test mode", 'Yes' if test_mode else 'No')

    train_iter = ParallelSequentialIterator(train, batch_size)
    val_iter = ParallelSequentialIterator(val, 1, repeat=False)
    test_iter = ParallelSequentialIterator(test, 1, repeat=False)

    return train_iter, test_iter, val_iter

def _get_evaluator(model):
    """
    Version of model that can be used to verify performance
    :return:
    """

    eval_model = model.copy()  # Model with shared params and distinct states
    eval_rnn = eval_model.predictor
    eval_rnn.train = False

    return eval_model


def _build_model(trnn, gpu):
    print("Init model complete")
    model = L.Classifier(trnn)
    model.compute_accuracy = False  # we only want the perplexity
    if gpu >= 0:
        chainer.cuda.get_device(gpu).use()  # make the GPU current
        model.to_gpu()

    return model

def evaluate(com, eval_rnn, test_iter, eval_model, gpu, name="Final loss"):
    print('test')
    eval_rnn.reset_state()
    evaluator = extensions.Evaluator(test_iter, eval_model, device=gpu)
    result = evaluator()
    print('Test loss:', float(result['main/loss']))
    com.add_text(name, result['main/loss'])


def run_training(com, eval_model, eval_rnn, model, grad_clip, train_iter, brpoplen,gpu,epoch,out,val_iter,test_mode,resume):


    # Set up an optimizer
    optimizer = chainer.optimizers.SGD(lr=.5)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))

    # ignore training of certain layers
    optimizer.add_hook(LayerFreezing(['embed', 'l1', 'l2']))

    # Set up a trainer
    updater = BPTTUpdater(train_iter, optimizer, brpoplen, gpu)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

    trainer.extend(extensions.Evaluator(
        val_iter, eval_model, device=gpu,
        # Reset the RNN state at the beginning of each evaluation
        eval_hook=lambda _: eval_rnn.reset_state()))

    interval = 10 if test_mode else 1000

    trainer.extend(extensions.ExponentialShift('lr', 0.25),
                   trigger=(15, 'epoch'))
    trainer.extend(extensions.LogReport(postprocess=compute_perplexity,
                                        trigger=(interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'perplexity', 'val_perplexity']
    ), trigger=(interval, 'iteration'))
    trainer.extend(extensions.ProgressBar(
        update_interval=10 if test_mode else 125))
    trainer.extend(extensions.snapshot())
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'))
    if resume:
        chainer.serializers.load_npz(resume, trainer)
    date = time.strftime("%Y-%m-%d_%H-%M-%S")
    fn_a = 'loss_%s.png' % date
    loss_r = extensions.PlotReport(['validation/main/loss', 'main/loss'], 'epoch', file_name=fn_a)

    trainer.extend(loss_r)

    trainer.run()

    com.add_image(os.path.join(out, fn_a), "loss")



def check_loss(com,lm, dump, gpu=-1, batch_size=128, max_seq_size=250000, test_mode=True):
    """
    Checks loss on tet set of the translation model build by using the same fit and flow model, without any training this
    model should give the same loss as the original language model

    :param com:
    :param lm:
    :param dump:
    :param gpu:
    :param batch_size:
    :param max_seq_size:
    :param test_mode:
    :return:
    """

    com.add_text("Task", "Checking")
    train, test, val ,voc = retrieve_and_split(dump, max_seq_size)
    n_vocab = len(voc)

    train_iter, test_iter, val_iter = _dataset_iterator(com, train, test, val, batch_size, test_mode)

    print("Original version")
    model = load_rnn_model(lm, n_vocab, 800)

    # port to gpu
    model.compute_accuracy = False  # we only want the perplexity
    if gpu >= 0:
        chainer.cuda.get_device(gpu).use()  # make the GPU current
        model.to_gpu()

    # Evaluation copy
    eval_model = _get_evaluator(model)
    eval_rnn = eval_model.predictor
    evaluate(com, eval_rnn, test_iter, eval_model, gpu, "Original model")


    print("Translation version")
    trnn = _build_trans_model(com, lm, lm, n_vocab, n_vocab)
    model = _build_model(trnn, gpu)

    # Evaluation copy
    eval_model = _get_evaluator(model)
    eval_rnn = eval_model.predictor
    evaluate(com, eval_rnn, test_iter, eval_model, gpu, "Translation model")


def check_weights_update(com, lm_fit, lm_flow, fit_dump, flow_dump, test_mode=False, epoch=5, batch_size=128, gpu=-1, out='result', grad_clip=True, brpoplen=35, resume='', max_seq_size=250000):
    """ This method checks if my optimizer indeed doesn't update the weights of my modle
    :return:
    """
    train, val, test, voc, n_vocab_fit, n_vocab_flow = _load_dataset(com, fit_dump, flow_dump, max_seq_size)
    trnn = _build_trans_model(com, lm_fit, lm_flow,n_vocab_fit,n_vocab_flow)
    train_iter, test_iter, val_iter = _dataset_iterator(com, train, test , val, batch_size, test_mode)


    model = _build_model(trnn, gpu)

    before = {}

    print("Before")
    for name, params in model.namedparams():
        # might only work with gpu:
        before[name] = np.array(params.data.get(), dtype=np.float32)
        print(name, np.sum(before[name]))

    # Evaluation copy
    eval_model = _get_evaluator(model)
    eval_rnn = eval_model.predictor

    #
    run_training(com, eval_model, eval_rnn, model, grad_clip, train_iter, brpoplen, gpu, epoch, out, val_iter, test_mode, resume)

    print("After...")
    for name, params in model.namedparams():
        # might only work with gpu:
        after = np.array(params.data.get(), dtype=np.float32)
        diff = np.subtract(after,before[name])
        print(name, np.sum(diff))


def train(com, lm_fit, lm_flow, fit_dump, flow_dump, test_mode=False, epoch=5, batch_size=128, gpu=-1, out='result', grad_clip=True, brpoplen=35, resume='', max_seq_size=250000):
    """

    """

    com.add_text("Task", "training")
    com.add_text("Output folder", out)
    train, val, test, voc, n_vocab_fit, n_vocab_flow = _load_dataset(com, fit_dump, flow_dump, max_seq_size)
    trnn = _build_trans_model(com, lm_fit, lm_flow,n_vocab_fit,n_vocab_flow)
    train_iter, test_iter, val_iter = _dataset_iterator(com, train, test , val, batch_size, test_mode)


    model = _build_model(trnn, gpu)

    # Evaluation copy
    eval_model = _get_evaluator(model)
    eval_rnn = eval_model.predictor

    run_training(com, eval_model, eval_rnn, model, grad_clip, train_iter, brpoplen, gpu, epoch, out, val_iter, test_mode, resume)


    # Evaluate the final model
    evaluate(com, eval_rnn, test_iter, eval_model, gpu)



def main():
    config = get_config()
    # Wraps arg parse functionallity around train function so that it can be provided as arguments
    parser = argparse.ArgumentParser(description='Trains a language model from a wiki dataset')
    parser.add_argument('lm_fit', help='The wiki dump name to train a language model for')
    parser.add_argument('lm_flow', help='Name of the model, used in exported files etc')
    parser.add_argument('fit_dump', help='The wiki dump name to train a language model for')
    parser.add_argument('flow_dump', help='Name of the model, used in exported files etc')
    parser.add_argument('--test-mode', help="makes dataset smaller to see if the script actually runs", action='store_true')
    parser.add_argument('--epochs',type=int,default=5, help="Number of epochs to run for")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--gpu',type=int, default=-1, help="Gpu to use")
    parser.add_argument('--out',default='result',help="Folder to put results")
    parser.add_argument('--grad-clip', default=True, help="Clip gradients")
    parser.add_argument('--brpoplen', type=int, default=35)
    parser.add_argument('--resume', default='')
    parser.add_argument('--max-seq-size', default=250000,type=int)
    args = parser.parse_args()
    com = Communication(args.out)
    com.add_text("Type", "Translation via matrix")

    # keep time
    com.add_text("Start date", time.strftime("%c"))
    start = time.time()

    train(com, args.lm_fit, args.lm_flow,args.fit_dump,args.flow_dump, args.test_mode, args.epochs, args.batch_size, args.gpu, args.out, args.grad_clip, args.brpoplen, args.resume, args.max_seq_size)
    diff = time.time() - start
    com.add_text('time',seconds_to_str(diff))
    com.send_slack(config.get('slack','channel'),config.get('slack','api_token'))




if __name__ == "__main__":
    main()