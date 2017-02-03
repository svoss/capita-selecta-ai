# Unsupervised word to word language translation using rnn

This repository contains the code that I used for a research in the capita selecta ai course at radboud university. In this research I wanted to see if I could learn a word to word translation of two languages without the use of any examples. Instead I assume that languages share certain patterns in which certain concepts or words would occur. And thus by matching these patterns we could see which words are likely to be in

## Code 
The scripts in the [code](https://github.com/svoss/unsupervised-language-translation-using-rnn/tree/master/code) folder. This scripts can be run stand alone(as a batch) and imported in other code. In the notebooks I developed the code and provide a more detailed explanation.

All my recurrent networks are implemented in Chainer.

My code consists of 3 parts:
- **Wiki dataset**: Where I obtain my datasets by extracting text from a wikidump, using the chainer helper functions
- **Language model(lm)**: Where I train a simple language model that predicts the next word in a sequence. Using the dataset obtained from wikpedia and the pbt tutorial from chainer.
- **Unsupervised word 2 word(uw2w)**: Where I use the network parameters of one language to predict the words of the other language. While doing so I learn a linear transformation matrix that maps words from language a to b.