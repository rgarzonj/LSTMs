#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 11:29:54 2017

@author: rubengarzon
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import h5py
import os

import tensorflow as tf


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def _write_dict(w2idx):
    out = open("words.dict", "w")
    items = [(v, k) for k, v in w2idx.iteritems()]
    items.sort()
    for v, k in items:
        out.write(str(k) + ", " + str(v) + "\n")
    out.close()

def ptb_raw_data(data_path=None, store=False):
    """Load PTB raw data from data directory "data_path".
    Args:
      data_path: string path to the directory where simple-examples.tgz has
        been extracted.
    Returns:
      tuple (train_data, valid_data, vocabulary)
      where each of the data objects can be passed to PTBIterator.
    """

    train_path = os.path.join(data_path, "train.txt")
    valid_path = os.path.join(data_path, "valid.txt")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    vocabulary = len(word_to_id)

    print(len(train_data))
    if store:
        _write_dict(word_to_id)
        f = h5py.File("train.h5", "w")
        f["word_ids"] = train_data
        f.close()
    return train_data, valid_data, vocabulary

folder = "/Users/rubengarzon/Documents/Projects/phD/Repo/tensorflow-statereader/data/ptb-word"
a,b,c = ptb_raw_data (data_path=folder)

