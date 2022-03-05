import codecs
import numpy as np
from operator import itemgetter
import os
import logging
import torch
from gensim.models import KeyedVectors
import hashlib

import json


class Metrics:
    '''
    metric object
    '''


    def __init__(self):
        '''
        init
        '''
        self.metrics = {}

    def __len__(self):
        return len(self.metrics)

    def add(self, metric_name, metric_value):
        """ Add metric value for the given metric name

        :param metric_name: metric name
        :type metric_name: str
        :param metric_value: metric value
        :type metric_value:
        :return: None
        :rtype: None
        """
        self.metrics[metric_name] = metric_value



def loda_dataset(filePath):
    '''
    obtain train_set
    :param filePath:
    :return:
    '''

    raw_dataSet_list = []

    with codecs.open(filePath, 'r', 'utf-8') as f:
        for line in f:
            line = line.strip()
            raw_dataSet_list.append(line)

    return raw_dataSet_list;



def load_biContext(filePath,word2index,index2word):
    '''
    obtain the biContexts of entities from files.
    :param filePath:
    :param word2index:
    :param index2word:
    :return:
    '''

    BiContextIndexDict = {}
    BiContextWordDict = {}
    with codecs.open(filePath, 'r', 'utf-8') as f:
        for line in f:
            try:
                line = line.strip()
                lineList = line.split("###")
                contextList = lineList[1].split("\t")
                contextList = itemgetter(*contextList)(word2index)
                BiContextIndexDict[word2index[lineList[0]]] = np.array(contextList)
                BiContextWordDict[lineList[0]] = np.array(contextList)

            except:
                print(line)
    context_array = np.zeros([len(word2index),10],dtype = np.long)

    for k,v in BiContextIndexDict.items():
        context_array[k,0:v.shape[0]] = v

    return BiContextWordDict,BiContextIndexDict,context_array

def load_embedding(fi, embed_name="word2vec"):
    '''
    obtain the embeddings from files.
    :param fi:
    :param embed_name:
    :return:

        - embedding : embeddings
        - index2word: map from element index to element
        - word2index: map from element to element index
        - vocab_size: size of element pool
        - embed_dim: embedding dimension

    :rtype: (gensim.KeyedVectors, list, dict, int, int)
    '''

    if embed_name == "word2vec":
        embedding = KeyedVectors.load_word2vec_format(fi)
    else:
        # TODO: allow training embedding from scratch later
        print("[ERROR] Please specify the pre-trained embedding")
        exit(-1)

    vocab_size, embed_dim = embedding.vectors.shape
    index2word = embedding.index2word
    word2index = {word: index for index, word in enumerate(index2word)}
    return embedding, index2word, word2index, vocab_size, embed_dim


def my_logger(name='', log_path='./'):
    '''
    Create a python logger
    :param name:
    :param log_path:
    :return:
    '''
    logger = logging.getLogger(name)
    if len(logger.handlers) != 0:
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)
        print('create new logger: {}'.format(name))
    else:
        print('create new logger: {}'.format(name))
    fn = os.path.join(log_path, 'run-{}.log'.format(name))
    if os.path.exists(fn):
        print('[warning] log file {} already existed'.format(fn))
    else:
        print('saving log to {}'.format(fn))


    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d][%(funcName)s] %(levelname)s-> %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S', filename=fn, filemode='w')

    console = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d][%(funcName)s] %(levelname)s-> %(message)s',
                                  datefmt='%a %d %b %Y %H:%M:%S')

    console.setFormatter(formatter)

    logger.addHandler(console)

    return logger


def print_args(args, interested_args="all"):
    '''
    Print arguments in command line
    :param args:
    :param interested_args:
    :return:
    '''
    print("\nParameters:")
    if interested_args == "all":
        for attr, value in sorted(args.__dict__.items()):
            print("\t{}={}".format(attr.upper(), value))
    else:
        for attr, value in sorted(args.__dict__.items()):
            if attr in interested_args:
                print("\t{}={}".format(attr.upper(), value))
    print('-' * 89)


class ModelResults:
    '''
    A result class for saving results to file
    '''

    def __init__(self, resultPath ,resultName):
        self._filename = os.path.join(resultPath, resultName)
        if os.path.exists(resultPath):
            with open(self._filename, 'a+') as fl:
                pass
        else:
            os.makedirs(resultPath)
            with open (self._filename, 'a+') as fl:
                pass


    def _hash(self, x):

        return hashlib.md5(json.dumps(x, sort_keys=True).encode('utf-8')).hexdigest()

    def save_metrics(self, hyperparams, metrics):
        '''
        Save model hyper-parameters and evaluation results to the file
        :param hyperparams:
        :param metrics:
        :return:
        '''
        result = metrics.metrics  # a dict
        result["hash"] = self._hash(hyperparams)
        result.update(hyperparams)
        with open(self._filename, 'a+') as out:
            out.write(json.dumps(result) + '\n')

    def __getitem__(self, hyperparams):
        params_hash = self._hash(hyperparams)

        with open(self._filename, 'r+') as fle:
            for line in fle:
                datum = json.loads(line)

                if datum['hash'] == params_hash:
                    del datum['hash']
                    return datum

        raise KeyError

    def __contains__(self, x):
        try:
            self[x]
            return True
        except KeyError:
            return False

    def __iter__(self):
        with open(self._filename, 'r+') as fle:
            for line in fle:
                datum = json.loads(line)

                del datum['hash']

                yield datum


def load_model(model, load_dir=None, load_prefix=None, steps=None,remark=None):
    '''
    load model from file
    :param model:
    :param load_dir:
    :param load_prefix:
    :param steps:
    :param remark:
    :return:
    '''

    if load_prefix == "whole_best":
        model.load_state_dict(torch.load(load_dir+"\\"+remark+"_whole_best.pt"))
    else:
        model_prefix = os.path.join(load_dir, load_prefix)
        model_path = "{}_steps_{}.pt".format(model_prefix, steps)
        model.load_state_dict(torch.load(model_path))

def save_model(model, save_dir, save_prefix, steps,remark=None):
    '''
    Save model to file
    :param model:
    :param save_dir:
    :param save_prefix:
    :param steps:
    :param remark:
    :return:
    '''
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if save_prefix == "whole_best":
        torch.save(model.state_dict(), save_dir+"\\"+remark+"_whole_best.pt")
    else:
        save_prefix = os.path.join(save_dir, save_prefix)
        save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
        torch.save(model.state_dict(), save_path)
