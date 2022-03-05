from configs import read_configs, load_options
from tensorboardX import SummaryWriter
from tools import my_logger, load_embedding, load_biContext,loda_dataset
from modelRuning import runing
from PSO_opt import PSO_start
import entity_set
import os
import random
import torch
import numpy as np
import pandas as pd
import configparser
import logging



def mkdir(path):
    '''
    ***
    Create directory

    :param path: the path need to create
    :return: Void
    '''
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        pass


def read_ini(inikey,inivaluse):
    '''
    ***
    read the project initial settings from ini_parameters.ini
    :param inikey:
    :param inivaluse:
    :return:
    '''
    config = configparser.ConfigParser()
    config.read("ini_parameters.ini",encoding="utf-8")
    convaluse=config.get(inikey,inivaluse)
    return convaluse


if __name__ == '__main__':

    optionsList ={}
    optionsList["mode"] = read_ini("ini_parameters_mode","mode") # read the project mode from ini_parameters.ini
    if optionsList["mode"] == "train": # read the initial settings for train
        optionsList["dataset"] = read_ini("ini_parameters_train", "dataset")
        optionsList["mu"] = read_ini("ini_parameters_train", "mu")
        optionsList["delta"] = read_ini("ini_parameters_train", "delta")
        optionsList["kappa"] = read_ini("ini_parameters_train", "kappa")
        optionsList["lamb"] = read_ini("ini_parameters_train", "lamb")
    elif optionsList["mode"] == "eval_opt": # read the initial settings for eval_opt
        optionsList["dataset"] = read_ini("ini_parameters_evaluation_opt", "dataset")
        optionsList["mu"] = read_ini("ini_parameters_evaluation_opt", "mu")
        optionsList["delta"] = read_ini("ini_parameters_evaluation_opt", "delta")
        optionsList["kappa"] = read_ini("ini_parameters_evaluation_opt", "kappa")
        optionsList["lamb"] = read_ini("ini_parameters_evaluation_opt", "lamb")
        optionsList["processNum"] = read_ini("ini_parameters_evaluation_opt", "mutil_processNum")
    elif optionsList["mode"] == "test": # read the initial settings for eval_opt
        optionsList["dataset"] = read_ini("ini_parameters_test", "dataset")
        optionsList["mu"] = read_ini("ini_parameters_test", "mu")
        optionsList["delta"] = read_ini("ini_parameters_test", "delta")
        optionsList["kappa"] = read_ini("ini_parameters_test", "kappa")
        optionsList["lamb"] = read_ini("ini_parameters_test", "lamb")

    datasetDict = load_options(optionsList["dataset"]) # obtain all the filenames for dataset
    args = read_configs(datasetDict, optionsList)  #  obtain the initial settings for project

    if args.mode == "train": # train processing

        mu_list = args.mu
        for mm in mu_list:
            options = vars(args)
            random.seed(args.random_seed)  # initialize random seed
            torch.manual_seed(args.random_seed)  # initialize torch random seed
            np.random.seed(args.random_seed)  # initialize numpy random seed
            if args.device_id != -1:  # initialize cuda random seed
                torch.cuda.manual_seed_all(args.random_seed)
                torch.backends.cudnn.deterministic = True
            torch.set_printoptions(precision=9)  # Number of digits of precision for floating point output (default = 4)
            torch.set_num_threads(1)  # Sets the number of threads used for parallelizing operations.

            embedding, index2word, word2index, vocab_size, embed_dim = load_embedding(options["w2vPathe"], options[
                "w2vName"])  # need some times,  obtain the embeddings
            options["embedding"] = embedding
            options["index2word"] = index2word
            options["word2index"] = word2index
            options["vocabSize"] = vocab_size
            options["embed_dim"] = embed_dim
            print("Finish loading embedding: embed_dim = {}, vocab_size = {}".format(embed_dim, vocab_size))

            BiContextWordDict, BiContextIndexDict, contexArray = load_biContext(options["BiContextPath"],
                                                                                options["word2index"], options[
                                                                                    "index2word"])  # obtain the bicontext for the entity
            print("Finish loading bicontext.")
            options["BiContextWordDict"] = BiContextWordDict
            options["BiContextIndexDict"] = BiContextIndexDict
            options["contexArray"] = torch.from_numpy(contexArray)

            raw_training_dataSets = loda_dataset(options["trainingSetPath"])  # obtain train_set
            random.shuffle(raw_training_dataSets)  # shuffle the raw_dataSet_list
            training_set = entity_set.EntitySet("training_set", options["data_format"], options,
                                                raw_training_dataSets)  # obtain the training_set

            raw_testing_dataSets = loda_dataset(options["testingSetPath"])  # obtain train_set
            random.shuffle(raw_testing_dataSets)  # shuffle the raw_dataSet_list
            testing_set = entity_set.EntitySet("testing_set", options["data_format"], options,
                                               raw_testing_dataSets)  # obtain the testing_set
            options["training_set"] = training_set
            options["testing_set"] = testing_set


            options["mu"] = mm
            writer = SummaryWriter(log_dir="runs\\training_log_mu" + str(options["mu"]) + "_" + options["dataset"],
                                   comment=options["dataset"])
            mkdir("runs\\training_log_mu" + str(options["mu"]) + "_" + options["dataset"])
            logger = my_logger(name='training_logger',
                               log_path="runs\\training_log_mu" + str(options["mu"]) + "_" + options["dataset"])
            logger.setLevel(0)
            writer.add_text('Text', 'Hyper-parameters: {}'.format(options), 0)
            logger.info("#" * 50)
            logger.info("#" * 50)
            logger.info("Start train model, mu: {}".format(options["mu"]))
            logger.info("#" * 50)
            logger.info("#" * 50)
            runing(options, options["training_set"], options["testing_set"], mode="train", tb_writer=writer,
                   my_logger=logger)
            logger.info("#" * 50)
            logger.info("#" * 50)
            logger.info("Over train model, mu: {}".format(options["mu"]))
            logger.info("#" * 50)
            logger.info("#" * 50)
            writer.close()

    elif args.mode == "eval_opt": # eval_opt processing
        ## using the PSO to optimal the best evaluation parameters for the project
        options = vars(args)
        random.seed(args.random_seed)  # initialize random seed
        torch.manual_seed(args.random_seed)  # initialize torch random seed
        np.random.seed(args.random_seed)  # initialize numpy random seed
        if args.device_id != -1:  # initialize cuda random seed
            torch.cuda.manual_seed_all(args.random_seed)
            torch.backends.cudnn.deterministic = True

        torch.set_printoptions(precision=9)  # Number of digits of precision for floating point output (default = 4)
        torch.set_num_threads(1)  # Sets the number of threads used for parallelizing operations.

        embedding, index2word, word2index, vocab_size, embed_dim = load_embedding(options["w2vPathe"], options[
            "w2vName"])  # need some times,  obtain the embeddings

        options["embedding"] = embedding
        options["index2word"] = index2word
        options["word2index"] = word2index
        options["vocabSize"] = vocab_size
        options["embed_dim"] = embed_dim
        print("Finish loading embedding: embed_dim = {}, vocab_size = {}".format(embed_dim, vocab_size))

        BiContextWordDict, BiContextIndexDict, contexArray = load_biContext(options["BiContextPath"],
                                                                            options["word2index"], options[
                                                                                "index2word"])  # obtain the bicontext for the entity
        print("Finish loading bicontext.")
        options["BiContextWordDict"] = BiContextWordDict
        options["BiContextIndexDict"] = BiContextIndexDict
        options["contexArray"] = torch.from_numpy(contexArray)

        raw_training_dataSets = loda_dataset(options["trainingSetPath"])  # obtain train_set
        random.shuffle(raw_training_dataSets)  # shuffle the raw_dataSet_list
        training_set = entity_set.EntitySet("training_set", options["data_format"], options,
                                            raw_training_dataSets)  # obtain the training_set

        raw_testing_dataSets = loda_dataset(options["testingSetPath"])  # obtain train_set
        random.shuffle(raw_testing_dataSets)  # shuffle the raw_dataSet_list
        testing_set = entity_set.EntitySet("testing_set", options["data_format"], options,
                                           raw_testing_dataSets)  # obtain the testing_set
        options["training_set"] = training_set
        options["testing_set"] = testing_set


        mkdir("runs\\evaluation_log_" + options["dataset"])
        logger = my_logger(name="eval_optimal_logger", log_path="runs\\evaluation_log_" + options["dataset"])
        PSO_start(options,processNum=options["processNum"],logger=logger)  # start PSO OPT

    elif args.mode == "test": # test processing
        options = vars(args)
        random.seed(args.random_seed)  # initialize random seed
        torch.manual_seed(args.random_seed)  # initialize torch random seed
        np.random.seed(args.random_seed)  # initialize numpy random seed
        if args.device_id != -1:  # initialize cuda random seed
            torch.cuda.manual_seed_all(args.random_seed)
            torch.backends.cudnn.deterministic = True

        torch.set_printoptions(precision=9)  # Number of digits of precision for floating point output (default = 4)
        torch.set_num_threads(1)  # Sets the number of threads used for parallelizing operations.

        embedding, index2word, word2index, vocab_size, embed_dim = load_embedding(options["w2vPathe"], options[
            "w2vName"])  # need some times,  obtain the embeddings

        options["embedding"] = embedding
        options["index2word"] = index2word
        options["word2index"] = word2index
        options["vocabSize"] = vocab_size
        options["embed_dim"] = embed_dim
        print("Finish loading embedding: embed_dim = {}, vocab_size = {}".format(embed_dim, vocab_size))

        BiContextWordDict, BiContextIndexDict, contexArray = load_biContext(options["BiContextPath"],
                                                                            options["word2index"], options[
                                                                                "index2word"])  # obtain the bicontext for the entity
        print("Finish loading bicontext.")
        options["BiContextWordDict"] = BiContextWordDict
        options["BiContextIndexDict"] = BiContextIndexDict
        options["contexArray"] = torch.from_numpy(contexArray)

        raw_training_dataSets = loda_dataset(options["trainingSetPath"])  # obtain train_set
        random.shuffle(raw_training_dataSets)  # shuffle the raw_dataSet_list
        training_set = entity_set.EntitySet("training_set", options["data_format"], options,
                                            raw_training_dataSets)  # obtain the training_set

        raw_testing_dataSets = loda_dataset(options["testingSetPath"])  # obtain train_set
        random.shuffle(raw_testing_dataSets)  # shuffle the raw_dataSet_list
        testing_set = entity_set.EntitySet("testing_set", options["data_format"], options,
                                           raw_testing_dataSets)  # obtain the testing_set
        options["training_set"] = training_set
        options["testing_set"] = testing_set


        mu_list = options["mu"]
        delta_list = options["delta"]
        kappa_list = options["kappa"]
        lamb_list = options["lamb"]

        for mm in mu_list:
            for dd in delta_list:
                for kk in kappa_list:
                    for ll in lamb_list:
                        options["mu"] = mm
                        options["delta"] = dd
                        options["kappa"] = kk
                        options["lamb"] = ll

                        mkdir("runs\\testing_log_" + options["dataset"])
                        logger = my_logger(name='testing_logger_mu' + str(options["mu"]) + "_delta" + str(
                            options["delta"]) + "_kappa" + str(options["kappa"]) + "_lamb" + str(options["lamb"]) + "_",
                                           log_path="runs\\testing_log_" + options["dataset"])
                        logger.setLevel(0)
                        runing(options, options["training_set"], options["testing_set"], mode="test", my_logger=logger) # start
                        logging.shutdown()







