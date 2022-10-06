import argparse
import os
import torch
from tools import print_args


def load_options(datasetName):
    '''
    ***
    obtain the filenames for dataset  (BaiDu or SoGouCA)
    :param datasetName:
    :return:
    '''
    optionsSubDict = {}
    if datasetName == "BaiDu":
        optionsSubDict["DataSet"] = "BaiDu"
        optionsSubDict["W2vName"] = "BaiDuEmbedd.emb"
        optionsSubDict["BiContext"] = "BaiDuBiContext.txt"
        optionsSubDict["Entity"] = "BaiDuEntity.txt"
        optionsSubDict["Syn"] = "BaiDuSyn.txt"
        optionsSubDict["Train"] = "BaiDuTrain.txt"
        optionsSubDict["Test"] = "BaiDuTest.txt"
    elif datasetName == "SoGouCA":
        optionsSubDict["DataSet"] = "SoGouCA"
        optionsSubDict["W2vName"] = "SoGouCAEmbedd.emb"
        optionsSubDict["BiContext"] = "SoGouCABiContext.txt"
        optionsSubDict["Entity"] = "SoGouCAEntity.txt"
        optionsSubDict["Syn"] = "SoGouCASyn.txt"
        optionsSubDict["Train"] = "SoGouCATrain.txt"
        optionsSubDict["Test"] = "SoGouCATest.txt"

    return optionsSubDict



def read_configs(options, optionsList):
    '''
    ***
     set all initial settings for project
    :param options:
    :param optionsList:
    :return: args
    '''

    parser = argparse.ArgumentParser(description='Chinese Entity synonym set expansion algorithm')
    parser.add_argument('-dataset', default=options["DataSet"], type=str, help='dataset name')


    if optionsList["dataset"] == "BaiDu":
        # Data parameters
        parser.add_argument('-w2vPathe', default=".\\DataSets\\" + options["DataSet"] + "\\" + options["W2vName"],
                            type=str, help='path of the w2v')
        parser.add_argument('-w2vName', default="word2vec", type=str, help='name of the w2v')
        parser.add_argument('-trainingSetPath', default=".\\DataSets\\" + options["DataSet"] + "\\" + options["Train"],
                            type=str, help='training set path')
        parser.add_argument('-testingSetPath', default=".\\DataSets\\" + options["DataSet"] + "\\" + options["Test"],
                            type=str, help='testing set path')
        parser.add_argument('-BiContextPath',
                            default=".\\DataSets\\" + options["DataSet"] + "\\" + options["BiContext"], type=str,
                            help='BiContext path')
        parser.add_argument('-EntityPath', default=".\\DataSets\\" + options["DataSet"] + "\\" + options["Entity"],
                            type=str, help='Entity path')
        parser.add_argument('-SynPath', default=".\\DataSets\\" + options["DataSet"] + "\\" + options["Syn"], type=str,
                            help='Syn path')

        parser.add_argument('-mode', default=optionsList["mode"], type=str, choices=['train', 'eval', 'test'],
                            help='specify model running mode, \'train\' for model training, \'eval\' for model evaluation,'
                                 'and \'test\' for model testing. ' )

        # Model parameters
        parser.add_argument('-early-stop', default=200, type=int, help='early stop epoch number')
        parser.add_argument('-eval-epoch-step', default=10, type=int, help='average number of epochs for evaluation')
        parser.add_argument('-data-format', default="set", type=str, choices=['set'],
                            help='format of input training dataset [Default: set]')
        parser.add_argument('-modelName', default='np_lrlr_sd_lrlrdl', type=str,
                            help='which prediction model is used')
        parser.add_argument('-pretrained-embedding', default='embed', type=str,
                            choices=['none', 'embed', 'tfidf', 'fastText-no-subword.embed',
                                     'fastText-with-subword.embed'],
                            help='whether to use pretrained embedding, none means training embedding from scratch')
        parser.add_argument('-embed-fine-tune', default=0, type=int,
                            help='fine tune word embedding or not, 0 means no fine tune')
        parser.add_argument('-embedSize', default=100, type=int, help='element embed size')
        parser.add_argument('-node-hiddenSize', default=250, type=int, help='hidden size used in node post_embedder')
        parser.add_argument('-combine-hiddenSize', default=500, type=int, help='hidden size used in combiner')
        parser.add_argument('-max-set-size', default=20, type=int, help='maximum size for training batch')
        # Learning options
        parser.add_argument('-batch-size', default=128, type=int, help='batch size for training')

        if optionsList["mode"] == "train":
            parser.add_argument('-mu', default=eval(optionsList["mu"]), type=list,
                                help='hyper parameter for contex and embedding')
            parser.add_argument('-delta', default=eval(optionsList["delta"]), type=int, help='hyper parameter for cos and KL')
            parser.add_argument('-kappa', default=eval(optionsList["kappa"]), type=int,
                                help='hyper parameter for threshold κ ')
            parser.add_argument('-lamb', default=eval(optionsList["lamb"]), type=int, help='hyper parameter for threshold λ')
        elif optionsList["mode"] == "eval_opt":
            parser.add_argument('-mu', default=eval(optionsList["mu"]), type=list,
                                help='hyper parameter for contex and embedding')
            parser.add_argument('-delta', default=eval(optionsList["delta"]), type=list, help='hyper parameter for cos and KL')
            parser.add_argument('-kappa', default=eval(optionsList["kappa"]), type=list,
                                help='hyper parameter for threshold κ ')
            parser.add_argument('-lamb', default=eval(optionsList["lamb"]), type=list, help='hyper parameter for threshold λ')
            parser.add_argument('-processNum', default=optionsList["processNum"], type=int, help='hyper parameter for threshold λ')
        elif optionsList["mode"] == "test":
            parser.add_argument('-mu', default=eval(optionsList["mu"]), type=int,
                                help='hyper parameter for contex and embedding')
            parser.add_argument('-delta', default=eval(optionsList["delta"]), type=int, help='hyper parameter for cos and KL')
            parser.add_argument('-kappa', default=eval(optionsList["kappa"]), type=int,
                                help='hyper parameter for threshold κ ')
            parser.add_argument('-lamb', default=eval(optionsList["lamb"]), type=int, help='hyper parameter for threshold λ')

        parser.add_argument('-lr', default=0.00005, type=float, help='initial learning rate')
        parser.add_argument('-loss-fn', default="self_margin_rank_bce", type=str,
                            choices=['cross_entropy', 'max_margin', 'margin_rank', 'self_margin_rank',
                                     'self_margin_rank_bce'], help='loss function used in training model')
        parser.add_argument('-margin', default=0.5, type=float,
                            help='margin used in max_margin loss and margin_rank loss')
        parser.add_argument('-epochs', default=1000, type=int, help='number of epochs for training')
        parser.add_argument('-neg-sample-size', default=50, type=int,
                            help='number of negative samples generated for each set')
        parser.add_argument('-neg-sample-method', default="complete_random", type=str,
                            choices=["complete_random"], help='negative sampling method')

        # Regularization parameters
        parser.add_argument('-dropout', default=0.4, type=float, help='Dropout between layers')
        parser.add_argument('-random-seed', default=111111, type=int,
                            help='random seed used for model initialization and negative sample generation')
        parser.add_argument('-size-opt-clus', default=0, type=int,
                            help='whether conduct size optimized clustering prediction'
                                 'this saves GPU memory sizes but consumes more training time)'
                                 'when expecting a large number of small sets, set this option to be 0;'
                                 'when expecting a small number of huge sets, set this option to be 1')
        parser.add_argument('-max-K', default=-1, type=int, help='maximum cluster number, -1 means auto-infer')
        parser.add_argument('-T', default=1.0, type=int, help='temperature scaling, 1.0 means no scaling')

        # Device options
        parser.add_argument('-device-id', default=0, type=int, help='device to use for iterate data, -1 means cpu')

        # Model saving/loading options
        parser.add_argument('-save-dir', default="./snapshots/", type=str, help='location to save models')
        parser.add_argument('-best-save-dir', default="./snapshots_trianed_best/BaiDu/", type=str,
                            help='location to save best models')
        parser.add_argument('-load-model', default="", type=str, help='path to loaded model')
        parser.add_argument('-snapshot', default="", type=str, help='path to model snapshot')
        parser.add_argument('-tune-result-file', default="tune_prefix", type=str,
                            help='path to save all tuning results')


    #**************************************************************************************************
    elif optionsList["dataset"] == "SoGouCA":
        # Data parameters

        parser.add_argument('-w2vPathe', default=".\\DataSets\\" + options["DataSet"] + "\\" + options["W2vName"],
                            type=str, help='path of the w2v')
        parser.add_argument('-w2vName', default="word2vec", type=str, help='name of the w2v')
        parser.add_argument('-trainingSetPath', default=".\\DataSets\\" + options["DataSet"] + "\\" + options["Train"],
                            type=str, help='training set path')
        parser.add_argument('-testingSetPath', default=".\\DataSets\\" + options["DataSet"] + "\\" + options["Test"],
                            type=str, help='testing set path')
        parser.add_argument('-BiContextPath',
                            default=".\\DataSets\\" + options["DataSet"] + "\\" + options["BiContext"], type=str,
                            help='BiContext path')
        parser.add_argument('-EntityPath', default=".\\DataSets\\" + options["DataSet"] + "\\" + options["Entity"],
                            type=str, help='Entity path')
        parser.add_argument('-SynPath', default=".\\DataSets\\" + options["DataSet"] + "\\" + options["Syn"], type=str,
                            help='Syn path')

        parser.add_argument('-mode', default=optionsList["mode"], type=str, choices=['train', 'eval', 'test'],
                            help='specify model running mode, \'train\' for model training, \'eval\' for model evaluation,'
                                 'and \'test\' for model testing. '
                                 )

        # Model parameters
        parser.add_argument('-early-stop', default=200, type=int, help='early stop epoch number')
        parser.add_argument('-eval-epoch-step', default=10, type=int, help='average number of epochs for evaluation')
        parser.add_argument('-data-format', default="set", type=str, choices=['set'],
                            help='format of input training dataset [Default: set]')
        parser.add_argument('-modelName', default='np_lrlr_sd_lrlrdl', type=str,
                            help='which prediction model is used')
        parser.add_argument('-pretrained-embedding', default='embed', type=str,
                            choices=['none', 'embed'],
                            help='whether to use pretrained embedding, none means training embedding from scratch')
        parser.add_argument('-embed-fine-tune', default=0, type=int,
                            help='fine tune word embedding or not, 0 means no fine tune')
        parser.add_argument('-embedSize', default=100, type=int, help='element embed size')
        parser.add_argument('-node-hiddenSize', default=250, type=int, help='hidden size used in node post_embedder')
        parser.add_argument('-combine-hiddenSize', default=500, type=int, help='hidden size used in combiner')
        parser.add_argument('-max-set-size', default=20, type=int, help='maximum size for training batch')

        # Learning options
        parser.add_argument('-batch-size', default=64, type=int, help='batch size for training')
        if optionsList["mode"] == "train":
            parser.add_argument('-mu', default=eval(optionsList["mu"]), type=list,
                                help='hyper parameter for contex and embedding')
            parser.add_argument('-delta', default=eval(optionsList["delta"]), type=int, help='hyper parameter for cos and KL')
            parser.add_argument('-kappa', default=eval(optionsList["kappa"]), type=int,
                                help='hyper parameter for threshold κ ')
            parser.add_argument('-lamb', default=eval(optionsList["lamb"]), type=int, help='hyper parameter for threshold λ')
        elif optionsList["mode"] == "eval_opt":
            parser.add_argument('-mu', default=eval(optionsList["mu"]), type=list,
                                help='hyper parameter for contex and embedding')
            parser.add_argument('-delta', default=eval(optionsList["delta"]), type=list, help='hyper parameter for cos and KL')
            parser.add_argument('-kappa', default=eval(optionsList["kappa"]), type=list,
                                help='hyper parameter for threshold κ ')
            parser.add_argument('-lamb', default=eval(optionsList["lamb"]), type=list, help='hyper parameter for threshold λ')
            parser.add_argument('-processNum', default=optionsList["processNum"], type=int, help='hyper parameter for threshold λ')
        elif optionsList["mode"] == "test":
            parser.add_argument('-mu', default=eval(optionsList["mu"]), type=list,
                                help='hyper parameter for contex and embedding')
            parser.add_argument('-delta', default=eval(optionsList["delta"]), type=list, help='hyper parameter for cos and KL')
            parser.add_argument('-kappa', default=eval(optionsList["kappa"]), type=list,
                                help='hyper parameter for threshold κ ')
            parser.add_argument('-lamb', default=eval(optionsList["lamb"]), type=list, help='hyper parameter for threshold λ')
        parser.add_argument('-lr', default=0.00005, type=float, help='initial learning rate')
        parser.add_argument('-loss-fn', default="self_margin_rank_bce", type=str,
                            choices=['cross_entropy', 'max_margin', 'margin_rank', 'self_margin_rank',
                                     'self_margin_rank_bce'], help='loss function used in training model')
        parser.add_argument('-margin', default=0.5, type=float,
                            help='margin used in max_margin loss and margin_rank loss')
        parser.add_argument('-epochs', default=1000, type=int, help='number of epochs for training')
        parser.add_argument('-neg-sample-size', default=50, type=int,
                            help='number of negative samples generated for each set')
        parser.add_argument('-neg-sample-method', default="complete_random", type=str,
                            choices=["complete_random"], help='negative sampling method')

        # Regularization parameters
        parser.add_argument('-dropout', default=0.4, type=float, help='Dropout between layers')
        parser.add_argument('-random-seed', default=111111, type=int,
                            help='random seed used for model initialization and negative sample generation')
        parser.add_argument('-size-opt-clus', default=0, type=int,
                            help='whether conduct size optimized clustering prediction'
                                 'this saves GPU memory sizes but consumes more training time)'
                                 'when expecting a large number of small sets, set this option to be 0;'
                                 'when expecting a small number of huge sets, set this option to be 1')
        parser.add_argument('-max-K', default=-1, type=int, help='maximum cluster number, -1 means auto-infer')
        parser.add_argument('-T', default=1.0, type=int, help='temperature scaling, 1.0 means no scaling')

        # Device options
        parser.add_argument('-device-id', default=0, type=int, help='device to use for iterate data, -1 means cpu')

        # Model saving/loading options
        parser.add_argument('-save-dir', default="./snapshots/", type=str, help='location to save models')
        parser.add_argument('-best-save-dir', default="./snapshots_trianed_best/SoGouCA/", type=str,
                            help='location to save best models')
        parser.add_argument('-load-model', default="", type=str, help='path to loaded model')
        parser.add_argument('-snapshot', default="", type=str, help='path to model snapshot')
        parser.add_argument('-tune-result-file', default="tune_prefix", type=str,
                            help='path to save all tuning results')


    try:
        args = parser.parse_args()
        parser.add_argument('-remark', default=options["DataSet"] + "_mu" + str(args.mu) + "_delta" + str(
            args.delta) + "_kappa" + str(args.kappa) + "_lamb" + str(args.lamb), help='reminder of this run')
        args = parser.parse_args()
        print_args(args)#打印config
    except:
        parser.error("Unable to parse arguments")

    # Update Device information
    if args.device_id == -1:
        args.device = torch.device("cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
        args.device = torch.device("cuda:0")

    if args.max_K == -1:
        args.max_K = None

    args.size_opt_clus = (args.size_opt_clus == 1)

    return args
