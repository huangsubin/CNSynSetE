import torch
import torch.nn as nn

import math
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from math import floor
from tools import Metrics,save_model, load_model
import evaluator as evaluator
import cluster_predict as cluster_predict


def initialize_weights(moduleList, itype="xavier"):
    '''
    init  modules weights
    :param moduleList:
    :param itype:
    :return:
    '''

    assert itype == 'xavier', 'Only Xavier initialization supported'

    for moduleId, module in enumerate(moduleList):
        if hasattr(module, '_modules') and len(module._modules) > 0:
            # Iterate again
            initialize_weights(module, itype)
        else:
            # Initialize weights
            name = type(module).__name__
            # If linear or embedding
            if name == 'Embedding' or name == 'Linear':
                fanIn = module.weight.data.size(0)
                fanOut = module.weight.data.size(1)

                factor = math.sqrt(2.0/(fanIn + fanOut))
                weight = torch.randn(fanIn, fanOut) * factor
                module.weight.data.copy_(weight)

            # Check for bias and reset
            if hasattr(module, 'bias') and hasattr(module.bias, 'data'):
                module.bias.data.fill_(0.0)




class CNSYN(nn.Module):


    def __init__(self, params):
        '''
        init
        :param params:
        '''
        super(CNSYN, self).__init__()

        self.initialize(params)

        if params['loss_fn'] == "cross_entropy":
            self.criterion = nn.NLLLoss()
        elif params['loss_fn'] == "max_margin":
            self.criterion = nn.MultiMarginLoss(margin=params['margin'])
        elif params['loss_fn'] in ["margin_rank", "self_margin_rank"]:
            self.criterion = nn.MarginRankingLoss(margin=params['margin'])
        elif params['loss_fn'] == "self_margin_rank_bce":
            self.criterion = nn.BCEWithLogitsLoss()

        # TODO: avoid the following self.params = params
        self.params = params
        # transfer parameters to self, therefore we have self.modelName
        for key, val in self.params.items():
            setattr(self, key, val)

        self.temperature = params["T"]  # use for temperature scaling





    def initialize(self, params):
        '''
        initialize the NN layers
        :param params:
        :return:
        '''
        self.entity_embedder = nn.Embedding(params['vocabSize'],
                                     params['embedSize'])  # in paper, referred as "Embedding Layer"


        self.Q1Layer = nn.Sequential(
            nn.Linear(params['embedSize'], params['embedSize'], bias=False),
            nn.ReLU(),
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.ReLU()
        )

        self.Q2Layer = nn.Sequential(
            nn.Linear(params['node_hiddenSize'], params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize'] / 2)),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(floor(params['combine_hiddenSize'] / 2), 1),
        )

        self.Q1HatLayer = nn.Sequential(
            nn.Linear(params['embedSize'], params['embedSize'], bias=False),
            nn.ReLU(),
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.ReLU()
        )

        self.Q2HatLayer = nn.Sequential(
            nn.Linear(params['node_hiddenSize'], params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize'] / 2)),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(floor(params['combine_hiddenSize'] / 2), 1),
        )

        self.Entity_pooler = torch.sum  # in the paper, this is fixed to be "sum", but you can replace it with mean/max/min function

        self.ContextR = nn.Sequential(
            nn.Linear(params['embedSize'], params['embedSize'], bias=False),
        )


        modules = ['entity_embedder', 'Q1Layer', 'Q2Layer',"ContextR"]


        modules = [getattr(self, mod) for mod in modules if hasattr(self, mod)]
        initialize_weights(modules, 'xavier')

        if params['pretrained_embedding'] != "none":
            pretrained_embedding = params['embedding'].vectors
            self.entity_embedder.weight.data.copy_(torch.from_numpy(pretrained_embedding))
            if params['embed_fine_tune'] == 0:  # fix embedding without fine-tune
                self.entity_embedder.weight.requires_grad = False



    def _set_Q2Layer(self, set_tensor):
        '''
         return the quality score of a batch of sets:embedding feature extractor
        :param set_tensor:
        :return:
        '''
        # Element encoding
        mask = (set_tensor != 0).float().unsqueeze_(-1)  # (batch_size, max_set_size, 1)
        entity_embedder_set = self.entity_embedder(set_tensor.long())
        setEmbed = self.Q1Layer(entity_embedder_set) * mask
        setEmbed = self.Entity_pooler(setEmbed, dim=1)  # (batch_size, node_hiddenSize)
        setQ2Layer = self.Q2Layer(setEmbed)  # (batch_size, 1)

        return setQ2Layer

    def _set_Q2HatLayer(self, set_tensor):
        '''
        return the quality score of a batch of sets : bilateral-context-based feature extractor
        :param set_tensor:
        :return:
        '''

        entity_embedder_set_t = self.entity_embedder(set_tensor.long())
        entity_embedder_set_t = entity_embedder_set_t.unsqueeze(2)
        setExpen = entity_embedder_set_t.expand(set_tensor.shape[0], set_tensor.shape[1], 10, 100)
        setExpen = setExpen.permute(0, 1, 3, 2)
        train_batch_set = set_tensor
        train_set_context = self.params["contexArray"][train_batch_set.long()]
        train_set_context = train_set_context.cuda()
        train_set_context_embedd = self.entity_embedder(train_set_context.long())
        alpha_weight = torch.matmul(train_set_context_embedd, setExpen)
        alpha_weight_sum = alpha_weight.sum(dim=2)
        alpha_weight_t = alpha_weight_sum.unsqueeze(2)
        alpha_weight_t = alpha_weight_t.expand(set_tensor.shape[0], set_tensor.shape[1], 10, 10)
        alpha_weight_finall = alpha_weight / alpha_weight_t
        alpha_weight_finall = alpha_weight_finall.repeat(1, 1, 1, 10)
        train_set_context_embedd = torch.mul(train_set_context_embedd, alpha_weight_finall)
        train_set_context_embedd = train_set_context_embedd.sum(dim=2)
        train_set_ = torch.zeros(
            [train_set_context_embedd.shape[0], train_set_context_embedd.shape[1], train_set_context_embedd.shape[2]],
            dtype=torch.float32,
            device=self.params["device"])
        train_set_context_embedd = torch.where(torch.isnan(train_set_context_embedd), train_set_,
                                               train_set_context_embedd)
        mask = (set_tensor != 0).float().unsqueeze_(-1)
        contextEmbd = self.Q1HatLayer(train_set_context_embedd) * mask
        contextEmbd = self.Entity_pooler(contextEmbd, dim=1)  # (batch_size, node_hiddenSize)
        contextQ2HatLayer = self.Q2HatLayer(contextEmbd)  # (batch_size， 1)
        return contextQ2HatLayer


    def forward(self, train_batch):
        '''
        Train the model on the given train_batch

        :param train_batch: a dictionary containing training batch in <set, instance> pair format
        :type train_batch: dict
        :return: batch_loss, true_positive_num, false_positive_num, false_negative_num, true_positive_num
        :rtype: tuple
        :param train_batch:
        :return:
        '''
        ######################Embedding feature extractor#############################################
        mask = (train_batch['set'] != 0).float().unsqueeze_(-1)
        entity_embedder_set = self.entity_embedder(train_batch['set'].long())
        setEmbed = self.Q1Layer(entity_embedder_set) * mask
        setEmbed = self.Entity_pooler(setEmbed, dim=1)  # (batch_size, node_hiddenSize)
        setQ2Layer = self.Q2Layer(setEmbed)  # (batch_size， 1)


        entity_embedder_inst = self.entity_embedder(train_batch['inst'].long())
        instEmbed = self.Q1Layer(entity_embedder_inst).squeeze(1)  # (batch_size, node_hiddenSize)
        setInstSumQ2Layer = self.Q2Layer(setEmbed + instEmbed)  # (batch_size, 1)
        ######################Embedding feature extractor#############################################



        ######################Bilateral-context-based feature extractor with context-level attention mechanism#############################################
        entity_embedder_set_t = self.entity_embedder(train_batch['set'].long())  #
        entity_embedder_set_t = entity_embedder_set_t.unsqueeze(2)
        SetExpen_t = entity_embedder_set_t.expand(train_batch['set'].shape[0], train_batch['set'].shape[1], 10, 100)
        SetExpen_t = SetExpen_t.permute(0, 1, 3, 2)
        train_batch_set =  train_batch['set']
        train_set_context = self.params["contexArray"][train_batch_set.long()]
        train_set_context = train_set_context.cuda()
        train_set_context_embedd = self.entity_embedder(train_set_context.long())#
        alpha_weight = torch.matmul(train_set_context_embedd,SetExpen_t)
        alpha_weight_sum = alpha_weight.sum(dim=2)
        alpha_weight_t = alpha_weight_sum.unsqueeze(2)
        alpha_weight_t = alpha_weight_t.expand(train_batch['set'].shape[0], train_batch['set'].shape[1], 10, 10)
        alpha_weight_finall = alpha_weight / alpha_weight_t
        alpha_weight_finall = alpha_weight_finall.repeat(1, 1, 1, 10)

        train_set_context_embedd = torch.mul(train_set_context_embedd,alpha_weight_finall)
        train_set_context_embedd = train_set_context_embedd.sum(dim=2)
        train_set_ = torch.zeros([train_set_context_embedd.shape[0], train_set_context_embedd.shape[1], train_set_context_embedd.shape[2]], dtype=torch.float32,
                   device=self.params["device"])
        train_set_context_embedd = torch.where(torch.isnan(train_set_context_embedd),train_set_,train_set_context_embedd)
        contextEmbd = self.Q1HatLayer(train_set_context_embedd) * mask
        contextEmbd = self.Entity_pooler(contextEmbd, dim=1)  # (batch_size, node_hiddenSize)
        contextQ2HatLayer = self.Q2HatLayer(contextEmbd)  # (batch_size， 1)


        entity_embedder_inst_t = self.entity_embedder(train_batch['inst'].long())
        entity_embedder_inst_t = entity_embedder_inst_t.unsqueeze(2)
        instExpen_t = entity_embedder_inst_t.expand(train_batch['inst'].shape[0], train_batch['inst'].shape[1], 10, 100)
        instExpen_t = instExpen_t.permute(0, 1, 3, 2)
        train_batch_inst = train_batch['inst']
        train_inst_context = self.params["contexArray"][train_batch_inst.long()]
        train_inst_context = train_inst_context.cuda()
        train_inst_context_embedd = self.entity_embedder(train_inst_context.long())
        alpha_inst_weight = torch.matmul(train_inst_context_embedd, instExpen_t)
        alpha_inst_weight_sum = alpha_inst_weight.sum(dim=2)
        alpha_inst_weight_t = alpha_inst_weight_sum.unsqueeze(2)
        alpha_inst_weight_t = alpha_inst_weight_t.expand(train_batch['inst'].shape[0], train_batch['inst'].shape[1], 10, 10)
        alpha_inst_weight_finall = alpha_inst_weight / alpha_inst_weight_t
        alpha_inst_weight_finall = alpha_inst_weight_finall.repeat(1, 1, 1, 10)
        train_inst_context_embedd = torch.mul(train_inst_context_embedd, alpha_inst_weight_finall)
        train_inst_context_embedd = train_inst_context_embedd.sum(dim=2)
        context_inst_Embd = self.Q1HatLayer(train_inst_context_embedd).squeeze(1)
        context_inst_Q2HatLayer = self.Q2HatLayer(contextEmbd + context_inst_Embd)  # (batch_size， 1)
        ######################Bilateral-context-based feature extractor with context-level attention mechanism#############################################

        setQ2LayerFinally = (1 - self.params["mu"]) * setQ2Layer + self.params["mu"] * contextQ2HatLayer  # combine embedding feature extractor bilateral-context-based feature extractor
        setInstSumQ2LayerFinally  = (1 - self.params["mu"]) * setInstSumQ2Layer + self.params["mu"] * context_inst_Q2HatLayer # # combine embedding feature extractor bilateral-context-based feature extractor

        score_diff = (setInstSumQ2LayerFinally - setQ2LayerFinally)  # (batch_size, 1)
        score_diff = score_diff.squeeze(-1)  # (batch_size, )
        score_diff /= self.temperature  # temperature scaling
        target = train_batch['label'].squeeze(-1).float()  # (batch_size, )
        loss = self.criterion(score_diff, target)
        loss.backward()

        # return additional target information of current batch, this may slow down model training
        y_true = target.cpu().numpy()
        y_pred = (score_diff > 0.0).squeeze().cpu().numpy()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return loss.item(), tn, fp, fn, tp

    def predict(self, batch_set_tensor, batch_inst_tensor):
        '''
         set instance pair prediction
        :param batch_set_tensor:
        :param batch_inst_tensor:
        :return:
        '''

        setQ2Layer = self._set_Q2Layer(batch_set_tensor.long()) # embedding feature extractor
        setInstSumQ2Layer = self._set_Q2Layer(torch.cat([batch_inst_tensor.long(), batch_set_tensor.long()], dim=1)) # embedding feature extractor

        setQ2HatLayer = self._set_Q2HatLayer(batch_set_tensor.long())#bilateral-context-based feature extractor.
        setInstSumHatQ2Layer = self._set_Q2HatLayer(torch.cat([batch_inst_tensor.long(), batch_set_tensor.long()], dim=1)) #bilateral-context-based feature extractor.


        setQ2LayerFinally = (1 - self.params["mu"]) * setQ2Layer + self.params["mu"] * setQ2HatLayer  # combine embedding feature extractor bilateral-context-based feature extractor

        setInstSumQ2LayerFinally  = (1 - self.params["mu"]) * setInstSumQ2Layer + self.params["mu"] * setInstSumHatQ2Layer# combine embedding feature extractor bilateral-context-based feature extractor
        setInstSumQ2LayerFinally /= self.temperature
        setQ2LayerFinally /= self.temperature
        prediction = torch.sigmoid(setInstSumQ2LayerFinally - setQ2LayerFinally)
        return setQ2LayerFinally, setInstSumQ2LayerFinally, prediction

    def _get_test_sip_batch_size(self, x):
        if len(x) <= 1000:
            return len(x)
        elif len(x) > 1000 and (len(x) <= 1000 * 1000):
            return len(x) / 1000
        else:
            return 1000








def runing(options, training_set, dev_set, mode="train", tb_writer=None, my_logger=None):
    CNSYN_mode = None
    CNSYN_mode = CNSYN(options)

    if mode == "train":
        CNSYN_mode = CNSYN_mode.to(options["device"])
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, CNSYN_mode.parameters()), lr=options["lr"],
                                     amsgrad=True)
        results = Metrics()

        # Training phase
        training_set._shuffle()
        train_set_size = len(training_set)
        print("train_set_size: {}".format(train_set_size))

        CNSYN_mode.train()
        early_stop_metric_name = "FMI"  # metric used for early stop
        best_early_stop_metric = 0.0
        last_best_step = 0
        save_dir_path = options["save_dir"] + "\\training_snapshot_mu" + str(options["mu"]) + "_" + \
                        options["dataset"]
        save_model(CNSYN_mode, save_dir_path, 'best', 0)  # save the initial first model
        sip_f1_temp = []
        for epoch in tqdm(range(options["epochs"])):
            loss = 0
            epoch_samples = 0
            epoch_tn = 0
            epoch_fp = 0
            epoch_fn = 0
            epoch_tp = 0
            for train_batch in training_set.get_train_batch(max_set_size=options["max_set_size"],
                                                            neg_sample_size=options["neg_sample_size"],
                                                            neg_sample_method=options["neg_sample_method"],
                                                            batch_size=options["batch_size"]):
                train_batch["data_format"] = "sip"
                optimizer.zero_grad()
                cur_loss, tn, fp, fn, tp = CNSYN_mode(train_batch)
                optimizer.step()

                loss += cur_loss
                epoch_tn += tn
                epoch_fp += fp
                epoch_fn += fn
                epoch_tp += tp
                epoch_samples += (tn + fp + fn + tp)

                epoch_precision, epoch_recall, epoch_f1 = evaluator.calculate_precision_recall_f1(tp=epoch_tp,
                                                                                                  fp=epoch_fp,
                                                                                                  fn=epoch_fn)
            epoch_accuracy = 1.0 * (epoch_tp + epoch_tn) / epoch_samples
            loss /= epoch_samples

            my_logger.info("    train/loss (per instance): {}".format(loss))
            my_logger.info("    train/precision: {}".format(epoch_precision))
            my_logger.info("    train/recall: {}".format(epoch_recall))
            my_logger.info("    train/accuracy: {}".format(epoch_accuracy))
            my_logger.info("    train/f1: {}".format(epoch_f1))
            tb_writer.add_scalar('train/loss (per instance)', loss, epoch)
            tb_writer.add_scalar('train/precision', epoch_precision, epoch)
            tb_writer.add_scalar('train/recall', epoch_recall, epoch)
            tb_writer.add_scalar('train/accuracy', epoch_accuracy, epoch)
            tb_writer.add_scalar('train/f1', epoch_f1, epoch)

            if epoch % options["eval_epoch_step"] == 0 and epoch != 0:

                metrics = evaluator.evaluate_set_instance_prediction(CNSYN_mode, dev_set) # set-instance pair prediction evaluation
                tb_writer.add_scalar('val-sip/sip-precision', metrics["precision"], epoch)
                tb_writer.add_scalar('val-sip/sip-recall', metrics["recall"], epoch)
                tb_writer.add_scalar('val-sip/sip-f1', metrics["f1"], epoch)
                tb_writer.add_scalar('val-sip/sip-loss', metrics["loss"], epoch)
                my_logger.info("    val/sip-precision: {}".format(metrics["precision"]))
                my_logger.info("    val/sip-recall: {}".format(metrics["recall"]))
                my_logger.info("    val/sip-f1: {}".format(metrics["f1"]))
                my_logger.info("    val/sip-loss: {}".format(metrics["loss"]))

                # clustering evaluation
                vocab = dev_set.vocab
                cls_pred = cluster_predict.set_expansion(CNSYN_mode, vocab, size_opt_clus=options["size_opt_clus"],
                                                          max_K=options["max_K"])
                cls_true = dev_set.positive_sets
                metrics_cls = evaluator.evaluate_clustering(cls_pred, cls_true)

                tb_writer.add_scalar('val-cluster/ARI', metrics_cls["ARI"], epoch)
                tb_writer.add_scalar('val-cluster/FMI', metrics_cls["FMI"], epoch)
                tb_writer.add_scalar('val-cluster/NMI', metrics_cls["NMI"], epoch)
                tb_writer.add_scalar('val-cluster/em', metrics_cls["num_of_exact_set_prediction"], epoch)
                tb_writer.add_scalar('val-cluster/mwm_jaccard', metrics_cls["maximum_weighted_match_jaccard"], epoch)
                tb_writer.add_scalar('val-cluster/inst_precision', metrics_cls["pair_precision"], epoch)
                tb_writer.add_scalar('val-cluster/inst_recall', metrics_cls["pair_recall"], epoch)
                tb_writer.add_scalar('val-cluster/inst_f1', metrics_cls["pair_f1"], epoch)
                tb_writer.add_scalar('val-cluster/cluster_num', metrics_cls["num_of_predicted_clusters"], epoch)
                my_logger.info("    val/ARI: {}".format(metrics_cls["ARI"]))
                my_logger.info("    val/FMI: {}".format(metrics_cls["FMI"]))
                my_logger.info("    val/NMI: {}".format(metrics_cls["NMI"]))
                my_logger.info("    val/em: {}".format(metrics_cls["num_of_exact_set_prediction"]))
                my_logger.info("    val/mwm_jaccard: {}".format(metrics_cls["maximum_weighted_match_jaccard"]))
                my_logger.info("    val/inst_precision: {}".format(metrics_cls["pair_precision"]))
                my_logger.info("    val/inst_recall: {}".format(metrics_cls["pair_recall"]))
                my_logger.info("    val/inst_f1: {}".format(metrics_cls["pair_f1"]))
                my_logger.info("    val/cluster_num: {}".format(metrics_cls["num_of_predicted_clusters"]))
                my_logger.info(
                    "    val/clus_size2num_pred_clus: {}".format(metrics_cls["cluster_size2num_of_predicted_clusters"]))
                sip_f1_temp.append(metrics["f1"])
                # Early stop based on clustering results
                if metrics_cls[early_stop_metric_name] > best_early_stop_metric:
                    best_early_stop_metric = metrics_cls[early_stop_metric_name]
                    last_best_step = epoch

                    save_model(CNSYN_mode, save_dir_path, 'best', epoch) # save modl
                my_logger.info("Max sip_f1: " + str(max(sip_f1_temp)))
                my_logger.info(str(epoch) + "-" * 80)

            if epoch - last_best_step > options["early_stop"]: # Early stop
                my_logger.info("sip_f1: " + str(sip_f1_temp))
                print("Early stop by {} steps, best {}: {}, best step: {}".format(epoch, early_stop_metric_name,
                                                                                  best_early_stop_metric,
                                                                                  last_best_step))
                break

            training_set._shuffle()
        ##################################################################################
        my_logger.info("Max inst-f1: " + str(max(sip_f1_temp)))
        my_logger.info("**********************************Final Results:********************************")
        my_logger.info("Loading model: {}/best_iter_{}.pt".format(options["save_dir"], last_best_step))
        load_model(CNSYN_mode, save_dir_path, 'best', last_best_step)
        model = CNSYN_mode.to(options["device"])
        save_model(CNSYN_mode, options["best_save_dir"], 'whole_best', last_best_step, remark=options["dataset"]+"_mu"+str(options["mu"]))

        my_logger.info("=== Set-Instance Prediction Metrics ===")
        metrics = evaluator.evaluate_set_instance_prediction(model, dev_set)
        for metric in metrics:
            my_logger.info("    {}: {}".format(metric, metrics[metric]))

        my_logger.info("=== Clustering Metrics ===")
        vocab = dev_set.vocab
        #set_expansion
        cls_pred = cluster_predict.set_expansion(model, vocab, size_opt_clus=options["size_opt_clus"],
                                                  max_K=options["max_K"])
        cls_true = dev_set.positive_sets
        metrics_cls = evaluator.evaluate_clustering(cls_pred, cls_true)
        for metric in metrics_cls:
            if not isinstance(metrics_cls[metric], list):
                my_logger.info("    {}: {}".format(metric, metrics_cls[metric]))
        my_logger.info("Max inst-f1: " + str(max(sip_f1_temp)))
        # save all metrics
        results.add("sip-f1", metrics["f1"])
        results.add("sip-precision", metrics["precision"])
        results.add("sip-recall", metrics["recall"])
        results.add("ARI", metrics_cls["ARI"])
        results.add("FMI", metrics_cls["FMI"])
        results.add("NMI", metrics_cls["NMI"])
        results.add("pred_clus_num", metrics_cls["num_of_predicted_clusters"])
        results.add("em", metrics_cls["num_of_exact_set_prediction"])
        results.add("mwm_jaccard", metrics_cls["maximum_weighted_match_jaccard"])
        results.add("inst-precision", metrics_cls["pair_precision"])
        results.add("inst-recall", metrics_cls["pair_recall"])
        results.add("inst-f1", metrics_cls["pair_f1"])
        return results

    elif mode == "eval_opt": #eval_opt processing
        #load the existed trained model using the mu
        load_model(CNSYN_mode, options["best_save_dir"],"whole_best", remark=options["dataset"]+"_mu"+str(options["mu"]))

        model = CNSYN_mode.to(options["device"])
        model.eval()

        vocab = dev_set.vocab
        cls_pred = cluster_predict.set_expansion(model, vocab, size_opt_clus=options["size_opt_clus"],
                                                  max_K=options["max_K"])
        cls_true = dev_set.positive_sets
        metrics_cls = evaluator.evaluate_clustering(cls_pred, cls_true)
        CNSYN_mode = CNSYN_mode.to(torch.device("cpu"))
        CNSYN_mode = None
        return metrics_cls["FMI"],metrics_cls["ARI"],metrics_cls["NMI"]

    elif mode == "test":
        my_logger.info("**********************************test results:********************************")
        load_model(CNSYN_mode, options["best_save_dir"], "whole_best",
                   remark=options["dataset"] + "_mu" + str(options["mu"]))
        model = CNSYN_mode.to(options["device"])
        model.eval()
        my_logger.info("test:{}_mu{}_delta{}_kappa{}_lamb{}".format(options["dataset"],options["mu"] ,options["delta"],options["kappa"],options["lamb"]))


        my_logger.info("=== Set-Instance Prediction Metrics ===")
        metrics = evaluator.evaluate_set_instance_prediction(model, dev_set)
        for metric in metrics:
            my_logger.info("    {}: {}".format(metric, metrics[metric]))

        my_logger.info("=== Clustering Metrics ===")
        vocab = dev_set.vocab
        cls_pred = cluster_predict.set_expansion(model, vocab, size_opt_clus=options["size_opt_clus"],
                                                  max_K=options["max_K"])
        cls_true = dev_set.positive_sets

        import codecs

        with codecs.open("cls_pred.txt", 'w', 'utf-8') as fp:
            fp.write(str(cls_pred) + "\n")
            fp.flush()


        with codecs.open("cls_true.txt", 'w', 'utf-8') as fp:
            fp.write(str(cls_true) + "\n")
            fp.flush()



        metrics_cls = evaluator.evaluate_clustering(cls_pred, cls_true)
        for metric in metrics_cls:
            if not isinstance(metrics_cls[metric], list):
                my_logger.info("    {}: {}".format(metric, metrics_cls[metric]))
        my_logger.info("**********************************test results:********************************")





