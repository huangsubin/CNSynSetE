
import torch
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
import itertools
import networkx as nx


def calculate_precision_recall_f1(tp, fp, fn):
    '''
    obtain precision, recall, and f1 score
    :param tp:
    :param fp:
    :param fn:
    :return:
    '''

    if (tp + fp) == 0:
        precision = 0.0
    else:
        precision = 1.0 * tp / (tp + fp)

    if (tp + fn) == 0:
        recall = 0.0
    else:
        recall = 1.0 * tp / (tp + fn)

    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)

    return precision, recall, f1





def end2end_evaluation_matching(groundtruth, result):
    '''
    obtain the maximum weighted jaccard matching of groundtruth clustering and predicted clustering

    :param groundtruth: a list of element lists representing the ground truth clustering
    :type groundtruth: list
    :param result: a list of element lists representing the model predicted clustering
    :type result: list
    :return: best matching score
    :rtype: float
    '''

    n = len(groundtruth)
    m = len(result)
    G = nx.DiGraph()
    S = n + m
    T = n + m + 1
    C = 1e8
    for i in range(n):
        for j in range(m):
            s1 = groundtruth[i]
            s2 = result[j]
            s12 = set(s1) & set(s2)
            weight = len(s12) / (len(s1) + len(s2) - len(s12))
            weight = int(weight * C)
            if weight > 0:
                G.add_edge(i, n + j, capacity=1, weight=-weight)
    for i in range(n):
        G.add_edge(S, i, capacity=1, weight=0)
    for i in range(m):
        G.add_edge(i + n, T, capacity=1, weight=0)
    mincostFlow = nx.algorithms.max_flow_min_cost(G, S, T)
    mincost = nx.cost_of_flow(G, mincostFlow) / C
    return -mincost / m


def evaluate_set_instance_prediction(model, dataset):
    '''
    evaluate model on the given dataset for set-instance pair prediction task

    :param model:
    :param dataset:
    :return:
    '''
    model.eval()

    y_true = []
    y_pred = []
    set_size = []
    score_pred = []


    # the following max_set_size and batch_size number need to be set such that one test batch can fit GPU memory

    max_set_size = 20 # TODO: make this value dynamtically changeable

    batch_size = int(len(dataset.sip_triplets) / 2)
    for test_batch in dataset.get_test_batch(max_set_size=max_set_size, batch_size=batch_size):
        batch_set_size = torch.sum((test_batch['set'] != 0), dim=1)
        if model.device_id != -1:
            batch_set_size = batch_set_size.to(torch.device("cpu"))
        batch_set_size = list(batch_set_size.numpy())
        set_size += batch_set_size

        # start real prediction
        ######################Embedding feature extractor#############################################
        mask = (test_batch['set'] != 0).float().unsqueeze(-1)
        entity_embedder_set = model.entity_embedder(test_batch['set'].long())
        setEmbed = model.Q1Layer(entity_embedder_set) * mask
        setEmbed = model.Entity_pooler(setEmbed, dim=1)
        setQ2Layer = model.Q2Layer(setEmbed)  # (batch_size， 1)

        entity_embedder_inst = model.entity_embedder(test_batch['inst'].long())
        instEmbed = model.Q1Layer(entity_embedder_inst).squeeze_(1)
        setInstSumQ2Layer = model.Q2Layer(setEmbed + instEmbed)
        ######################Embedding feature extractor#############################################

        ######################Bilateral-context-based feature extractor with context-level attention mechanism#############################################
        entity_embedder_set_t = model.entity_embedder(test_batch['set'].long())
        entity_embedder_set_t = entity_embedder_set_t.unsqueeze(2)
        SetExpen_t = entity_embedder_set_t.expand(test_batch['set'].shape[0], test_batch['set'].shape[1], 10, 100)
        SetExpen_t = SetExpen_t.permute(0, 1, 3, 2)
        train_batch_set = test_batch['set']
        train_set_context = model.params["contexArray"][train_batch_set.long()]
        train_set_context = train_set_context.cuda()
        train_set_context_embedd = model.entity_embedder(train_set_context.long())
        alpha_weight = torch.matmul(train_set_context_embedd, SetExpen_t)
        alpha_weight_sum = alpha_weight.sum(dim=2)
        alpha_weight_t = alpha_weight_sum.unsqueeze(2)
        alpha_weight_t = alpha_weight_t.expand(test_batch['set'].shape[0], test_batch['set'].shape[1], 10, 10)
        alpha_weight_finall = alpha_weight / alpha_weight_t
        alpha_weight_finall = alpha_weight_finall.repeat(1, 1, 1, 10)
        train_set_context_embedd = torch.mul(train_set_context_embedd, alpha_weight_finall)
        train_set_context_embedd = train_set_context_embedd.sum(dim=2)
        train_set_ = torch.zeros(
            [train_set_context_embedd.shape[0], train_set_context_embedd.shape[1], train_set_context_embedd.shape[2]],
            dtype=torch.float32,
            device=model.params["device"])
        train_set_context_embedd = torch.where(torch.isnan(train_set_context_embedd), train_set_,
                                               train_set_context_embedd)
        contextEmbd = model.Q1HatLayer(train_set_context_embedd) * mask
        contextEmbd = model.Entity_pooler(contextEmbd, dim=1)  # (batch_size, node_hiddenSize)
        contextQ2HatLayer = model.Q2HatLayer(contextEmbd)  # (batch_size， 1)



        entity_embedder_inst_t = model.entity_embedder(test_batch['inst'].long())
        entity_embedder_inst_t = entity_embedder_inst_t.unsqueeze(2)
        instExpen_t = entity_embedder_inst_t.expand(test_batch['inst'].shape[0], test_batch['inst'].shape[1], 10, 100)
        instExpen_t = instExpen_t.permute(0, 1, 3, 2)
        train_batch_inst = test_batch['inst']
        train_inst_context = model.params["contexArray"][train_batch_inst.long()]
        train_inst_context = train_inst_context.cuda()
        train_inst_context_embedd = model.entity_embedder(train_inst_context.long())
        alpha_inst_weight = torch.matmul(train_inst_context_embedd, instExpen_t)
        alpha_inst_weight_sum = alpha_inst_weight.sum(dim=2)
        alpha_inst_weight_t = alpha_inst_weight_sum.unsqueeze(2)
        alpha_inst_weight_t = alpha_inst_weight_t.expand(test_batch['inst'].shape[0], test_batch['inst'].shape[1], 10,
                                                         10)
        alpha_inst_weight_finall = alpha_inst_weight / alpha_inst_weight_t
        alpha_inst_weight_finall = alpha_inst_weight_finall.repeat(1, 1, 1, 10)
        train_inst_context_embedd = torch.mul(train_inst_context_embedd, alpha_inst_weight_finall)
        train_inst_context_embedd = train_inst_context_embedd.sum(dim=2)
        context_inst_Embd = model.Q1HatLayer(train_inst_context_embedd).squeeze(1)
        context_inst_Q2HatLayer = model.Q2HatLayer(contextEmbd + context_inst_Embd)  # (batch_size， 1)
        ######################Bilateral-context-based feature extractor with context-level attention mechanism#############################################


        setQ2LayerFinally =(1 - model.params["mu"]) * setQ2Layer + model.params["mu"] * contextQ2HatLayer  # combine embedding feature extractor bilateral-context-based feature extractor
        setInstSumQ2LayerFinally  = (1 - model.params["mu"]) * setInstSumQ2Layer + model.params["mu"] * context_inst_Q2HatLayer # combine embedding feature extractor bilateral-context-based feature extractor


        score_diff = setInstSumQ2LayerFinally - setQ2LayerFinally
        prediction = torch.sigmoid(score_diff)
        if model.device_id != -1:
            prediction = prediction.to(torch.device("cpu"))
        cur_pred = (prediction > 0.5).squeeze().numpy()
        y_pred += list(cur_pred)
        score_pred += list(prediction)
        target = test_batch['label'].float()
        loss = model.criterion(score_diff, target).item()
        if model.device_id != -1:
            target = target.to(torch.device("cpu"))
        cur_true = target.squeeze().numpy()
        y_true += list(cur_true)

    # obtain set-size-wise accuracy
    set_size2num = Counter(set_size)
    set_size2correct = defaultdict(int)
    for t, p, s in zip(y_true, y_pred, set_size):
        if t == p:
            set_size2correct[s] += 1
    set_size2accuracy = {}
    for set_size in set_size2correct:
        set_size2accuracy[set_size] = set_size2correct[set_size] / set_size2num[set_size]

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    num_pred_pos = int(np.sum(y_pred))
    num_pred_neg = y_true.shape[0] - num_pred_pos
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y_true, y_pred)

    model.train()
    metrics = {"precision": precision, "recall": recall, "f1": f1, "num_pred_pos": num_pred_pos,
               "num_pred_neg": num_pred_neg, "tn": tn, "fp": fp, "fn": fn, "tp": tp, "loss": loss,
               "accuracy": accuracy}

    return metrics


def evaluate_clustering(cls_pred, cls_true):
    '''
    evaluate clustering results
    :param cls_pred:
    :param cls_true:
    :return:
    '''

    vocab_pred = set(itertools.chain(*cls_pred))
    vocab_true = set(itertools.chain(*cls_true))
    assert (vocab_pred == vocab_true), "Unmatched vocabulary during clustering evaluation"

    # Cluster number
    num_of_predict_clusters = len(cls_pred)

    # Cluster size histogram
    cluster_size2num_of_predicted_clusters = Counter([len(cluster) for cluster in cls_pred])

    # Exact cluster prediction
    pred_cluster_set = set([frozenset(cluster) for cluster in cls_pred])
    gt_cluster_set = set([frozenset(cluster) for cluster in cls_true])
    num_of_exact_set_prediction = len(pred_cluster_set.intersection(gt_cluster_set))

    # Clustering metrics
    word2rank = {}
    wordrank2gt_cluster = {}
    rank = 0
    for cid, cluster in enumerate(cls_true):
        for word in cluster:
            if word not in word2rank:
                word2rank[word] = rank
                rank += 1
            wordrank2gt_cluster[word2rank[word]] = cid
    gt_cluster_vector = [ele[1] for ele in sorted(wordrank2gt_cluster.items())]

    wordrank2pred_cluster = {}
    for cid, cluster in enumerate(cls_pred):
        for word in cluster:
            wordrank2pred_cluster[word2rank[word]] = cid
    pred_cluster_vector = [ele[1] for ele in sorted(wordrank2pred_cluster.items())]

    ARI = adjusted_rand_score(gt_cluster_vector, pred_cluster_vector)
    FMI = fowlkes_mallows_score(gt_cluster_vector, pred_cluster_vector)
    NMI = normalized_mutual_info_score(gt_cluster_vector, pred_cluster_vector,average_method="arithmetic")

    # Pair-based clustering metrics
    def pair_set(labels):
        S = set()
        cluster_ids = np.unique(labels)
        for cluster_id in cluster_ids:
            cluster = np.where(labels == cluster_id)[0]
            n = len(cluster)  # number of elements in this cluster
            if n >= 2:
                for i in range(n):
                    for j in range(i + 1, n):
                        S.add((cluster[i], cluster[j]))
        return S

    F_S = pair_set(gt_cluster_vector)
    F_K = pair_set(pred_cluster_vector)
    if len(F_K) == 0:
        pair_recall = 0
        pair_precision = 0
        pair_f1 = 0
    else:
        common_pairs = len(F_K & F_S)
        pair_recall = common_pairs / len(F_S)
        pair_precision = common_pairs / len(F_K)
        eps = 1e-6
        pair_f1 = 2 * pair_precision * pair_recall / (pair_precision + pair_recall + eps)

    # KM matching
    mwm_jaccard = end2end_evaluation_matching(cls_true, cls_pred)

    metrics = {"ARI": ARI, "FMI": FMI, "NMI": NMI, "pair_recall": pair_recall, "pair_precision": pair_precision,
               "pair_f1": pair_f1, "predicted_clusters": cls_pred, "num_of_predicted_clusters": num_of_predict_clusters,
               "cluster_size2num_of_predicted_clusters": cluster_size2num_of_predicted_clusters,
               "num_of_exact_set_prediction": num_of_exact_set_prediction,
               "maximum_weighted_match_jaccard": mwm_jaccard}

    return metrics
