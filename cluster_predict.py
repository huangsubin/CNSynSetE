import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
def multiple_set_single_instance_prediction(model, sets, instance, size_optimized=False):
    '''
    learn whether or not a new single entity should be added to the existing synonym set
    apply the given model to predict the probabilities of adding that one entity into the existing synonym set
    :param model:
    :param sets:
    :param instance:
    :param size_optimized:
    :return:
    '''

    if not size_optimized:  # when there exists no single big cluster, no need for complex size optimization
        return _multiple_set_single_instance_prediction(model, sets, instance)
    else:
        if len(sets) <= 10:
            return _multiple_set_single_instance_prediction(model, sets, instance)

        set_sizes = [len(ele) for ele in sets]
        tmp = sorted(enumerate(set_sizes), key=lambda x: x[1])  # (old index, set_size)
        n2o = {n: ele[0] for n, ele in enumerate(tmp)}  # new index -> old index
        o2n = {n2o[n]: n for n in n2o}  # old index -> new index
        sorted_set_sizes = [ele[1] for ele in tmp]

        # the bining method is a combination of 'sturges' and 'fd' estimators, another choice is set "bins="sturges", which generates more bins
        # c.f.: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges
        _, bin_edges = np.histogram(sorted_set_sizes, bins="auto")
        inds = np.digitize(sorted_set_sizes, bin_edges)

        sorted_setScores = []
        sorted_setInstSumScores = []
        sorted_positive_prob = []
        cur_ind = inds[0]
        cur_set = [sets[tmp[0][0]]]
        for i in range(1, len(inds)):
            if inds[i] == cur_ind:
                cur_set.append(sets[tmp[i][0]])
            else:
                cur_setScores, cur_setInstSumScores, cur_positive_prob = _multiple_set_single_instance_prediction(
                    model, cur_set, instance)
                sorted_setScores += cur_setScores
                sorted_setInstSumScores += cur_setInstSumScores
                sorted_positive_prob += cur_positive_prob
                cur_ind = inds[i]
                cur_set = [sets[tmp[i][0]]]
        if len(cur_set) > 0:  # working on the last bin
            cur_setScores, cur_setInstSumScores, cur_positive_prob = _multiple_set_single_instance_prediction(
                model, cur_set, instance)
            sorted_setScores += cur_setScores
            sorted_setInstSumScores += cur_setInstSumScores
            sorted_positive_prob += cur_positive_prob

        if len(sets) != len(sorted_positive_prob):
            assert "Mismatch after binning optimization"

        setScores = []
        setInstSumScores = []
        positive_prob = []
        for o in range(len(sets)):
            setScores.append(sorted_setScores[o2n[o]])
            setInstSumScores.append(sorted_setInstSumScores[o2n[o]])
            positive_prob.append(sorted_positive_prob[o2n[o]])

        return setScores, setInstSumScores, positive_prob


def _multiple_set_single_instance_prediction(model, sets, instance):
    '''
    learn whether or not a new single entity should be added to the existing synonym set
    apply the given model to predict the probabilities of adding that one entity into the existing synonym set
    :param model:
    :param sets:
    :param instance:
    :return:
    '''

    model.eval()

    # generate tensors
    batch_size = len(sets)
    max_set_size = max([len(ele) for ele in sets])
    batch_set_tensor = np.zeros([batch_size, max_set_size], dtype=np.int)
    for row_id, row in enumerate(sets):
        batch_set_tensor[row_id][:len(row)] = row
    batch_set_tensor = torch.from_numpy(batch_set_tensor)  # (batch_size, max_set_size)
    batch_inst_tensor = torch.tensor(instance).unsqueeze(0).expand(batch_size, 1)  # (batch_size, 1)
    batch_set_tensor = batch_set_tensor.to(model.device)
    batch_inst_tensor = batch_inst_tensor.to(model.device)

    # inference
    setScores, setInstSumScores, prediction = model.predict(batch_set_tensor, batch_inst_tensor)

    # convert to probability of each sip
    positive_prob = prediction.squeeze(-1).detach()
    positive_prob = list(positive_prob.to(torch.device("cpu")).numpy())

    setScores = setScores.squeeze(-1).detach()
    setScores = list(setScores.to(torch.device("cpu")).numpy())

    setInstSumScores = setInstSumScores.squeeze(-1).detach()
    setInstSumScores = list(setInstSumScores.to(torch.device("cpu")).numpy())

    model.train()
    return setScores, setInstSumScores, positive_prob


def multiple_set_single_instance_similar_KL(model, sets, instance, size_optimized=False):
    '''
    do similarity filtering and domain filtering
    :param model:
    :param sets:
    :param instance:
    :param size_optimized:
    :return:
    '''
    model.eval()
    batch_size = len(sets)
    max_set_size = max([len(ele) for ele in sets])
    batch_set_tensor_ori = np.zeros([batch_size, max_set_size], dtype=np.int)
    for row_id, row in enumerate(sets):
        batch_set_tensor_ori[row_id][:len(row)] = row
    batch_set_tensor = torch.from_numpy(batch_set_tensor_ori)  # (batch_size, max_set_size)
    batch_inst_tensor = torch.tensor(instance).unsqueeze(0).expand(batch_size, batch_set_tensor.shape[1])
    batch_set_context_ori = model.params["contexArray"][batch_set_tensor.long()]
    batch_inst_context_ori = model.params["contexArray"][batch_inst_tensor.long()]

    #similarity filtering-cosine_similarity
    batch_set_tensor = batch_set_tensor.to(model.device)
    batch_inst_tensor = batch_inst_tensor.to(model.device)
    mask1 = (batch_set_tensor != 0).float().unsqueeze_(-1)
    batch_set_tensor_embedding = model.entity_embedder(batch_set_tensor.long()) * mask1
    batch_inst_tensor_embedding = model.entity_embedder(batch_inst_tensor.long()) * mask1
    similar_set_inst = torch.cosine_similarity(batch_set_tensor_embedding, batch_inst_tensor_embedding, dim=-1)
    similar_avg = similar_set_inst.cpu().numpy()
    similar_avg = np.where(similar_avg != 0, similar_avg, np.nan)
    similar_avg = np.nanmean(similar_avg,axis=-1)
    #similarity filtering-cosine_similarity


    #domain filtering-Kullback-Leibler (KL)
    mask3 = (batch_set_context_ori != 0).long()
    mask4 = (batch_set_tensor_ori != 0).astype(np.long)
    batch_inst_context = batch_inst_context_ori.long() * mask3
    batch_set_context = batch_set_context_ori.float()
    batch_inst_context = batch_inst_context.float()
    logp_x1 = F.log_softmax(batch_set_context, dim=-1)
    p_y1 = F.softmax(batch_inst_context, dim=-1)
    KL1 = F.kl_div(logp_x1, p_y1, reduction='none')
    KL1 = KL1.mean(dim=-1)
    logp_x2 = F.log_softmax(batch_inst_context.float(), dim=-1)
    p_y2 = F.softmax(batch_set_context.float(), dim=-1)
    KL2 = F.kl_div(logp_x2, p_y2, reduction='none')
    KL2 = KL2.mean(dim=-1)
    KL = (KL1 + KL2) / 2.0
    KL = KL.numpy()
    mask4_sum  = np.sum(mask4,axis=-1)
    KL = np.sum(KL,axis=-1)
    KL = KL / mask4_sum
    TanhKL = 1 - np.tanh(KL)
    #domain filtering-Kullback-Leibler (KL)

    Finall_similar_list = list(similar_avg * (1-model.params["delta"]) + TanhKL *model.params["delta"]) #a linear function to combinate the similarity filtering and domain filtering
    return Finall_similar_list


def set_expansion(model, vocab, threshold_kapa=0.5, threshold_lambda = 0.5, eid2ename=None, size_opt_clus=False, max_K=None, verbose=False):
    '''
    Entity synonym set expansion algorithm with entity expansion filtering strategy


    :param model:
    :param vocab:
    :param threshold_kapa:
    :param threshold_lambda:
    :param eid2ename:
    :param size_opt_clus:
    :param max_K:
    :param verbose:
    :return:
    '''

    model.eval()
    threshold_kapa = model.params["kappa"] #  threshold
    threshold_lambda = model.params["lamb"] # threshold
    clusters = []  # will be a list of lists
    candidate_pool = vocab
    if verbose:
        print("{}\t{}".format("vocab", [eid2ename[eid] for eid in vocab]))

    if verbose:
        g = tqdm(range(len(candidate_pool)), desc="Cluster prediction (aggressive one pass)...")
    else:
        g = range(len(candidate_pool))
    for i in g:
        inst = candidate_pool[i]
        if i == 0:
            cluster = [inst]
            clusters.append(cluster)
        else:
            setScores, setInstSumScores, cluster_probs = multiple_set_single_instance_prediction(
                model, clusters, inst, size_optimized=size_opt_clus
            )
            best_matching_existing_cluster_idx = -1
            best_matching_existing_cluster_prob = 0.0

            cluster_probs_np = np.array(cluster_probs)
            max_cluster_prob = np.max(cluster_probs_np)
            if max_cluster_prob > best_matching_existing_cluster_prob:
                best_matching_existing_cluster_prob = max_cluster_prob
                best_matching_existing_cluster_idx = np.where(cluster_probs_np==max_cluster_prob)[0][0]


            #similarity filtering and domain filtering
            Finall_similar_list = multiple_set_single_instance_similar_KL(model, clusters, inst, size_optimized=size_opt_clus)
            #similarity filtering and domain filtering



            if verbose:
                print("Current Cluster Pool:",
                      [(cid, [eid2ename[ele] for ele in cluster]) for cid, cluster in enumerate(clusters)])
                print("-" * 20)
                print("Entity: {:<30}  best_prob = {:<8} Best-matching Cluster: {:<80} (cid={})".format(eid2ename[inst], best_matching_existing_cluster_prob, str(
                    [eid2ename[eid] for eid in clusters[best_matching_existing_cluster_idx]]), best_matching_existing_cluster_idx))

            if max_K and len(clusters) >= max_K:
                clusters[best_matching_existing_cluster_idx].append(inst)
                if verbose:
                    print("!!! Add Entity In")
            else:
                # similar filtering and domain filtering
                if best_matching_existing_cluster_prob > threshold_kapa and Finall_similar_list[best_matching_existing_cluster_idx] > threshold_lambda: #
                    clusters[best_matching_existing_cluster_idx].append(inst)
                    if verbose:
                        print("!!! Add Entity In")
                else:
                    new_cluster = [inst]
                    clusters.append(new_cluster)

            if verbose:
                print("-" * 120)
    if model.params["mode"] == "train":
        model.train()
    return clusters
