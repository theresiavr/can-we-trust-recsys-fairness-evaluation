# -*- encoding: utf-8 -*-
# @Time    :   2020/08/04
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com

# UPDATE
# @Time    :   2020/08/12, 2021/8/29, 2020/9/16, 2021/7/2
# @Author  :   Kaiyuan Li, Zhichao Feng, Xingyu Pan, Zihan Lin
# @email   :   tsotfsk@outlook.com, fzcbupt@gmail.com, panxy@ruc.edu.cn, zhlin@ruc.edu.cn

r"""
recbole.evaluator.metrics
############################

Suppose there is a set of :math:`n` items to be ranked. Given a user :math:`u` in the user set :math:`U`,
we use :math:`\hat R(u)` to represent a ranked list of items that a model produces, and :math:`R(u)` to
represent a ground-truth set of items that user :math:`u` has interacted with. For top-k recommendation, only
top-ranked items are important to consider. Therefore, in top-k evaluation scenarios, we truncate the
recommendation list with a length :math:`K`. Besides, in loss-based metrics, :math:`S` represents the
set of user(u)-item(i) pairs, :math:`\hat r_{u i}` represents the score predicted by the model,
:math:`{r}_{u i}` represents the ground-truth labels.

"""

from logging import getLogger

import numpy as np
from collections import Counter
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import mean_absolute_error, mean_squared_error

from recbole.evaluator.utils import _binary_clf_curve
from recbole.evaluator.base_metric import AbstractMetric, TopkMetric, LossMetric
from recbole.utils import EvaluatorType

import torch
from scipy.spatial.distance import pdist, cdist

from pytest import approx

from itertools import groupby
from scipy.sparse import dok_array, csr_matrix

import warnings
warnings.simplefilter("ignore")
# TopK Metrics




class RelMetrics(TopkMetric):
    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, pos_len = self.used_info(dataobject)

        hit, result = self.hit(pos_index)
        metric_dict = self.topk_result('HR', hit)

        mrr = self.mrr(pos_index)
        metric_dict.update(self.topk_result('MRR', mrr))

        prec = self.prec(pos_index)
        metric_dict.update(self.topk_result('P', prec))

        MAP = self.MAP(pos_index, pos_len)
        metric_dict.update(self.topk_result('MAP', MAP))

        recall = self.recall(result, pos_len)
        metric_dict.update(self.topk_result('R', recall))

        NDCG = self.ndcg(pos_index, pos_len)
        metric_dict.update(self.topk_result('NDCG', NDCG))
     
        return metric_dict

    def prec(self, pos_index):
        return pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)

    def ndcg(self, pos_index, pos_len):
        len_rank = np.full_like(pos_len, pos_index.shape[1])
        idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

        iranks = np.zeros_like(pos_index, dtype=np.float)
        iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
        for row, idx in enumerate(idcg_len):
            idcg[row, idx:] = idcg[row, idx - 1]

        ranks = np.zeros_like(pos_index, dtype=np.float)
        ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        dcg = 1.0 / np.log2(ranks + 1)
        dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

        result = dcg / idcg
        return result

    def recall(self, result, pos_len):
        return result / pos_len.reshape(-1, 1)

    def MAP(self, pos_index, pos_len):
        pre = pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)
        sum_pre = np.cumsum(pre * pos_index.astype(np.float), axis=1)
        len_rank = np.full_like(pos_len, pos_index.shape[1])
        actual_len = np.where(pos_len > len_rank, len_rank, pos_len)
        result = np.zeros_like(pos_index, dtype=np.float)
        for row, lens in enumerate(actual_len):
            ranges = np.arange(1, pos_index.shape[1] + 1)
            ranges[lens:] = ranges[lens - 1]
            result[row] = sum_pre[row] / ranges
        return result

    def hit(self, pos_index):
        result = np.cumsum(pos_index, axis=1)
        return (result > 0).astype(int), result

    def mrr(self, pos_index):
        idxs = pos_index.argmax(axis=1)
        result = np.zeros_like(pos_index, dtype=np.float)
        for row, idx in enumerate(idxs):
            if pos_index[row, idx] > 0:
                result[row, idx:] = 1 / (idx + 1)
            else:
                result[row, idx:] = 0
        return result

class FairWORel(AbstractMetric):
    metric_type = EvaluatorType.RANKING
    smaller = False
    metric_need = ['rec.items', 'data.num_items']

    def __init__(self, config):
        super().__init__(config)
        self.topk = config['topk']

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and number of items in total item set"""
        item_matrix = dataobject.get('rec.items')
        num_items = dataobject.get('data.num_items') - 1
        return item_matrix.numpy(), num_items

    def calculate_metric(self, dataobject):
        item_matrix, num_items = self.used_info(dataobject)

        metric_dict = {}
        for k in self.topk:
            item_matrix_k = item_matrix[:, :k]
            item_count = np.unique(item_matrix_k, return_counts=True)[1]
            slot = item_matrix_k.size

            floor_km_n = slot//num_items
            km_mod_n = slot%num_items

            jain_ori, jain_our = self.get_jain(item_count, slot, num_items, k, floor_km_n, km_mod_n)
            qf_ori, qf_our = self.get_QF(item_matrix_k, slot, num_items, k)
            ent_ori, ent_our = self.get_entropy(item_count,slot,num_items, k, floor_km_n, km_mod_n)
            gini_ori, gini_our = self.get_gini(item_matrix_k,item_count,slot,num_items, k, km_mod_n)
            fsat_ori, fsat_our = self.get_fsat(item_count,slot,num_items, k)

            key = '{}@{}'.format('Jain_ori', k)
            metric_dict[key] = round(jain_ori, self.decimal_place)

            key = '{}@{}'.format('Jain_our', k)
            metric_dict[key] = round(jain_our, self.decimal_place)

            key = '{}@{}'.format('QF_ori', k)
            metric_dict[key] = round(qf_ori, self.decimal_place)
        
            key = '{}@{}'.format('QF_our', k)
            metric_dict[key] = round(qf_our, self.decimal_place)

            key = '{}@{}'.format('Ent_ori', k)
            metric_dict[key] = round(ent_ori, self.decimal_place)

            key = '{}@{}'.format('Ent_our', k)
            metric_dict[key] = round(ent_our, self.decimal_place)

            key = '{}@{}'.format('Gini_ori', k)
            metric_dict[key] = round(gini_ori, self.decimal_place)

            key = '{}@{}'.format('Gini_our', k)
            metric_dict[key] = round(gini_our, self.decimal_place)

            key = '{}@{}'.format('FSat_ori', k)
            metric_dict[key] = round(fsat_ori, self.decimal_place)
        
            key = '{}@{}'.format('FSat_our', k)
            metric_dict[key] = round(fsat_our, self.decimal_place)
        
        return metric_dict

    def get_fsat(self, item_count, slot, num_items, k):
        n = num_items
        maximinshare = slot//n

        sat_items = (item_count >= maximinshare).sum()
        if maximinshare == 0:
            sat_items += n - item_count.shape[0]
        
        fsat_ori = sat_items / n

        fsat_our = (fsat_ori -  k/n)/ (1- k/n)

        return fsat_ori, fsat_our

    def get_gini(self,item_matrix,item_count,slot,num_items, k,km_mod_n):
        #gini_ori
        num_recommended_items = item_count.size
        item_count.sort()
        sorted_count = item_count

        idx = np.arange(num_items - num_recommended_items + 1, num_items + 1)
        gini_index = np.sum((2 * idx - num_items - 1) * sorted_count) / slot
        gini_index /= num_items

        #gini_our
        n = num_items
        gini_min = (n-km_mod_n)*km_mod_n/(n*slot)
        numerator = gini_index - gini_min
        denom = 1 - k/n - gini_min
        gini_our = numerator/denom

        rounded_gini_our = round(gini_our,15)
        assert approx(gini_our) == approx(rounded_gini_our)
        gini_our = rounded_gini_our

        if approx(gini_our) == 0:
            gini_our = 0
        elif approx(gini_our) == 1:
            gini_our = 1
        else:
            assert gini_our >= 0 and gini_our <=1, "need to be non-negative and not more than 1"


        assert abs(gini_our) == gini_our



        return gini_index, abs(gini_our)

    def get_entropy(self, item_count,slot,num_items, k, floor_km_n, km_mod_n):

        p = item_count/slot
        p_log_p = -p * (np.log(p)/np.log(num_items))
        ent_our_before_norm = p_log_p.sum()

        log_n_k = (np.log(k)/np.log(num_items))
        numerator = ent_our_before_norm - log_n_k

        if slot >= num_items:

            x = floor_km_n/slot 
            ent_max = -(num_items - km_mod_n) * (x*np.log(x)/np.log(num_items))
            
            x = (floor_km_n + 1)/slot
            ent_max -= km_mod_n * (x*np.log(x)/np.log(num_items))
            denom = ent_max - log_n_k

        else:
            denom = (np.log(slot/k)/np.log(num_items))

        ent_our = numerator/denom

        item_count = np.append(item_count, np.zeros(num_items-item_count.size)) #include count of unexposed items

        p = item_count/slot
        p_log_p = -p * (np.log(p)/np.log(num_items))
        ent_ori = p_log_p.sum()


        if not np.isnan(ent_ori):
            rounded_ent_ori = round(ent_ori,15)
            assert approx(ent_ori) == approx(rounded_ent_ori)
            ent_ori = rounded_ent_ori
            assert ent_ori >= 0 and ent_ori <=1, "need to be non-negative and not more than 1"

        rounded_ent_our = round(ent_our,15)
        assert approx(ent_our) == approx(rounded_ent_our)
        ent_our = rounded_ent_our

        if approx(ent_our) == 0:
            ent_our = 0
        elif approx(ent_our) == 1:
            ent_our = 1
        else:
            assert ent_our >= 0 and ent_our <=1, "need to be non-negative and not more than 1"

        return ent_ori, ent_our

    def get_jain(self, item_count, slot, num_items, k, floor_km_n, km_mod_n):

        numerator = slot**2
        sum_of_squared_count = (item_count**2).sum()

        assert sum_of_squared_count >= 0, "must be non-negative"

        denominator = num_items * sum_of_squared_count

        assert numerator >= 0, "numerator must be non-negative"
        assert denominator > 0, "denominator must be positive"

        jain_index = numerator/denominator

        rounded_jain = round(jain_index,15)
        assert approx(jain_index) == approx(rounded_jain)
        jain_index = rounded_jain

        assert jain_index >= 0 and jain_index <=1, "need to be non-negative and not more than 1"

        n = num_items
        jain_max = numerator / n
        jain_max /= n* (floor_km_n**2) + km_mod_n*(2*floor_km_n + 1)

        norm_jain_index = (jain_index - k/n)/(jain_max - k/n)
        rounded_jain_our = round(norm_jain_index,15)
        assert approx(norm_jain_index) == approx(rounded_jain_our)
        norm_jain_index = rounded_jain_our

        if approx(norm_jain_index) == 0:
            norm_jain_index = 0
        elif approx(norm_jain_index) == 1:
            norm_jain_index = 1
        else:
            assert norm_jain_index >= 0 and norm_jain_index <=1, "need to be non-negative and not more than 1"


        assert abs(norm_jain_index) == norm_jain_index

        return jain_index, abs(norm_jain_index) #because if it passes the assertion, it is 0, but it appears as -0.0

    def get_QF(self, item_matrix, slot, num_items, k):
        unique_count = np.unique(item_matrix).size
        qf_ori = unique_count / num_items

        numerator =  unique_count - k
        if slot >= num_items:
            denom =  num_items - k
        else:
            m = slot/k
            denom = k*(m-1)

        qf_our = numerator/denom

        return qf_ori, qf_our
class IAA(AbstractMetric):
    metric_type = EvaluatorType.RANKING
    metric_need = ['data.num_items','rec.score',"data.pos_items"]
    smaller = True
    def __init__(self, config):
        super().__init__(config)
        self.topk = config['topk']

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and number of items in total item set"""
        num_items = dataobject.get('data.num_items') - 1
        pred_rel = dataobject.get('rec.score')
        pos_items =  dataobject.get("data.pos_items")

        return num_items, pred_rel[:,1:], pos_items #pred_rel from 1 onwards to remove extra item in front

    def calculate_metric(self, dataobject):
        num_items, pred_rel, pos_items = self.used_info(dataobject)
        metric_dict = {}
        for k in self.topk:

            IAA_true_ori = self.get_IAA(num_items, pred_rel, pos_items, k)

            key = '{}@{}'.format('IAA_true_ori', k)
            metric_dict[key] = round(IAA_true_ori, self.decimal_place)

        return metric_dict

    def get_IAA(self, num_items, pred_rel, pos_items, k):
        #attention part
        att = np.zeros_like(pred_rel, dtype='float64')
        att[:,:k] = (k-np.arange(1,k+1))/(k-1) #normalised


        #TRUE rel - ORI
        #rel part
        full_rec_mat = pred_rel.sort(descending=True, stable=True).indices.numpy()
        rel = np.array([np.in1d(full_rec_mat[i]+1, pos_items[i], assume_unique=True) for i in range(pos_items.size)], dtype=int) 
        #i here is the index for users, not items!
        #rec_mat need +1 because item index in pos_items start from 1 instead of 0
        #pos_items size because it is np array of type object (different length), true IAA cannot use size but shape[0] instead
        IAA_u_true = np.abs(att-rel).sum(axis=1) / num_items 
        IAA_ori_true = IAA_u_true.mean()

        return IAA_ori_true

class IAArerank(AbstractMetric):
    metric_type = EvaluatorType.RANKING
    metric_need = ['data.num_items','rec.score',"data.pos_items", "rec.all_items"]
    smaller = True
    def __init__(self, config):
        super().__init__(config)
        self.topk = config['topk']

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and number of items in total item set"""
        num_items = dataobject.get('data.num_items') - 1
        pred_rel = dataobject.get('rec.score')
        pos_items =  dataobject.get("data.pos_items")
        full_rec_mat = dataobject.get("rec.all_items")

        return num_items, pred_rel[:,1:], pos_items, full_rec_mat

    def calculate_metric(self, dataobject):
        num_items, pred_rel, pos_items, full_rec_mat = self.used_info(dataobject)
        metric_dict = {}
        for k in self.topk:

            IAA_true_ori = self.get_IAA(num_items, pred_rel, pos_items, k, full_rec_mat)
            key = '{}@{}'.format('IAA_true_ori', k)
            metric_dict[key] = round(IAA_true_ori, self.decimal_place)

        return metric_dict

    def get_IAA(self, num_items, pred_rel, pos_items, k, full_rec_mat):
        #attention part
        att = np.zeros_like(pred_rel, dtype='float64')
        att[:,:k] = (k-np.arange(1,k+1))/(k-1) #normalised

        #TRUE rel - ORI
        #rel part
        rel = np.array([np.in1d(full_rec_mat[u], pos_items[u], assume_unique=True) for u in range(pos_items.size)], dtype=int) 
        #u here is the index for users, not items!
        #rec_mat does NOT need +1 because item index in full_rec_mat[u] already starts from 1
        #pos_items size because it is np array of type object (different length), true IAA cannot use size but shape[0] instead
        IAA_u_true = np.abs(att-rel).sum(axis=1) / num_items 
        IAA_ori_true = IAA_u_true.mean()

        return IAA_ori_true
    
class IAAinsert(IAArerank):
    metric_type = EvaluatorType.RANKING
    metric_need = ['data.num_items','rec.score',"data.pos_items", "rec.all_items"]
    smaller = True

    def get_IAA(self, num_items, pred_rel, pos_items, k, full_rec_mat):
        #attention part
        att = np.zeros_like(pred_rel, dtype='float64')
        att[:,:k] = (k-np.arange(1,k+1))/(k-1) #normalised

        #TRUE rel - ORI
        #rel part
        rel = np.array([np.in1d(full_rec_mat[u], pos_items[u], assume_unique=True) for u in range(pos_items.shape[0])], dtype=int) 
        #u here is the index for users, not items!
        #rec_mat does NOT need +1 because item index in full_rec_mat[u] already starts from 1
        #pos_items shape[0] because it is [num_user, num_k] based on artificial insertion experiment
        IAA_u_true = np.abs(att-rel).sum(axis=1) / num_items 
        IAA_ori_true = IAA_u_true.mean()

        return IAA_ori_true

class MME_IIF_AIF(AbstractMetric):
    metric_type = EvaluatorType.RANKING
    metric_need = ['rec.items', 'data.num_items', "data.pos_items"]
    smaller = True
    def __init__(self, config):
        super().__init__(config)
        self.topk = config['topk']

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and number of items in total item set"""
        
        item_matrix = dataobject.get('rec.items')
        num_items = dataobject.get('data.num_items') - 1
        pos_items = dataobject.get('data.pos_items')

        return item_matrix.numpy(), num_items, pos_items

    def calculate_metric(self, dataobject):
        item_matrix, num_items, pos_items = self.used_info(dataobject)
        metric_dict = {}
        for k in self.topk:
            MME, II_F, AI_F = self.get_metrics(item_matrix[:, :k], num_items, pos_items)

            key = '{}@{}'.format('MME_ori', k)
            metric_dict[key] = round(MME, self.decimal_place)

            key = '{}@{}'.format('II-F_ori', k)
            metric_dict[key] = round(II_F, self.decimal_place)

            key = '{}@{}'.format('AI-F_ori', k)
            metric_dict[key] = round(AI_F, self.decimal_place)

        return metric_dict

    def get_metrics(self, item_matrix, num_items, pos_items):
        rec = item_matrix
        rel = pos_items
        m = rec.shape[0]

        gamma = 0.8

        #build user-item matrix
        user_item_rel = np.zeros((m, num_items))


        rank_matrix = np.zeros_like(user_item_rel)
        for i in range(len(rec)):
            rank_matrix[i][rec[i]-1] = np.where(rec[i])[0]+1
        
        user_item_exp_inv = np.copy(rank_matrix)
        user_item_exp_rbp = np.copy(rank_matrix)

        user_item_exp_inv[user_item_exp_inv.nonzero()] = 1/user_item_exp_inv[user_item_exp_inv.nonzero()]
        user_item_exp_rbp[user_item_exp_rbp.nonzero()] = gamma**(user_item_exp_rbp[user_item_exp_rbp.nonzero()]-1)

        for i in range(len(rel)):
            user_item_rel[i][rel[i]-1] = 1

        max_envies = np.zeros(num_items)

        #Credits to https://github.com/usaito/kdd2022-fair-ranking-nsw/blob/main/src/synthetic/func.py#L64
        for i in range(num_items):
            u_d_swap = (user_item_exp_inv * user_item_rel[:, [i]*num_items]).sum(0)
            d_envies = u_d_swap - u_d_swap[i]
            max_envies[i] = d_envies.max()

        MME = max_envies.mean() / m

        #start II-F and AI-F
        r_u_star = user_item_rel.sum(1)[:,np.newaxis]

        e_ui_star = user_item_rel  * (1-np.power(gamma, r_u_star))/(1-gamma)
        r_u_star[r_u_star==0] = 1

        e_ui_star /= r_u_star

        diff = user_item_exp_rbp - e_ui_star
        II_F = np.power(diff, 2).mean()
        AI_F = np.power(diff.mean(0),2).mean()

        return MME, II_F, AI_F

class IBO_IWO(AbstractMetric):
    metric_type = EvaluatorType.RANKING
    metric_need = ['rec.items', 'data.num_items', "data.pos_items"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config['topk']

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and number of items in total item set"""
        
        item_matrix = dataobject.get('rec.items')
        num_items = dataobject.get('data.num_items') - 1
        pos_items = dataobject.get('data.pos_items')

        return item_matrix.numpy(), num_items, pos_items

    def calculate_metric(self, dataobject):
        item_matrix, num_items, pos_items = self.used_info(dataobject)
        metric_dict = {}

        for k in self.topk:
            
            IBO_ori, IWO_ori, IBO_our, IWO_our = self.get_IBOIWO(item_matrix[:, :k], num_items, pos_items)
            key = '{}@{}'.format('IBO_ori', k)
            metric_dict[key] = round(IBO_ori, self.decimal_place)

            key = '{}@{}'.format('IWO_ori', k)
            metric_dict[key] = round(IWO_ori, self.decimal_place)

            key = '{}@{}'.format('IBO_our', k)
            metric_dict[key] = round(IBO_our, self.decimal_place)

            key = '{}@{}'.format('IWO_our', k)
            metric_dict[key] = round(IWO_our, self.decimal_place)

        return metric_dict

    def get_IBOIWO(self, item_matrix, num_items, pos_items):
        rec = item_matrix
        rel = pos_items
        m = rec.shape[0]
        k = rec.shape[1]
        inv = 1/(np.arange(k, dtype="int")+1)


        user_item_rel = np.zeros((m, num_items))
        rank_matrix = np.zeros_like(user_item_rel)
        for i in range(len(rec)):
            rank_matrix[i][rec[i]-1] = np.where(rec[i])[0]+1
        
        user_item_exp_inv = np.copy(rank_matrix)
        user_item_exp_inv[user_item_exp_inv.nonzero()] = 1/user_item_exp_inv[user_item_exp_inv.nonzero()]

        for i in range(len(rel)):
            user_item_rel[i][rel[i]-1] = 1

        imp_i_arr = (user_item_exp_inv* user_item_rel).sum(0)/m
        imp_unif_arr = user_item_rel.sum(0) * inv.sum()/num_items/m

        ratio = imp_i_arr/imp_unif_arr
        if any(np.isnan(ratio)):
            IBO_ori = np.nan
            IWO_ori = np.nan
        else:
            better_ori = ratio >= 1.1
            worse_ori = ratio <= 0.9
            IBO_ori = better_ori.sum()/num_items
            IWO_ori = worse_ori.sum()/num_items

        mask = imp_unif_arr!=0
        filtered_imp_unif = imp_unif_arr[mask] #only items that have at least 1 relevant user

        filtered_imp_i = imp_i_arr[mask]

        better_our = filtered_imp_i >= (1.1 * filtered_imp_unif)
        worse_our = filtered_imp_i <= (0.9 * filtered_imp_unif)
        
        IBO_our = better_our.mean()
        IWO_our = worse_our.mean()

        return IBO_ori, IWO_ori, IBO_our, IWO_our 

class IFD(AbstractMetric):
    metric_type = EvaluatorType.RANKING
    metric_need = ['rec.score', 'data.num_items', "data.pos_items", "rec.topk"]
    smaller = True

    def __init__(self, config):
        super().__init__(config)
        self.topk = config['topk']

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and number of items in total item set"""
        
        pred_rel = dataobject.get('rec.score')
        num_items = dataobject.get('data.num_items') - 1
        pos_items = dataobject.get('data.pos_items')
        rel_top_k = dataobject.get('rec.topk')

        return pred_rel, num_items, pos_items, rel_top_k.numpy()

    def calculate_metric(self, dataobject):
        pred_rel, num_items, pos_items, rel_top_k = self.used_info(dataobject)
        metric_dict = {}

        for k in self.topk:
            
            IFD_div, IFD_mul = self.get_IFD(pred_rel, num_items, pos_items, rel_top_k[:,:k], k)
            key = '{}@{}'.format('IFD_div', k)
            metric_dict[key] = round(IFD_div, self.decimal_place)

            key = '{}@{}'.format('IFD_mul', k)
            metric_dict[key] = round(IFD_mul, self.decimal_place)

        return metric_dict

    def get_full_rec_mat(self, pred_rel):
        return pred_rel[:,1:]\
                            .sort(stable=True, descending=True)\
                            .indices + 1 # +1 to correct index (item index starts from 1 in pos_items)

    def get_IFD(self, pred_rel, num_items, pos_items, rel_top_k, k):

        def get_item_rank(full_item_rec_for_u, item_arr):
            rank_pos = np.where(np.in1d(full_item_rec_for_u, item_arr))[0] #since it's 1d, take the first dim
            rank_weight = 1/np.log2(rank_pos+2)
            return rank_weight

        full_rec_mat = self.get_full_rec_mat(pred_rel)
        
        rel = rel_top_k

        m = rel.shape[0]
        n = num_items

        rank_plus_two = np.arange(k) + 2
        rank_plus_two = np.tile(rank_plus_two,(m,1))
        rank_weight = 1/np.log2(rank_plus_two)
    
        
        J_mul_to_extend = rank_weight * rel
        J_mul = np.append(J_mul_to_extend, np.zeros((m,n-k)), axis=1)

        J_mul_for_pdist = J_mul.reshape(-1, num_items,1)
    
        IFD_mul_u = 0
        IFD_div_u = 0
        for u, J_mul_u in enumerate(J_mul_for_pdist):
            IFD_mul_u += pdist(J_mul_u, 'sqeuclidean').mean()

            #in our case, r_ui is either 0 (but excluded from the computation) or 1, so we can skip the division by r_ui
            J_div_for_this_u = get_item_rank(full_rec_mat[u], pos_items[u])
            J_div_for_cdist = J_div_for_this_u.reshape(-1,1)

            #to create H-u, we can do this again because of the r_ui = 1 if and only if the item is relevant
            IFD_div_u += cdist(J_div_for_cdist, J_div_for_cdist, lambda u, v: max(0,u-v)).mean() 

        IFD_mul = IFD_mul_u / m
        IFD_div = IFD_div_u / m
        return IFD_div, IFD_mul 

class IFDrerank(IFD):
    metric_type = EvaluatorType.RANKING
    metric_need = ['rec.all_items', 'data.num_items', "data.pos_items", "rec.topk"]
    smaller = True

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and number of items in total item set"""
        
        full_rec_mat = dataobject.get('rec.all_items')
        num_items = dataobject.get('data.num_items') - 1
        pos_items = dataobject.get('data.pos_items')
        rel_top_k = dataobject.get('rec.topk')
        return full_rec_mat, num_items, pos_items, rel_top_k.numpy() #full_rec_mat replaces pred_rel

    def get_full_rec_mat(self, pred_rel): #here the pred_rel is actually full_rec_mat
        return pred_rel

class HD(AbstractMetric):
    metric_type = EvaluatorType.RANKING
    metric_need = ['rec.items', 'data.num_items', "data.pos_items", 'rec.topk']
    smaller = True

    def __init__(self, config):
        super().__init__(config)
        self.topk = config['topk']

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and number of items in total item set"""
        
        item_matrix = dataobject.get('rec.items')
        num_items = dataobject.get('data.num_items') - 1
        pos_items = dataobject.get('data.pos_items')
        rel_top_k = dataobject.get("rec.topk")

        return item_matrix.numpy(), num_items, pos_items, rel_top_k

    def calculate_metric(self, dataobject):
        item_matrix, num_items, pos_items, rel_top_k = self.used_info(dataobject)
        metric_dict = {}

        for k in self.topk:
            
            HD = self.get_HD(item_matrix[:, :k], num_items, pos_items, rel_top_k[:, :k], k)
            key = '{}@{}'.format('HD', k)
            metric_dict[key] = round(HD, self.decimal_place)

        return metric_dict

    def get_HD(self, item_matrix, num_items, pos_items, rel_top_k, k):
        '''
        For consistency with the original measure, major parts of this code are taken and
        modified from https://github.com/olivierjeunen/EARS-recsys-2021/blob/main/src/environment.py
        Credits to the original author for the code
        '''
        
        #initialize
        gamma = 0.9 
        
        rec = item_matrix
        rel_items = pos_items
        m = rec.shape[0]

        r_ui = np.zeros((m, num_items))
        for u in range(len(rel_items)): #this is actually iterating over users, not items
            r_ui[u][rel_items[u]-1] = 1

        #get optimal recos
        padded_gt = np.argsort(-r_ui, axis=1) + 1 #+1 tocorrect index; np.argsort is by default quicksort, not stable sort
        padded_gt_until_k = padded_gt[:,:k]

        #Q (Relevance) Part; in the original code, Q is R
        #r_ui_prime = normalized_P_R, contains normalised true relevance
        r_ui_prime = r_ui / r_ui.sum(axis=1, keepdims=True)
        r_ui_prime_extend = np.append(np.full((r_ui_prime.shape[0],1),np.inf), r_ui_prime, axis=1) #first col is dummy

        Q = np.take_along_axis(r_ui_prime_extend, padded_gt_until_k, axis=1)
        Q = Q.mean(axis=0)

        #C (Click) Part
        #Exposure term
        one_minus_rel = 1 - rel_top_k
        s_up = np.cumprod(one_minus_rel, axis=-1)[:,:-1]

        #first column needs to be ones
        s_up = np.hstack((np.ones(m).reshape(-1,1), s_up))

        gamma_e_rbp = np.cumprod(gamma*np.ones_like(s_up[0,:]))
        exposure_term = gamma_e_rbp * s_up

        #top k recommended items
        c_up = rel_top_k * exposure_term 

        #user-wise normalization
        c_up_prime = c_up / c_up.sum(axis=1, keepdims=True)
        np.nan_to_num(c_up_prime, copy=False) #important because there might be division by 0, convert nans to 0

        indptr = np.arange(m + 1) * rec.shape[1]
        indices = rec.flatten()
        data = c_up_prime.flatten()
        c_ui_full = csr_matrix((data, indices, indptr), shape = (m, num_items+1))

        c_up_gt = np.take_along_axis(c_ui_full, padded_gt_until_k, axis=1)
        c_up_gt /= c_up_gt.sum(axis=1)
        np.nan_to_num(c_up_gt, copy=False)

        C = c_up_gt.mean(axis=0).A1

        #Hellinger Distance between Q and C
        #same result as original code, but slightly faster
        squared_diff = (np.sqrt(Q) - np.sqrt(C))**2
        HD = np.sqrt(1/2) * np.sqrt(squared_diff.sum())

        return HD 
