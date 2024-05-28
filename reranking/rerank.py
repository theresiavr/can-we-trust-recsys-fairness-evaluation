import os
import pickle
import time
import builtins
import torch

import numpy as np
import statistics

from copy import deepcopy
from math import floor
from collections import Counter

import datetime

from recbole.config import Config
from recbole.evaluator.evaluator import Evaluator

import warnings
warnings.simplefilter("ignore")

def load_dataset_model(dataset: str, model: str, k: int, top_k_extend: int):
    # print(f"Loading {dataset} -- {model}")

    list_file = os.listdir("cluster/best_struct/")
    file_for_dataset_model = [x for x in list_file if dataset in x and model in x]
    assert len(file_for_dataset_model) == 1

    with open("cluster/best_struct/"+file_for_dataset_model[0],"rb") as f:
        struct = pickle.load(f)
    
    rec_score = deepcopy(struct.get("rec.score"))

    rec_topk = deepcopy(struct.get("rec.topk"))
    pos_item = struct.get("data.pos_items")

    #get top k extended items
    if k == top_k_extend:
        rec_items = deepcopy(struct.get("rec.items")[:,:k])
    else:
        rec_items = rec_score[:,1:]\
                                .sort(descending=True, stable=True)\
                                .indices[:,:top_k_extend]+1 #because item index starts from 1
    
    return struct, rec_score, rec_items, rec_topk, pos_item

def compute_coverage(rec_items, k: int, top_k_extend: int):
    item_id, num_times = torch.unique(rec_items.flatten(), return_counts=True)
    coverage = Counter(dict(map(lambda tup: (x.item() for x in tup), zip(item_id, num_times))))
    if k != top_k_extend:
        item_id, num_times_outside_top_k = torch.unique(rec_items[:,k:].flatten(), return_counts=True)
        coverage_outside_top_k = Counter(dict(map(lambda tup: (x.item() for x in tup), zip(item_id, num_times_outside_top_k))))
        coverage.subtract(coverage_outside_top_k)
    return coverage


def get_score_per_user(item, recommendation, score, mode="combmnz"):
    #to be used with borda and combnz
    match mode:
        case "combmnz":
            return torch.where(recommendation[:,:k]==item, score[:,:k], 0).sum(1)


def combmnz(dict_score_rec_items, dict_score_new_rec_items, coverage):
    total_score = {}

    for key in coverage.keys():
        val_ori = dict_score_rec_items[key]
        val_new = dict_score_new_rec_items[key]

        stacked = np.stack([val_ori, val_new], axis=1)

        multiplier = np.count_nonzero(stacked, axis=1)

        total_score[key] = stacked.sum(1) * multiplier

    return total_score

def rerank_combmnz(rec_score, rec_items, rec_topk, pos_item, k, top_k_extend):
    coverage = compute_coverage(rec_items, k, top_k_extend)
    ori_pred_rel = deepcopy(rec_score)
    combmnz_rec_topk = deepcopy(rec_topk)
    combmnz_rec_topk = torch.cat((combmnz_rec_topk[:,:k], combmnz_rec_topk[:,-1:]), axis=1)

    #within the extended top-k, get ranking based on relevance (ori_rec_items)
    ori_rec_items = deepcopy(rec_items)
    ori_pred_rel_extended_top_k = ori_pred_rel\
                                                .sort(descending=True, stable=True)\
                                                .values[:,:top_k_extend]

    #min-max normalise relevance score
    max_pred_rel = ori_pred_rel_extended_top_k.max()
    min_pred_rel = np.nanmin(ori_pred_rel_extended_top_k[ori_pred_rel_extended_top_k != -np.inf])
    normalised_rel_items = (ori_pred_rel_extended_top_k - min_pred_rel) / (max_pred_rel-min_pred_rel)

    #within the extended top-k, get ranking based on coverage (new_rec_items)
    least_to_most_coverage = sorted(coverage, key=lambda i: coverage.get(i))
    for_indices = deepcopy(rec_items)
    for_indices.apply_(lambda x: least_to_most_coverage.index(x)) #inplace
    new_rec_items = for_indices.sort().values.apply_(lambda x: least_to_most_coverage[x])

    #get score from coverage of items
    coverage_rec_items = deepcopy(new_rec_items)
    coverage_rec_items.apply_(lambda x: coverage[x]) #inplace

    #min-max normalise coverage score, and calculate 1-score (to promote least covered item)
    max_cov = coverage_rec_items.max()
    min_cov = coverage_rec_items.min()

    normalised_coverage_items = 1 - (coverage_rec_items-min_cov) / (max_cov - min_cov)

    dict_score_rec_items = {i: get_score_per_user(i, ori_rec_items, normalised_rel_items, "combmnz") for i in coverage.keys()}
    dict_score_new_rec_items = {i: get_score_per_user(i, new_rec_items, normalised_coverage_items, "combmnz") for i in coverage.keys()}

    total_score = combmnz(dict_score_rec_items, dict_score_new_rec_items, coverage)

    mnz_score = torch.zeros_like(rec_score, dtype=torch.float64)

    for key, val in total_score.items():
        mnz_score[:,key] = torch.from_numpy(val)


    combmnz_full_rec_mat = mnz_score[:,1:]\
                                .sort(descending=True, stable=True)\
                                .indices + 1 #first column is dummy
    combmnz_rec_item = combmnz_full_rec_mat[:,:k]

    #update relevance value
    for u in range(len(combmnz_rec_item)):
        new_indicator = torch.isin(combmnz_rec_item[u], torch.from_numpy(pos_item[u])).int()
        combmnz_rec_topk[u,:-1] = new_indicator

    combmnz_rec_topk = torch.cat((combmnz_rec_topk[:,:k], combmnz_rec_topk[:,-1:]), axis=1)

    return combmnz_rec_item, combmnz_rec_topk, combmnz_full_rec_mat

def evaluate(ori_struct, rec_items, rec_topk, full_rec_mat):
    #update the recommended items, true relevance value of item at the top k, and insert the full rec mat to evaluate IAA just for reranking
    start_time = time.time()
    updated_struct = deepcopy(ori_struct)
    updated_struct.set("rec.items", rec_items)
    updated_struct.set("rec.topk", rec_topk)
    updated_struct.set("rec.all_items", full_rec_mat)

        
    return updated_struct

def save_result_struct(exp_name, struct):

    with open(f"reranking/rerank_struct/{filename_prefix}_{exp_name}.pickle","wb") as f:
        pickle.dump(
            struct,
            f, 
            pickle.HIGHEST_PROTOCOL
        )                     

#start main
list_k = [
    10, 
    ]

list_dataset = [
                "Amazon-lb", 
                "Lastfm", 
                "ML-10M",
                "QK-video", 
                ]

list_model = [
            "ItemKNN",
              "MultiVAE",
              "BPR",
              "NCL",
              ]


list_extended_k = [25]

for dataset in list_dataset:
    for model in list_model:
        for k in list_k:
            for extended_k in list_extended_k:
                filename_prefix = f"{dataset}_{model}_at{k}_rerank{extended_k}"
                ori_struct, rec_score, rec_items, rec_topk, pos_item = load_dataset_model(dataset, model, k, extended_k)

                #COMBMNZ will only cause change in fairness scores if we rerank outside the top-k, because they are used for user-wise reranking
                if extended_k != k:
                    #reranking with COMBMNZ
                    exp_name = "combmnz"

                    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(now)
                    print("Starting combmnz")
                    combmnz_rec_items, combmnz_rec_topk, combmnz_full_rec_mat = rerank_combmnz(rec_score, rec_items, rec_topk, pos_item, k, extended_k)

                    combnz_struct = evaluate(ori_struct, combmnz_rec_items, combmnz_rec_topk, combmnz_full_rec_mat)
                    save_result_struct(exp_name, combnz_struct)
            

        