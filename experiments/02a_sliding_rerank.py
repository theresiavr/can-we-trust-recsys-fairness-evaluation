import builtins
from recbole.config import Config
from recbole.evaluator.evaluator import Evaluator

import pickle


import warnings 
warnings.filterwarnings('ignore')

import os
import time

import datetime
import torch


def print(*args, **kwargs):
    with open(f'experiments/sliding/log_sliding_{dataset}_{model_name}_{rerank}.txt', 'a+') as f:
        return builtins.print(*args, file=f, **kwargs)
    

list_dataset = [
                "Amazon-lb", 
                "Lastfm", 
                "QK-video",
                "ML-10M", 
                ]
list_model = [
    "NCL"
]

list_rerank = ["combmnz"]

window = 5
total_k = 5

for dataset in list_dataset:
    for model_name in list_model:
        for rerank in list_rerank:
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(now)

            print(f"Doing {dataset} - {model_name} - {rerank}")


            config = Config(
                    model=model_name, 
                    dataset="new_"+dataset, 
                    config_file_list=["RecBole/recbole/properties/overall.yaml"],

                    config_dict={"topk": [window], 
                                "metrics":[
                                        "RelMetrics",
                                        "FairWORel",
                                        "IAArerank",
                                        "MME_IIF_AIF",
                                        "IBO_IWO",
                                        "IFDrerank",
                                        "HD"
                                    ]})

            evaluator = Evaluator(config)

            list_filename = [f for f in os.listdir("reranking/rerank_struct") if dataset in f and model_name in f and rerank in f and "rerank25" in f and "at10" in f]

            assert len(list_filename) == 1

            with open(f"reranking/rerank_struct/{list_filename[0]}","rb") as f:
                struct = pickle.load(f)

            item_matrix = struct.get('rec.items')
            rec_mat = struct.get('rec.topk')
            full_rec_mat = struct.get('rec.all_items')

            for k in range(0,total_k,1):
                print(f"Doing {dataset}, {model_name}, {k}:{k+window}")

                #slice item_matrix and relevance_matrix
                updated_item_matrix = item_matrix[:,k:k+window]
                updated_rec_mat = torch.cat((rec_mat[:,k:k+window], rec_mat[:,-1:]), axis=1)
                updated_full_rec_mat = torch.cat((full_rec_mat[:,k:], full_rec_mat[:,:k]), axis=1)

                struct.set("rec.items",updated_item_matrix)
                struct.set("rec.topk",updated_rec_mat)
                struct.set("rec.all_items",updated_full_rec_mat)
                
                start_time = time.time()
                result = evaluator.evaluate(struct)
                print("total time taken: ", time.time() - start_time)
                print(result)

                with open(f"experiments/sliding/result_{dataset}_{model_name}_{rerank}_at5_{k}-{k+window}.pickle","wb") as f:
                    pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)