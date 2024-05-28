import builtins
from recbole.config import Config
from recbole.evaluator.evaluator import Evaluator

import pickle
import torch

import warnings 
warnings.filterwarnings('ignore')

import os
import time
import datetime

path = "experiments/base"

def print(*args, **kwargs):
    with open(f"{path}/log_{dataset}_{model_name}.txt", 'a+') as f:
        return builtins.print(*args, file=f, **kwargs)
    

list_dataset = [
                "Amazon-lb", 
                "Lastfm", 
                "QK-video",
                "ML-10M", 
                ]
list_model = [
    "BPR",
    "MultiVAE",
    "ItemKNN",
    "NCL"
]

list_k = [10]
max_k = max(list_k)

for dataset in list_dataset:

    for model_name in list_model:
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(now)

        print(f"Doing {dataset} - {model_name}")

        try:
            with open(f"{path}/base_{dataset}_{model_name}.pickle","rb") as f:
                found = pickle.load(f)
                print("found existing evaluation result ")
                print(found)
        except:
            print(f"Cannot find existing result for {dataset} - {model_name}, proceed with eval")
            config = Config(
                model=model_name, 
                dataset="new_"+dataset, 
                config_file_list=["RecBole/recbole/properties/overall.yaml"],

                config_dict={"topk": list_k, 
                            "metrics":[
                                    "RelMetrics",
                                    "FairWORel",
                                    "IAA",
                                    "MME_IIF_AIF",
                                    "IBO_IWO",
                                    "IFD",
                                    "HD"
                                ]})

            evaluator = Evaluator(config)

            list_filename = [f for f in os.listdir("cluster/best_struct") if dataset in f and model_name in f]

            assert len(list_filename) == 1

            with open(f"cluster/best_struct/{list_filename[0]}","rb") as f:
                struct = pickle.load(f)

                #avoid problem with torch.split because the structs are from recommending 20 items (instead of 10)
                rec_topk = struct.get("rec.topk")
                rec_topk = torch.cat([rec_topk[:,:max_k],rec_topk[:,-1:]], dim=1)
                struct.set("rec.topk", rec_topk)

                start_time = time.time()
                result = evaluator.evaluate(struct)
                print("total time taken: ", time.time() - start_time)
                print(result)

                with open(f"{path}/base_{dataset}_{model_name}.pickle","wb") as f:
                    pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)