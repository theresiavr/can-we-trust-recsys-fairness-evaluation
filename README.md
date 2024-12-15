# Can We Trust Recommender System Fairness Evaluation? The Role of Fairness and Relevance ⚖ (SIGIR 2024 Full Paper) 

This repository contains the appendix as well as the code used for the experiments and analysis in "Can We Trust Recommender System Fairness Evaluation? The Role of Fairness and Relevance" by Theresia Veronika Rampisela, Tuukka Ruotsalo, Maria Maistro, and Christina Lioma. This work has been accepted to SIGIR 2024 as a full paper. 

[[ACM]](https://doi.org/10.1145/3626772.3657832) [[arXiv]](https://arxiv.org/abs/2405.18276) 

# Abstract
Relevance and fairness are two major objectives of recommender systems (RSs). Recent work proposes measures of RS fairness that are either independent from relevance (fairness-only) or conditioned on relevance (joint measures). While fairness-only measures have been studied extensively, we look into whether joint measures can be trusted. We collect all joint evaluation measures of RS relevance and fairness, and ask: How much do they agree with each other? To what extent do they agree with relevance/fairness measures? How sensitive are they to changes in rank position, or to increasingly fair and relevant recommendations? We empirically study for the first time the behaviour of these measures across 4 real-world datasets and 4 recommenders. We find that most of these measures: i) correlate weakly with one another and even contradict each other at times; ii) are less sensitive to rank position changes than relevance- and fairness-only measures, meaning that they are less granular than traditional RS measures; and iii) tend to compress scores at the low end of their range, meaning that they are not very expressive. We counter the above limitations with a set of guidelines on the appropriate usage of such measures, i.e., they should be used with caution due to their tendency to contradict each other and of having a very small empirical range.

# License and Terms of Usage
The code is usable under the MIT License. Please note that RecBole may have different terms of usage (see [their page](https://github.com/RUCAIBox/RecBole) for updated information).

# Citation
If you use the code for the fairness-only measures in `metrics.py`, please cite our paper and the original papers proposing the measures.
```BibTeX
@article{10.1145/3631943,
author = {Rampisela, Theresia Veronika and Maistro, Maria and Ruotsalo, Tuukka and Lioma, Christina},
title = {Evaluation Measures of Individual Item Fairness for Recommender Systems: A Critical Study},
year = {2024},
issue_date = {June 2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {3},
number = {2},
url = {https://doi.org/10.1145/3631943},
doi = {10.1145/3631943},
journal = {ACM Trans. Recomm. Syst.},
month = nov,
articleno = {18},
numpages = {52},
keywords = {Item fairness, individual fairness, fairness measures, evaluation measures, recommender systems}
}
```
If you use the code outside of RecBole's original code, please cite the following:
```BibTeX
@inproceedings{10.1145/3626772.3657832,
author = {Rampisela, Theresia Veronika and Ruotsalo, Tuukka and Maistro, Maria and Lioma, Christina},
title = {Can We Trust Recommender System Fairness Evaluation? The Role of Fairness and Relevance},
year = {2024},
isbn = {9798400704314},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3626772.3657832},
doi = {10.1145/3626772.3657832},,
pages = {271–281},
numpages = {11},
keywords = {fairness and relevance evaluation, recommender systems},
location = {Washington DC, USA},
series = {SIGIR '24}
}
```
# Datasets

## Downloads
We use the following datasets, that can be downloaded from the Google Drive folder provided by [RecBole](https://recbole.io/dataset_list.html), under ProcessedDatasets:
- Amazon-lb: this dataset can be found in the Amazon2018 folder. The name of the folder is Amazon_Luxury_Beauty
- Lastfm is under the LastFM folder
- Ml-10m is under the MovieLens folder; the file is ml-10m.zip

1. Download the zip files corresponding to the full datasets (not the examples) and place them inside an empty `dataset` folder in the main folder.
2. Unzip the files.
3. Ensure that the name of the folder and the .inter files are the same as in the [dataset properties](https://github.com/theresiavr/can-we-trust-recsys-fairness-evaluation/tree/main/RecBole/recbole/properties/dataset).

QK-video can be downloaded from the link on the [Tenrec repository](https://github.com/yuangh-x/2022-NIPS-Tenrec).

## Preprocessing
Please follow the instructions in the notebooks under the `preproc_data` folder.
Further preprocessing settings are configured under `Recbole/recbole/properties/dataset`

# Model training and hyperparameter tuning
Please see the `cluster` folder to find the hyperparameter tuning script.
The hyperparameter search space can be found in  `Recbole/hyperchoice`, and the file `cluster/bestparam.txt` contains the best hyperparameter configurations.
