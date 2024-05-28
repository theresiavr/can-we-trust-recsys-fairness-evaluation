# -*- coding: utf-8 -*-
# @Time   : 2020/7/24 15:57
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : run_hyper.py
# UPDATE:
# @Time   : 2020/8/20 21:17, 2020/8/29
# @Author : Zihan Lin, Yupeng Houconda 
# @Email  : linzihan.super@foxmail.com, houyupeng@ruc.edu.cn

import argparse

from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function

import pickle
from datetime import datetime


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', type=str, default=None, help='fixed config files')
    parser.add_argument('--params_file', type=str, default=None, help='parameters file')
    parser.add_argument('--output_file', type=str, default='hyper_example.result', help='output file')
    args, extra_args = parser.parse_known_args()

    # plz set algo='exhaustive' to use exhaustive search, in this case, max_evals is auto set
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    hp = HyperTuning(objective_function, algo='exhaustive',
                     params_file=args.params_file, fixed_config_file_list=config_file_list)
    hp.run()
    # hp.export_result(output_file=args.output_file)
    print('best params: ', hp.best_params)
    print('best result: ')
    print(hp.params2result[hp.params2str(hp.best_params)])

    now = datetime.now()
    time = str(now.strftime("%Y-%m-%d_%H%M"))

    data = extra_args[0].split("=")[-1]
    model = extra_args[1].split("=")[-1]
    
    with open(f'bestparam/result_{data}_{model}_{time}.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(hp.params2result[hp.params2str(hp.best_params)], f, pickle.HIGHEST_PROTOCOL)

    with open(f'bestparam/all_result_{data}_{model}_{time}.pickle', 'wb') as f:
        pickle.dump(hp.params2result, f, pickle.HIGHEST_PROTOCOL)

    with open(f'bestparam/param_{data}_{model}_{time}.pickle', 'wb') as f:
        pickle.dump(hp.best_params, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
