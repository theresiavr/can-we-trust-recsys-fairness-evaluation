from recbole.quick_start import run_recbole

dataset = [
    "Amazon-lb",
    "Lastfm",
    "ML-10M",
    "QK-video"
]

models = [
    "Pop",
    ]

for data in dataset:
    for model in models:      
        curr_result = run_recbole(
                                model=model,
                                dataset="new_"+data,
                                config_dict={
                                    "data_path":"preproc_data\\",
                                    "benchmark_filename": ['train', 'valid', 'test'],
                                }
                                )
