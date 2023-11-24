def get_config():
    return {
        "batch_size": 24,
        "num_epochs": 100,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'cfilt/iitb-english-hindi', 
        "train_data_size": 30000,
        "lang_src": "en",
        "lang_tgt": "hi",
        "model_folder": "output/weights",
        "model_basename": "tmodel_",
        "preload": None, # set --> "latest" here if you want to resume training from latest epoch. Set --> 01 here if you want to resume training from say 01th epoch. Set None here if you want to do training from scratch.
        "tokenizer_file": "output/vocab/tokenizer_{0}.json",
        "experiment_name": "tensorboard_logging"
    }
