from pathlib import Path

experiment_conf = {
   "mini_transformer":{
    "d_model": 256,
    "N":5,
    "h":4,
    "dropout": 0.3,
    "d_ff": 1024,
  },
  "medium_transformer":{
    "d_model": 512,
    "N":5,
    "h":4,
    "dropout": 0.1,
    "d_ff": 2048,
  },
  "transformer_base":{
    "d_model": 512,
    "N":4,
    "h":4,
    "dropout": 0.1,
    "d_ff": 2048,
  }
}

def get_config(folder='', experiment_name="transformer_base"):
  main_conf = {
    "batch_size": 64,
    "num_epochs": 20,
    "lr": 10**-4,
    "seq_len": 32,
    "lang_src": "en",
    "lang_tgt": "id",
    "model_folder": f"./{experiment_name}/weights",
    "model_basename": "tmodel_",
    "preload": "latest",
    "tokenizer_file": "./" + experiment_name + "/tokenizer_{0}.json",
    "experiment_name": f"./{experiment_name}/runs/tmodel",
    "path_files": f"./{experiment_name}/data-jawa2indonesia",
    "path_dataset": f"./{experiment_name}/dataset",
    "main_folder": folder
  }
  main_conf.update(experiment_conf[experiment_name])
  return main_conf

def get_weights_file_path(config, epoch: str):
  model_folder = f"{config['model_folder']}"
  model_filename = f"{config['model_basename']}{epoch}.pt"
  return str(Path(model_folder) / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
  model_folder = f"{config['model_folder']}"
  model_filename = f"{config['model_basename']}*"
  weights_files = list(Path(model_folder).glob(model_filename))
  if len(weights_files) == 0:
      return None
  weights_files.sort()
  return str(weights_files[-1])