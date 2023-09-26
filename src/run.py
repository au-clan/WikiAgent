import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import instantiate
from model_utils import SystemMessage, HumanMessage
from utils import printc
import torch
import os

# set env var HYDRA_FULL_ERROR=1 
os.environ['HYDRA_FULL_ERROR'] = '1'

@hydra.main(version_base = None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    messages = [SystemMessage("You're a nice assistant"), 
                HumanMessage("What is 1+1?"),
                ]
    print(OmegaConf.to_yaml(cfg))
    model = instantiate(cfg.model)
    print("model name:", model.model_name)
    print(model.half_precision)
    printc(model(messages))

if __name__ == "__main__":
    main()
