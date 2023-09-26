import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from model_utils import SystemMessage, HumanMessage
from logger import get_logger
from utils import printc

#os.environ['HYDRA_FULL_ERROR'] = '1'

log = get_logger()

@hydra.main(version_base = None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    messages = [SystemMessage("You're a nice assistant"), 
                HumanMessage("What is 1+1?"),
                ]
    log.debug("Initializing model")
    model = instantiate(cfg.model)
    log.debug(f"model name: {model.model_name}")

    _ = model(messages)
    log.debug(f"History:\n{model.get_history_str()}")

if __name__ == "__main__":
    main()
