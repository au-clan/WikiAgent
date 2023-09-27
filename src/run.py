import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from model_utils import SystemMessage, HumanMessage, AIMessage, BaseMessage
from logger import get_logger
from utils import printc
from pprint import pprint

#os.environ['HYDRA_FULL_ERROR'] = '1'

log = get_logger()

@hydra.main(version_base = None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    messages = [SystemMessage("You're a nice assistant"), 
                HumanMessage("How are you today?"),
                ]
    model = instantiate(cfg.model)
    _ = model(messages)
    log.debug(f"History:\n{model.get_history_str()}")

if __name__ == "__main__":
    main()
