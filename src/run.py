import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from model_utils import SystemMessage, HumanMessage

@hydra.main(version_base = None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    messages = [SystemMessage("You're a nice assistant"), HumanMessage("Hey my name is Mariama! How are you?")]

    model = instantiate(cfg.model_name)
    print(model(messages))
    

if __name__ == "__main__":
    main()
