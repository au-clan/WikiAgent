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
    # messages = [SystemMessage("You're a nice assistant"), 
    #             HumanMessage("How are you today?"),
    #             ]
    messages = [SystemMessage("You're a rude assistant"), 
                HumanMessage("How are you today?"),
                ]
    model = instantiate(cfg.model)
    def change_message_roles(messages: list[BaseMessage]):
        new_messages = []
        for message in messages:
            if message.role == 'system':
                new_messages.append(SystemMessage(message.content))
            elif message.role == 'user':
                new_messages.append(AIMessage(message.content))
            else:
                new_messages.append(HumanMessage(message.content))
        return new_messages

    for i in range(10):
        new_message = model(messages)
        messages.append(AIMessage(new_message.content))
        messages = change_message_roles(messages)

        pprint(messages)
        printc(model.get_history_str(), c='yellow')
   
    #log.debug(f"History:\n{model.get_history_str()}")

if __name__ == "__main__":
    main()
