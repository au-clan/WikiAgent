import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import os
import pandas as pd
from pprint import pprint
from time import time
import asyncio

from model_utils import SystemMessage, HumanMessage, AIMessage, BaseMessage
from prompt_utils import get_navigate_entire_path_prompt
from logger import get_logger
from utils import printc, load_tsv_to_dict_list
from navigate_links_only import get_path_demonstrations, get_paths_real_sample
from environments import LinkWikiEnviroment
from agents import LinkClickAgent
from batch_handler import BatchHandler

os.environ['HYDRA_FULL_ERROR'] = '1'

#log = get_logger()

async def main(cfg: DictConfig):
    start = time()
    n_paths = 20
    model = instantiate(cfg.model)
    handler = BatchHandler(model=model, batch_size=cfg.model.batch_size)

    #log.debug("Model instantiated.")
    fewshot = 5
    paths_real = get_paths_real_sample(n=n_paths)
    path_demonstrations = get_path_demonstrations()

    agents = []
    envs = []
    for i,path in enumerate(paths_real):
        env = LinkWikiEnviroment(path["src"])
        agent = instantiate(cfg.agent)
        agent.set_environment(env)
        agent.set_handler(handler)
        agent.set_target_page(path["tgt"])
        agent.set_agent_id(i)
        agents.append(agent)
        envs.append(env)

    agent_tasks = [asyncio.create_task(agent.run()) for agent in agents]
    batch_processor = asyncio.create_task(handler.process_batch())

    try:
        results = await asyncio.gather(*agent_tasks, batch_processor, return_exceptions=False)
        batch_processor.cancel()
        for result in results:
            print("result:",result)
    except Exception as e:
        raise e

        

 
    print(f"Total time taken: {time() - start} seconds")
    # model_output_df = pd.concat([model_output_df, pd.DataFrame({'src': path["src"], 'tgt': path["tgt"], 'human_path': path["output"],
    #                                                             'model_output': response, 'history': model.get_history_str(),
    #                                                             'temperature': model.temperature, 'top_p': model.top_p}, index=[0])], ignore_index=True)

@hydra.main(version_base = None, config_path="configs", config_name="config")
def hydra_main(cfg: DictConfig):
    asyncio.run(main(cfg))

if __name__ == "__main__":
    hydra_main()

#nohup python src/run.py &