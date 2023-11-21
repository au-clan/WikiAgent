import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import os
import pandas as pd
from pprint import pprint
from time import time
import asyncio
import torch
import lmql
from model_utils import SystemMessage, HumanMessage, AIMessage, BaseMessage
from logger import get_logger
from utils import printc, load_tsv_to_dict_list, get_paths_real_sample
from environments import LinkWikiEnviroment, WebNavigatorHTML, WebWikiEnvironment
from agents import *
from constrained_agents import *
from batch_handler import BatchHandler
import nest_asyncio
from selenium import webdriver


os.environ['HYDRA_FULL_ERROR'] = '1'

#log = get_logger()

async def main(cfg: DictConfig):
    start = time()
    model = lmql.model(f"local:{cfg.model['model_path']}" if cfg.model['model_path'] else f"local:{cfg.model['model_name']}", 
                        cuda=True if cfg.model['device']=='cuda' else False, 
                        **cfg.model.lmql
                        )
    paths_real = get_paths_real_sample(n=cfg.experiment.num_agents, paths_real_file=cfg.paths_real_file)
    agents = []
    for i,path in enumerate(paths_real):
        # path['src'] = cfg.experiment.src; path['tgt'] = cfg.experiment.tgt ###
        env_kwargs = {"current_page": path["src"], "links_file": cfg.links_file}
        agent = instantiate(cfg.agent)
        agent.initialize_environment(**env_kwargs)
        agent.model_cfg=cfg.model
        agent.model=model
        agent.agent_id = i
        agent.set_tokenizer()
        agent.set_target_page(path["tgt"])
        agents.append(agent)

    agent_tasks = [asyncio.create_task(agent.run()) for agent in agents]

    try:
        results = await asyncio.gather(*agent_tasks, return_exceptions=False)
        # cancel all tasks
        for task in agent_tasks:
            task.cancel()
        print(f"Run finished with {results.count(True)} successes and {results.count(False)} failures.")
    except Exception as e:
        raise e
    torch.cuda.empty_cache()
    print(f"Total time taken: {time() - start:.1f} seconds")
    print(f"Average time per action: {(time() - start) / sum([agent.num_actions for agent in agents]):.2f}s for {sum([agent.num_actions for agent in agents])} actions")

def hydra_wrapper(func):
    @hydra.main(version_base = None, config_path="configs", config_name="config")
    def wrapper(cfg: DictConfig):
        asyncio.run(func(cfg))
    wrapper()

if __name__ == "__main__":
    # we need nest_asyncio to run nested async loops with lmql
    nest_asyncio.apply()
    hydra_wrapper(main)

#nohup python src/run.py &
#nohup python src/run.py --multirun agent=link_click_agent,link_click_dfs_agent,link_click_dfs_back_agent agent.fewshot=0,5 run_name=agent_sweep &