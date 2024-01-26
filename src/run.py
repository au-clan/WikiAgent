import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import os
from time import time
import asyncio
import torch
import sys
import lmql
from lmql.runtime.bopenai import get_stats

from logger import get_logger
from utils import get_paths_real_sample
from agents import *
import nest_asyncio
import logging
from transformers import AutoTokenizer, BitsAndBytesConfig


os.environ['HYDRA_FULL_ERROR'] = '1'

#log = get_logger()

async def main(cfg: DictConfig):
    start = time()
    if cfg.model.lmql['dtype'].lower() == "none":
        # remove from dict
        del cfg.model.lmql.dtype
    
    if "gpt-3.5" not in cfg.model.model_name and "gpt-4" not in cfg.model.model_name:
        model_name = f"local:{cfg.model['model_path']}" if cfg.model['model_path'] else f"local:{cfg.model['model_name']}"
        model = lmql.model(model_name, 
                                cuda=True if cfg.model['device']=='cuda' else False,
                                **cfg.model.lmql
                                )
        tokenizer = AutoTokenizer.from_pretrained(cfg.model['model_name'])
    else: #openai model
        model = None
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    paths_real = get_paths_real_sample(n=cfg.experiment.num_agents, paths_real_file=cfg.paths_real_file, experiment=cfg.experiment.experiment, agent_batch_size=cfg.experiment.agent_batch_size, agent_batch_id=cfg.experiment.agent_batch_id)
    agents = []

    paths_real = [{'src': 'Theatre', 'tgt': 'Zebra'}, {'src': 'Theatre', 'tgt': 'Portugal'}] ###
    for i,path in enumerate(paths_real):
        if cfg.experiment.override:
            path['src'] = cfg.experiment.src; path['tgt'] = cfg.experiment.tgt ###
        env_kwargs = {"current_page": path["src"], "links_file": cfg.links_file, "pages_path": cfg.pages_folder}
        agent = instantiate(cfg.agent)
        agent.model_cfg=cfg.model
        agent.tokenizer = tokenizer
        agent.model=model
        agent.agent_id = i + cfg.experiment.agent_batch_id * cfg.experiment.agent_batch_size
        agent.initialize_environment(**env_kwargs)
        agent.set_target_page(path["tgt"])
        agents.append(agent)
    
    print(f"Agent 0:")
    print(f"Links seen: {id(agents[0].environment.links_seen_on_page)}\nHeadings visited: {id(agents[0].environment.visited_headings_on_page)}")
    print(f"\nEnvironment: {id(agents[0].environment)}\nNavigator: {id(agents[0].environment.nav)}")
    print(f"Visited links: {id(agents[0].environment.visited_links)}\n")
    print(f"Agent 1:")
    print(f"Links seen: {id(agents[1].environment.links_seen_on_page)}\nHeadings visited: {id(agents[1].environment.visited_headings_on_page)}")
    print(f"\nEnvironment: {id(agents[1].environment)}\nNavigator: {id(agents[1].environment.nav)}")
    print(f"Visited links: {id(agents[1].environment.visited_links)}\n")

    # in case we want to select a specific agent
    if cfg.experiment.agent_id >= 0:
        agents = [agents[cfg.experiment.agent_id]]
    
    # restarts unfinished agents from agent_runs_df.tsv file
    if cfg.restart_from_agents_file != "" and cfg.restart_from_agents_file != None:
        # load agents from file
        agent_runs_df = pd.read_csv(cfg.restart_from_agents_file, sep='\t')
        agent_ids_list = agent_runs_df.groupby('agent_id').agg({'num_actions': 'last', 'is_finished_reason': 'last'}).reset_index().query('is_finished_reason == "none"')['agent_id'].tolist()
        agents = [agents[agent_id - cfg.experiment.agent_batch_id * cfg.experiment.agent_batch_size] for agent_id in agent_ids_list]


    results = []
    
    if "agent_batch_size" in cfg.experiment and cfg.experiment.agent_batch_size > 0:
        for i in range(0, len(agents), cfg.experiment.agent_batch_size):
            agent_batches = agents[i:i+cfg.experiment.agent_batch_size]
            if "GPT" in str(cfg.agent._target_):
                # break down batches into ones of 100 agents to avoid timeouts
                agent_batches_gpt = [agent_batches[i:i+100] for i in range(0, len(agent_batches), 100)]
                agent_tasks = []
                for agent_batch_gpt in agent_batches_gpt:
                    agent_tasks += [asyncio.create_task(agent.run()) for agent in agent_batch_gpt]
                    try:
                        results += await asyncio.gather(*agent_tasks, return_exceptions=False)
                        # cancel all tasks
                        for task in agent_tasks:
                            task.cancel()
                    except Exception as e:
                        raise e
            else:
                agent_tasks = [asyncio.create_task(agent.run()) for agent in agent_batches]
                try:
                    results += await asyncio.gather(*agent_tasks, return_exceptions=False)
                    # cancel all tasks
                    for task in agent_tasks:
                        task.cancel()
                except Exception as e:
                    raise e
    else:
        agent_tasks = [asyncio.create_task(agent.run()) for agent in agents]
        try:
            results = await asyncio.gather(*agent_tasks, return_exceptions=False)
            # cancel all tasks
            for task in agent_tasks:
                task.cancel()
        except Exception as e:
            raise e
        
    # re-merge agents list
    print(f"Run finished with {results.count(True)} successes and {results.count(False)} failures.")
    print(f"Total time taken: {time() - start:.1f} seconds")
    print(f"Average time per model call: {(time() - start) / sum([agent.get_num_calls() for agent in agents]):.2f}s for {sum([agent.get_num_calls() for agent in agents])} calls")
    
    if str(cfg.agent._target_) == 'agents.GPTWikiTextAgent':
        total_prompt_tokens = sum([agent.used_prompt_tokens for agent in agents])
        total_completion_tokens = sum([agent.used_completion_tokens for agent in agents])
        model_costs = {"gpt-3.5-turbo-1106": [0.001/1000,0.002/1000], "gpt-4-1106-preview":[0.01/1000,0.03/1000]}
        print(f"Total prompt tokens: {total_prompt_tokens}")
        print(f"Total completion tokens: {total_completion_tokens}")
        print(f"Total cost: {total_prompt_tokens*model_costs[cfg.model.model_name][0] + total_completion_tokens*model_costs[cfg.model.model_name][1]:.2f}$")

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