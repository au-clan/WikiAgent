import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import os
import pandas as pd
from pprint import pprint
from time import time
import asyncio

from model_utils import SystemMessage, HumanMessage, AIMessage, AIMessagePrefix, BaseMessage, printc_prompt
from logger import get_logger
from utils import printc, load_tsv_to_dict_list
from environments import LinkWikiEnviroment
from agents import LinkClickAgent
from batch_handler import BatchHandler
from models.huggingface_model import preprocess_llama_chat

os.environ['HYDRA_FULL_ERROR'] = '1'

#log = get_logger()

@hydra.main(version_base = None, config_path="configs", config_name="config")
def batching(cfg: DictConfig):
    model = instantiate(cfg.model)
    start = time()
    messages = [
        [SystemMessage("You are a nice assistant!"), HumanMessage("Hello!")],
        [SystemMessage("You are a mean assistant!"), HumanMessage("Hello!")],
    ]
    responses = model(messages)
    pprint(responses)
    printc(f"Time taken: {time() - start} seconds", c="yellow")

###
async def batch_handler(cfg: DictConfig):
    env = LinkWikiEnviroment("Andy Warhol")
    model = instantiate(cfg.model)

    handler = BatchHandler(model=model, batch_size=1) ###
    num_agents = 3
    agents = [instantiate(cfg.agent) for i in range(num_agents)]
    for i, agent in enumerate(agents):
        agent.set_environment(env)
        agent.set_handler(handler)
        agent.set_target_page("Portugal")
        agent.set_agent_id(i)

    agent_tasks = [asyncio.create_task(agent.run()) for agent in agents]
    batch_processor = asyncio.create_task(handler.process_batch())

    try:
        results = await asyncio.gather(*agent_tasks, batch_processor, return_exceptions=False)
        batch_processor.cancel()
        for result in results:
            print("result:",result)
    except Exception as e:
        raise e

@hydra.main(version_base = None, config_path="configs", config_name="config")
def hydra_batch_handler(cfg: DictConfig):
    """ Do this bc the hydra decorator doesn't work with async functions."""
    asyncio.run(batch_handler(cfg))

def llama_multiturn_prompt():
    # model = instantiate(cfg.model)
    message = [SystemMessage("sys:You are a nice assistant!"), 
               HumanMessage("u:Hello!"), 
               AIMessage("ai:How are you?"),
               HumanMessage("u:I'm good!"),
               AIMessagePrefix("ai_p:That is"),
               AIMessage("ai: good!"),
    ]
    messages = [m.prepare_for_generation() for m in message]
    print(preprocess_llama_chat(messages))

async def game_environment(cfg: DictConfig):
    env = LinkWikiEnviroment(cfg.experiment.src)
    agent = instantiate(cfg.agent)
    agent.set_environment(env)
    agent.set_target_page(cfg.experiment.tgt)
    agent.set_agent_id(0)

    # for debugging, i want to manually step through the environment
    # and choose the next action
    while not agent.is_finished()[0]:
        env_data = agent.observe_environment()
        printc(f"Current page: {env_data['src']}", c="y")
        printc(f"\nLinks: {env_data['links']}", c="b")
        printc(f"\nContent: {env.content}", c="g")
        printc(f"\nPage history: {';'.join(env.page_history)}", c="r")
        printc(f"\nFull page history: {';'.join(env.full_page_history)}", c="r")
        printc(f"\nContent history: {env.content_history}", c="r")
        printc(f"Agent prompt:", c="y")
        printc_prompt(agent.build_prompt())

        next_page = input("Next page: ")

        try:
            agent.apply_actions({"next_page": next_page})
        except Exception as e:
            printc(f"Error: {e}", c="r")
            continue
        printc(f"agent.is_finished: {agent.is_finished()}", c='y')
        print("\n\n")
        if next_page == "exit":
            break

async def test_stuff(cfg: DictConfig):
    env = LinkWikiEnviroment(cfg.experiment.src)
    model=None
    model = instantiate(cfg.model) if not cfg.experiment.debug_mode else None
    handler = BatchHandler(batch_size=1, debug_mode=cfg.experiment.debug_mode, model=model)
    agent = instantiate(cfg.agent)
    agent.set_environment(env)
    agent.set_target_page(cfg.experiment.tgt)
    agent.set_handler(handler)
    agent.set_agent_id(0)

    agent_tasks = [asyncio.create_task(agent.run(debug=False))]
    batch_processor = asyncio.create_task(handler.process_batch())
    
    try:
        results = await asyncio.gather(*agent_tasks, batch_processor, return_exceptions=False)
        batch_processor.cancel()
        for result in results:
            print("result:",result)
    except Exception as e:
        raise e

def hydra_wrapper(func):
    @hydra.main(version_base = None, config_path="configs", config_name="config")
    def wrapper(cfg: DictConfig):
        asyncio.run(func(cfg))
    wrapper()

if __name__ == "__main__":
    #llama_multiturn_prompt()
    hydra_wrapper(test_stuff)