import sys
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import os
import pandas as pd
from pprint import pprint
from time import time
import asyncio
sys.path.append('../')

from model_utils import *
from logger import get_logger
from utils import *
from environments import LinkWikiEnviroment, WebWikiEnvironment, TextWikiEnvironment
from agents import *
from batch_handler import BatchHandler
from models.huggingface_model import *
from navigator import *
import lmql
from selenium import webdriver

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
    print("Raw prompt:")
    pprint(model.raw_prompt)
    for response in responses:
        print(response)
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

def mistral_multiturn_prompt():
    # model = instantiate(cfg.model)
    message = [SystemMessage("sys:You are a nice assistant!"), 
               HumanMessage("u:Hello!"), 
               AIMessage("ai:How are you?"),
               HumanMessage("u:I'm good!"),
               AIMessagePrefix("ai_p:That is"),
               AIMessage("ai: good!"),
    ]
    messages = [m.prepare_for_generation() for m in message]
    print(preprocess_mistral_chat(messages))

def llama_completion_prompt():
    # model = instantiate(cfg.model)
    messages = [SystemMessage("Instructions"), 
               HumanMessage("Hello!"), 
               AIMessage("How are you?"),
               HumanMessage("I'm good!"),
               AIMessagePrefix("That is"),
               AIMessageCompletion(" good!"),
    ]

    message = [m.prepare_for_completion() for m in messages]
    print(message)
    message = ''.join(message)
    print(message)

async def game_environment(cfg: DictConfig):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')  # Necessary for running in headless mode on some systems
    options.add_argument('--no-sandbox') 
    driver = webdriver.Chrome(options=options)

    nav = WebNavigatorHTML(driver, pages=[cfg.experiment.src], downloads_folder="", markdown=False)
    env = WebWikiEnvironment(nav=nav, current_page=cfg.experiment.src, html=True)

    agent = instantiate(cfg.agent)
    print(agent.action_list)
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

        actions = input("Actions: ")

        try:
            actions = agent.extract_actions(AIMessage(actions))
            agent.apply_actions(actions)
        except Exception as e:
            printc(f"Error: {e}", c="r")
            continue
        printc(f"agent.is_finished: {agent.is_finished()}", c='y')
        print("\n\n")

async def test_run(cfg: DictConfig):
    start = time()
    model = lmql.model(f"local:{cfg.model['model_path']}" if cfg.model['model_path'] else f"local:{cfg.model['model_name']}", 
                        cuda=True if cfg.model['device']=='cuda' else False, 
                        **cfg.model.lmql
                        )
    paths_real = get_paths_real_sample(n=cfg.experiment.num_agents, paths_real_file=cfg.paths_real_file)
    agents = []
    for i,path in enumerate(paths_real):
        # driver = webdriver.Chrome(options=options)
        # nav = WebNavigatorHTML(driver, pages=[path['src']], downloads_folder="")
        # env = WebWikiEnvironment(nav=nav, current_page=path['src'])
        env_kwargs = {"current_page": path["src"], 
                      "links_file": cfg.links_file}
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

def hydra_wrapper_noasync(func):
    @hydra.main(version_base = None, config_path="configs", config_name="config")
    def wrapper(cfg: DictConfig):
        func(cfg)
    wrapper()

def hydra_wrapper(func):
    @hydra.main(version_base = None, config_path="configs", config_name="config")
    def wrapper(cfg: DictConfig):
        asyncio.run(func(cfg))
    wrapper()

async def lmql_test(cfg: DictConfig):
    model_cfg = cfg.model
    prompt  = '''hello \\n\\n \\n\\n Hi''' 
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    printc(f"Prompt length: {len(tokenizer.encode(prompt))}", c="y")
    # substitute [ and ] with [[ and ]] to avoid lmql formatting
    prompt = re.sub(r'(?<!\[)\[(?!\[)', '[[', prompt)
    prompt = re.sub(r'(?<!\])\](?!\])', ']]', prompt)
    model = lmql.model(f"local:{cfg.model['model_path']}" if cfg.model['model_path'] else f"local:{cfg.model['model_name']}", 
                        cuda=True if cfg.model['device']=='cuda' else False, 
                        **cfg.model.lmql
                        )
    lmql_query_str = '''lmql
sample(n=1, temperature={model_cfg[temperature]}, top_p={model_cfg[top_p]}, max_len=4096)
"""{prompt}"""
"""[CONTINUE]""" where len(TOKENS(CONTINUE)) < 10
'''
    lmql_query_str = lmql_query_str.format(prompt=prompt, model_cfg=model_cfg)
    query = lmql.query(lmql_query_str)
    start = time()
    printc(query.lmql_code, c="p")
    response = await query(model=model)
    printc(f"Prompt: {[prompt]}", c="y")
    printc(response, c="g")
    print("Time:", time() - start)

async def lmql_agent_test(cfg: DictConfig):
    env = LinkWikiEnviroment(cfg.experiment.src, links_file=cfg.links_file)
    agent_config = dict(cfg.agent)
    agent_config.pop("_target_")
    agent = LMQLLinkClickDFSAgent(**agent_config, model_cfg=cfg.model)
    agent.fewshot=1
    agent.set_environment(env)
    agent.set_target_page(cfg.experiment.tgt)
    agent.set_agent_id(0)

    agent_tasks = [asyncio.create_task(agent.run(debug=False))]
    
    try:
        results = await asyncio.gather(*agent_tasks, return_exceptions=False)
        for result in results:
            print("result:",result)
    except Exception as e:
        raise e
    
    
    ## agent.act() ##
    prompt = preprocess_llama_chat(agent.build_prompt())
    print(prompt)
    links = env.links
    start = time()
    response = await agent.lmql_query(model=agent.model, prompt=prompt, candidates=links)    
    print(response)
    print("Time:", time() - start)

def navigate_web(cfg: DictConfig):
    current = "Euro"
    downloads_folder = ""
    print(formatPageURL(current))
    print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')  # Necessary for running in headless mode on some systems
    options.add_argument('--no-sandbox') 

    driver = webdriver.Chrome(options=options)

    nav = WebNavigatorHTML(driver, pages=[current], downloads_folder=downloads_folder, markdown=True)
    env = WebWikiEnvironment(nav=nav, current_page=current, html=True)

    while True:
        printc(env.content, c="g")
        printc(env.links, c="b")
        # printc(env.links, c="b")

        action=input("Action: ")
        try:
            if action=="moveDown":
                env.move("down")
            elif action=="moveUp":
                env.move("up")
            elif action=="back":
                env.back()
            elif action=="click":
                input_link = input("Link: ")
                env.change_page(input_link)
        except Exception as e:
            printc(f"Error: {e}", c="r")
            continue
    # print(nav.getHTML())
    # nav.moveDown()
    # print(nav.getHTML())

def navigate_TextWikiEnvironment(cfg: DictConfig):
    current = "Eukaryote"
    nav = NavigatorHTML(pages_path=cfg.pages_folder, model_name=cfg.model.model_name)
    nav.set_curr_doc(current, window_start=0)
    env = TextWikiEnvironment(nav=nav)

    print("All links:", env.get_links())
    printc(f"Current page:{env.get_current_page()}", c="g")
    printc(f"Current seen links:{env.get_clickable_links()}", c="b")
    printc(f"Current content:{env.get_content()}", c="y")
    printc(f"Current headings:{env.get_headings()}", c="r")


    #interact with the environment

    ###TODO: going back twice doesn't work
    while True:
        action = input("Action: ")
        if action=="moveDown":
            env.move("down")
        elif action=="moveUp":
            env.move("up")
        elif action=="back":
            env.back()
        elif action=="click":
            input_link = input("Link: ")
            env.change_page(input_link)
        elif action=="moveHeading":
            input_heading = input("Heading: ")
            env.move(input_heading)

        print("\n\n")
        printc(f"Current page:{env.get_current_page()}", c="g")
        printc(f"Current seen links:{env.get_clickable_links()}", c="b")
        printc(f"Current content:{env.get_content()}", c="y")
        printc(f"Current headings:{env.get_headings()}", c="r")


def chat_template():
    chat = [#SystemMessage("sys:You are a nice assistant!"), 
            HumanMessage("u:Hello!"), 
            AIMessage("ai:How are you?"),
            HumanMessage("u:I'm good!"),
            AIMessage("ai: good!"),
    ]
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    print(tokenizer)
    print(tokenizer.apply_chat_template(chat, tokenize=False))

if __name__ == "__main__":
    hydra_wrapper(navigate_TextWikiEnvironment)
    # chat_template()