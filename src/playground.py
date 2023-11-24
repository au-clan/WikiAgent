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
from environments import LinkWikiEnviroment, WebWikiEnvironment
from agents import *
from deprecated.constrained_agents import *
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
    prompt  = '''<|im_start|>system
You are a helpful assistant that helps the User complete tasks.<|im_end|>
<|im_start|>user
I'm playing a game with the objective of getting from a source Wikipedia article to a target article in the least amount of clicks.
I can only click on articles I've already seen by scrolling the page, and the clickable articles are formatted like this: [article](article_link).
I can also scroll down the page to see more content and links, or go back to the previous page.
Always respond to me in the format:
Thought:
[your thought process in one line. Example: "We want to get closer to the target page Einstein from the source page Science. Einstein was a famous physicist, and Physics is a subset of Science. There is a link to Physics on the page, so we should click on it."]
Action:
[action I should take. Example: "click(Physics)"]
If an error is generated by a response, the error will be shown to you and the response must be re-written accordingly. 
Here's what I see on the page Saltasaurus:
[TOP OF WINDOW]
_Neuquensaurus_ . The [fossils](Fossil) of _Saltasaurus_ include vertebrae, limb bones and several jaw bones — plus various pieces of armour. Some of these plates appear to have spikes as well, but there is not enough evidence available to be certain. 
##  Changing perceptions 
In the [Cretaceous](Cretaceous) Period, sauropod dinosaurs in [North America](North_America) were losing the survival game to duck-billed dinosaurs, such as _Edmontosaurus_ . However, like modern [Australia](Australia) , [South America](South_America) was an island [continent](Continent) and life evolved rather differently there. Specifically, the duck-billed dinosaurs never gained a foothold and so sauropods, specifically the titanosaurids, continued down their own path of [evolution](Evolution) . ( _See also:_ allopatric speciation.) 
_Saltasaurus_ was one such highly-evolved sauropod and lived 70 to 65 million years ago. When it was first discovered, in 1980, it forced palaeontologists to reconsider many of their assumptions about what was and what was not a sauropod because _Saltasaurus_ , although clearly a sauropod, had armour plating. Previously, it had been assumed that size alone was sufficient defence for the massive sauropods. 
However, a _Saltasaurus_ was discovered covered with bony knobs, 10 to 12 centimetres (4 to 5 inches) in diameter. Since then, palaeontologists have investigated the possibility that other sauropods may also have had armour; for example, the [Argentinian](Argentina) _Laplatasaurus_ . 
##  Eggs? 
A large titanosaur nesting ground was discovered in 1997, by Luis Chiappe and his team, near Auca Mahuevo, in Patagonia, Argentina. The small eggs, about 11 to 12 centimetres (4 to 5 inches) in diameter, contained fossilised embryos, complete with skin impressions (although there was no indication of feathers or dermal spines). These eggs may have belonged to _Saltasaurus_ . 
Apparently several hundred females dug holes, laid
[BOTTOM OF WINDOW]
I'm on the Wikipedia page for Saltasaurus and I want to get to the page for Battle_of_Stalingrad in the least amount of clicks. The article links I've already seen in this article and that I can click on are: Triassic;Cretaceous;Tyrannosaurus;Scientific_classification;Animal;Chordate;Reptile;Saurischia;Sauropodomorpha;Theropoda;Ornithischia;Triassic;Cretaceous;Scientific_classification;Animal;Chordate;Reptile;Theropoda;Sauropodomorpha;Theropoda;Sauropodomorpha;Tyrannosaurus;Dinosaur;Ornithischia;Carnivore;Theropoda;Herbivore;Sauropodomorpha;Cretaceous;Cretaceous-Tertiary_extinction_event;Bird;Bird;Dinosaur;Sauropodomorpha;Theropoda;Bird;Dinosaur;Sauropodomorpha;Theropoda;Bird;Dinosaur;Sauropodomorpha;Theropoda;Fossil;Scientific_classification;Animal;Chordate;Reptile;Saurischia;Sauropodomorpha;Binomial_nomenclature;Animal;Chordate;Reptile;Saurischia;Sauropodomorpha;Binomial_nomenclature;Dinosaur;Cretaceous;Herbivore;Uruguay;Fossil;Fossil;Cretaceous;North_America;Australia;South_America;Continent;Evolution;Argentina.
I have 1 scrolls left on this page. I can either: click on a link that will get me closer to the target, scroll down the page, or go back to the previous page: none. First tell me your thought process and then tell me what to do. Respond with "back", "click(article_link)", or "scroll_down".<|im_end|>
<|im_start|>assistant
Thought:
We want to find a link that leads us to the Battle_of_Stalingrad page. Since the page is about a dinosaur and the target is a historical event, we need to look for connections between the two topics. 
Action:
click(Cretaceous-Tertiary_extinction_event)<|im_end|>
<|im_start|>assistant
Thought:
We want to find a link that leads us to the Battle_of_Stalingrad page. Since the page is about a dinosaur and the target is a historical event, we need to look for connections between the two topics. 
Action:
click(Cretaceous-Tertiary_extinction_event)<|im_end|>
<|im_start|>user
Link Cretaceous-Tertiary_extinction_event not on current page. Try a different page.<|im_end|>
<|im_start|>assistant
Thought:
We want to find a link that leads us to the Battle_of_Stalingrad page. Since the page is about a dinosaur and the target is a historical event, we need to look for connections between the two topics.
Action:
click(Cretaceous-Tertiary_extinction_event)''' 
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
"""{prompt} [CONTINUE]"""
'''
    lmql_query_str = lmql_query_str.format(prompt=prompt, model_cfg=model_cfg)
    printc(lmql_query_str, c="p")
    query = lmql.query(lmql_query_str)
    start = time()
    
    response = await query(model=model)
    print("query response:", response)
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
    hydra_wrapper(lmql_test)
    # chat_template()