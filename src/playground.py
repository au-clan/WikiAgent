import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import os
import pandas as pd
from pprint import pprint
from time import time
import asyncio

from model_utils import *
from logger import get_logger
from utils import printc, load_tsv_to_dict_list
from environments import LinkWikiEnviroment
from agents import LinkClickAgent
from batch_handler import BatchHandler
from models.huggingface_model import *
from navigator import NavigatorHTML
import torch

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
    env = LinkWikiEnviroment(cfg.experiment.src, links_file=cfg.links_file)
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

async def test_run(cfg: DictConfig):
    env = LinkWikiEnviroment(cfg.experiment.src, links_file=cfg.links_file)
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

def navigate(cfg: DictConfig):
    nav = NavigatorHTML(pages_path=cfg.pages_folder, token_window=300, overlap=0.1)
    nav.set_curr_doc("United_States")
    print(nav.window_text)

def rellm_test():
    import regex
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rellm_new import complete_re
    model_name = "/dlabdata1/llama2_hf/Llama-2-7b-hf/" #"gpt2" # 
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if "llama" in model_name.lower():
        tokenizer.pad_token = "[PAD]"
        tokenizer.padding_side = "left"
    #torch.set_num_threads(16)

    # message = [SystemMessage("You are a nice assistant."), 
    #             HumanMessage("Is this a good demo?"), 
    #         ]
    # messages = [m.prepare_for_generation() for m in message]
    # prompt = preprocess_llama_chat(messages)
    prompt = """In Wikispeedia, given two Wikipedia articles, a source and a target, your goal is to reach the target article starting from the source article in as few clicks as possible. For the articles given this is always possible. Choose the page that will take the user closer to the target, **only** from the provided candidates.  If no available link takes you closer to the target, select "NA" as the next page.
##Input
You will be given input in the JSON format as described below
Input format:
{"source": "<source article>", "target": "<target article>", "links": "<outgoing links from current article>"}

##Output
You should only respond in the JSON format as described below
Output format:
{"next_page": "<chosen page for user to click on>"}
Ensure the response can be parsed by Python json.loads

Here are some previous conversations between the Assistant and the User:
User: {"source": "Owl", "target": "Penguin", "links": "Ancient_Egypt;Ancient_Rome;Animal;Antarctica;Athena;Aztec;Barn_Owl;Bird;Bird_of_prey;Cretaceous;Dinosaur;Eagle;Ecology;Eurasian_Eagle_Owl;Falcon;Finland;Fish;Fossil;Greek_mythology;Greenland;Hawk;Henry_David_Thoreau;Heron;Insect;Islam;Mammal;Maya_civilization;Middle_East;Paleocene;Roman_mythology;Scientific_classification;Tawny_Owl"}
Assistant: {"next_page": "Antarctica"}
User: {"source": "Jew", "target": "Hebrew_calendar", "links": "Adolf_Hitler;Age_of_Enlightenment;Alexander_the_Great;Algeria;Ancient_Egypt;Ancient_Greece;Arab-Israeli_conflict;Argentina;Asia;Assyria;Australia;Austria;Berlin;Berlin_Wall;Bible;Birth_control;Brazil;Canada;Chile;China;Christianity;Crusades;Culture;Democracy;Denmark;England;English_language;Ethiopia;Ethnic_group;Europe;First_Crusade;France;French_language;Geography;German_language;Germany;Greece;Hebrew_language;Heraclius;India;Iran;Isaac;Islam;Israel;Italy;Jerusalem;Jew;Judaism;Latin;Mexico;Middle_Ages;Middle_East;Morocco;Muhammad;Napoleon_I_of_France;Nazi_Germany;Nazism;Netherlands;North_Africa;Oman;Ottoman_Empire;Poland;Religion;Roman_Empire;Russia;Russian_language;Slavery;Socialism;South_Africa;Soviet_Union;Spain;Spanish_Inquisition;Sweden;Syria;The_Holocaust;Tunisia;Turkey;Ukraine;United_Kingdom;United_States;Uruguay;Venezuela;World_War_II;Yemen;Zeus;Zionism"}
Assistant: {"next_page": "Judaism"}
User: {"source": "Evolution", "target": "Organism", "links": "Agriculture;Allosaurus;Anatomy;Ape;Archaeopteryx;Armadillo;Arthropod;Baboon;Bacteria;Biodiversity;Bioinformatics;Biology;Bird;Birth_control;Cambrian_explosion;Cancer;Cell_(biology);Charles_Darwin;Chimpanzee;Computer;Creation-evolution_controversy;Creationism;Cretaceous-Tertiary_extinction_event;Crystal;DNA;Dinosaur;Ecology;Empiricism;Entropy;Eukaryote;Extinction;Fish;Fossil;Fossil_record;Francis_Crick;Galápagos_Islands;Genetic_code;Genetics;Geology;Glacier_National_Park_(US);Gorilla;Gravitation;Gregor_Mendel;History_of_Earth;Ice_age;James_D._Watson;Judaism;Life;Malaria;Mineral;Mitochondrion;Natural_selection;Organism;Paleontology;Pea;Photosynthesis;Plant;Precambrian;Prion;Protein;Punctuated_equilibrium;Rain;Science;Sociology;South_America;Stephen_Jay_Gould;The_Origin_of_Species;Tiktaalik;United_States;Vertebrate;Vestigial_structure"}
Assistant: {"next_page": "Organism"}
User: {"source": "Great_Britain", "target": "Elizabeth_I_of_England", "links": "Ancient_Greece;Ancient_Rome;Antelope;BBC;Bristol;British_Isles;British_Isles_(terminology);Channel_Islands;Climate;England;English_Channel;Europe;France;French_language;Hebrides;Ice_age;Ireland;Isle_of_Man;Isle_of_Wight;Isles_of_Scilly;James_I_of_England;Northern_Ireland;Olympic_Games;Orkney;Picts;Ptolemy;Republic_of_Ireland;Roman_Britain;Scotland;Scottish_Gaelic_language;Shetland;Ukraine;United_Kingdom;Viking;Wales"}
Assistant: {"next_page": "NA"}
User: {"source": "Thalassemia", "target": "Sanskrit", "links": "American_English;Anemia;Asthma;Bangladesh;British_English;Cambodia;China;Cyprus;Folic_acid;Greece;India;Italy;Kenya;Malaria;Pakistan;Pete_Sampras;Sickle-cell_disease;Thailand"}
Assistant: {"next_page": "India"}
Now continue the following conversation:
User: 
{""source"": ""Flamingo"", ""target"": ""Gold(III)_chloride"", ""links"": ""Animal;Bacteria;Bird;Carolus_Linnaeus;Cholera;Chordate;Cretaceous;Dove;Eocene;Fossil_record;Kenya;Protein;Scientific_classification;Tanzania;Water""}
Assistant:
"""
    pattern = regex.compile(r'\{"next_page": "(Animal|Bacteria|Bird|Cholera|Chordate|Dove|Eocene|Fossil_record|Kenya|Protein|Scientific_classification|Tanzania|Water)"\}')
                        
    print(prompt)
    print("ReLLM:")
    
    start = time()
    output = complete_re(tokenizer=tokenizer, 
                        model=model, 
                        prompt=prompt,
                        pattern=pattern,
                        do_sample=False,
                        max_new_tokens=20,
                        stop_after_match=True,
                        debug=True)
    print(output)
    print("Time per token:", (time() - start) / len(tokenizer.encode(output, return_tensors="pt")[0]))

    print("\nNormal:")
    start = time()
    output = model.generate(tokenizer.encode(prompt, return_tensors="pt").to(model.device),
                            max_new_tokens=20, 
                            pad_token_id=tokenizer.eos_token_id, 
                            do_sample=False)
    print(output[0])
    print(tokenizer.decode(output[0]))
    print("Time per token:", (time() - start) / (len(output[0]) - len(tokenizer.encode(prompt, return_tensors="pt")[0])))

def relm_test():
    import relm
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "gpt2-xl"
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                return_dict_in_generate=True,
                                                pad_token_id=tokenizer.eos_token_id)
    model = model.to(device)
    query_string = relm.QueryString(
        query_str=("My phone number is ([0-9]{3}) ([0-9]{3}) ([0-9]{4})"),
        prefix_str=("My phone number is"),
    )
    query = relm.SimpleSearchQuery(
    query_string=query_string,
        search_strategy=relm.QuerySearchStrategy.SHORTEST_PATH,
        tokenization_strategy=relm.QueryTokenizationStrategy.ALL_TOKENS,
        top_k_sampling=40,
        num_samples=10,
    )
    ret = relm.search(model, tokenizer, query)
    for x in ret:
        print(tokenizer.decode(x))


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

import lmql
def lmql_test():

    prompt  = """<s>[INST] <<SYS>>
Assistant is a expert JSON builder designed to assist the user in playing play the game of Wikispeedia. Assistant is familiar with all Wikipedia pages and can help the user find a path between two Wikipedia articles.
In Wikispeedia, given two Wikipedia articles, a source and a target, your goal is to reach the target article starting from the source article in as few clicks as possible. For the articles given this is always possible. Choose the page that will take the user closer to the target, **only** from the provided candidates. If no available link takes you closer to the target, select "NA" as the next page.

##Input
You will be given input in the JSON format as described below
Input format:
{"source": "<source article>", "target": "<target article>", "links": "<outgoing links from current article>"}

##Output
You should only respond in the JSON format as described below
Output format:
{"next_page": "<chosen page for user to click on>"}
Ensure the response can be parsed by Python json.loads
If an error is generated by a response, the error will be shown by the user and the response must be re-written accordingly. 
<</SYS>>

{"source": "Owl", "target": "Penguin", "links": "Ancient_Egypt;Ancient_Rome;Animal;Antarctica;Athena;Aztec;Barn_Owl;Bird;Bird_of_prey;Cretaceous;Dinosaur;Eagle;Ecology;Eurasian_Eagle_Owl;Falcon;Finland;Fish;Fossil;Greek_mythology;Greenland;Hawk;Henry_David_Thoreau;Heron;Insect;Islam;Mammal;Maya_civilization;Middle_East;Paleocene;Roman_mythology;Scientific_classification;Tawny_Owl"} [/INST] {"next_page": "Antarctica"} </s><s>[INST] {"source": "Jew", "target": "Hebrew_calendar", "links": "Adolf_Hitler;Age_of_Enlightenment;Alexander_the_Great;Algeria;Ancient_Egypt;Ancient_Greece;Arab-Israeli_conflict;Argentina;Asia;Assyria;Australia;Austria;Berlin;Berlin_Wall;Bible;Birth_control;Brazil;Canada;Chile;China;Christianity;Crusades;Culture;Democracy;Denmark;England;English_language;Ethiopia;Ethnic_group;Europe;First_Crusade;France;French_language;Geography;German_language;Germany;Greece;Hebrew_language;Heraclius;India;Iran;Isaac;Islam;Israel;Italy;Jerusalem;Jew;Judaism;Latin;Mexico;Middle_Ages;Middle_East;Morocco;Muhammad;Napoleon_I_of_France;Nazi_Germany;Nazism;Netherlands;North_Africa;Oman;Ottoman_Empire;Poland;Religion;Roman_Empire;Russia;Russian_language;Slavery;Socialism;South_Africa;Soviet_Union;Spain;Spanish_Inquisition;Sweden;Syria;The_Holocaust;Tunisia;Turkey;Ukraine;United_Kingdom;United_States;Uruguay;Venezuela;World_War_II;Yemen;Zeus;Zionism"} [/INST] {"next_page": "Judaism"} </s><s>[INST] {"source": "Evolution", "target": "Organism", "links": "Agriculture;Allosaurus;Anatomy;Ape;Archaeopteryx;Armadillo;Arthropod;Baboon;Bacteria;Biodiversity;Bioinformatics;Biology;Bird;Birth_control;Cambrian_explosion;Cancer;Cell_(biology);Charles_Darwin;Chimpanzee;Computer;Creation-evolution_controversy;Creationism;Cretaceous-Tertiary_extinction_event;Crystal;DNA;Dinosaur;Ecology;Empiricism;Entropy;Eukaryote;Extinction;Fish;Fossil;Fossil_record;Francis_Crick;Galápagos_Islands;Genetic_code;Genetics;Geology;Glacier_National_Park_(US);Gorilla;Gravitation;Gregor_Mendel;History_of_Earth;Ice_age;James_D._Watson;Judaism;Life;Malaria;Mineral;Mitochondrion;Natural_selection;Organism;Paleontology;Pea;Photosynthesis;Plant;Precambrian;Prion;Protein;Punctuated_equilibrium;Rain;Science;Sociology;South_America;Stephen_Jay_Gould;The_Origin_of_Species;Tiktaalik;United_States;Vertebrate;Vestigial_structure"} [/INST] {"next_page": "Organism"} </s><s>[INST] {"source": "Great_Britain", "target": "Elizabeth_I_of_England", "links": "Ancient_Greece;Ancient_Rome;Antelope;BBC;Bristol;British_Isles;British_Isles_(terminology);Channel_Islands;Climate;England;English_Channel;Europe;France;French_language;Hebrides;Ice_age;Ireland;Isle_of_Man;Isle_of_Wight;Isles_of_Scilly;James_I_of_England;Northern_Ireland;Olympic_Games;Orkney;Picts;Ptolemy;Republic_of_Ireland;Roman_Britain;Scotland;Scottish_Gaelic_language;Shetland;Ukraine;United_Kingdom;Viking;Wales"} [/INST] {"next_page": "NA"} </s><s>[INST] {"source": "Thalassemia", "target": "Sanskrit", "links": "American_English;Anemia;Asthma;Bangladesh;British_English;Cambodia;China;Cyprus;Folic_acid;Greece;India;Italy;Kenya;Malaria;Pakistan;Pete_Sampras;Sickle-cell_disease;Thailand"} [/INST] {"next_page": "India"} </s><s>[INST] {"source": "Roman_Empire", "target": "Sparrowhawk", "links": "3rd_century;Ancient_Rome;Architecture;Armenia;Augustus;Autocracy;Charlemagne;Christianity;Civilization;Claudius;Cleopatra_VII;Currency;Danube;Eastern_Orthodox_Church;Edward_Gibbon;Elagabalus;Europe;Franks;Hadrian's_Wall;Heraclius;Holy_Roman_Empire;Hungary;Istanbul;Italy;Jerusalem;Julius_Caesar;Latin;List_of_countries_by_system_of_government;Lyon;Maghreb;Mark_Antony;Mesopotamia;Middle_Ages;Milan;Monarchy;Ottoman_Empire;Paris;Persian_Empire;Pompeii;Rhine;Roman_Britain;Roman_law;Romulus_Augustus;Sculpture;Spain;United_States;Western_Roman_Empire"} [/INST] "
"""
    start = time()
    choose = lmql.F("{prompt} {{\"next_page\": \"[NEXT_PAGE]\", \"reason\": \"This link will bring the user closer to the target page Sparrowhawk because [REASON]\"}}", '''len(TOKENS(REASON)) < 60 and NEXT_PAGE in set(["3rd_century", "Ancient_Rome", "Architecture", "Armenia", "Augustus", "Autocracy", "Charlemagne", "Christianity", "Civilization", "Claudius", "Cleopatra_VII", "Currency", "Danube", "Eastern_Orthodox_Church", "Edward_Gibbon", "Elagabalus", "Europe", "Franks", "Hadrian's_Wall", "Heraclius", "Holy_Roman_Empire", "Hungary", "Istanbul", "Italy", "Jerusalem", "Julius_Caesar", "Latin", "List_of_countries_by_system_of_government", "Lyon", "Maghreb", "Mark_Antony", "Mesopotamia", "Middle_Ages", "Milan", "Monarchy", "Ottoman_Empire", "Paris", "Persian_Empire", "Pompeii", "Rhine", "Roman_Britain", "Roman_law", "Romulus_Augustus", "Sculpture", "Spain", "United_States", "Western_Roman_Empire"])''',                    
                    model=lmql.model("/dlabdata1/llama2_hf/Llama-2-7b-chat-hf"))
    # choose = lmql.F("{prompt} {{\"reason\": \"[REASON]\", \"next_page\": \"[NEXT_PAGE]\"}}", "len(TOKENS(REASON)) < 60 and STOP_AT(REASON, '.')",model=lmql.model("/dlabdata1/llama2_hf/Llama-2-7b-chat-hf"), max_len=4096)
    print(choose(prompt=prompt))
    print("Time:", time() - start)

if __name__ == "__main__":
    lmql_test()
    # hydra_wrapper(test_run)