# Define functions for prompt creation with few-shot learning
# A prompt is basically a list of BaseMessage objects
from attr import define
import sys 
sys.path.append("src/")
from typing import List
from model_utils import BaseMessage, ChatModel, AIMessage, SystemMessage, HumanMessage

# Link Only Navigation
def get_navigate_entire_path_prompt(system_prompt: str, user_prompt: str, 
                                    fewshot_demonstrations_prompt: str,
                                    src: str = "", tgt: str = "", path_demonstrations:list = [], 
                                    fewshot=5, n=1) -> List[BaseMessage]:
    messages = []
    fewshot_demonstrations = ""
    if fewshot>=1:
        for demo in path_demonstrations[0:fewshot]:
            fewshot_demonstrations += fewshot_demonstrations_prompt.format(src=demo["src"], tgt=demo["tgt"], output=demo["output"])
    messages.append(SystemMessage(system_prompt.format(fewshot_demonstrations=fewshot_demonstrations)))
    messages.append(HumanMessage(user_prompt.format(src=src, tgt=tgt)))
                    

    return messages
def get_navigate_with_click_prompt(system_prompt: str, user_prompt: str, 
                                   demonstration_user_prompt: str, demonstration_assistant_prompt: str,
                                   src: str, tgt:str, click_demonstrations:list, 
                                   fewshot=5, n=1):
    # backtracking to be implemented in the prompt
    pass

def wikify_article_name(name):
    return "_".join(name.strip().split(" "))

def un_wikify_article_name(name):
    return " ".join(name.strip().split("_"))