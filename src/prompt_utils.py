# Define functions for prompt creation with few-shot learning
# A prompt is basically a list of BaseMessage objects
from attr import define
import sys 
sys.path.append("src/")
from model_utils import ChatModel, AIMessage, SystemMessage, HumanMessage

# Link Only Navigation
def get_navigate_entire_path_prompt(system_prompt: str, user_prompt: str, src: str, tgt:str, path_demonstrations:list, fewshot=5, n=1):
    pass

def get_navigate_with_click_prompt(system_prompt: str, user_prompt: str, src: str, tgt:str, click_demonstrations:list, fewshot=5, n=1):
    # backtracking to be implemented in the prompt
    pass