# Base Agent class and subclasses
# Some notes:
# - Prompt = a list of BaseMessages
# - Template = a string with placeholders for the agent to fill in
import attr
from attr import define, field
import sys
import yaml
import asyncio
from model_utils import SystemMessage, HumanMessage, AIMessage, AIMessagePrefix, BaseMessage
from environments import Environment
from batch_handler import BatchHandler
from utils import load_tsv_to_dict_list, printc, extract_json_from_response, un_wikify_article_name, wikify_article_name
from time import time
import pandas as pd
import json

@define
class Agent:
    """
    Virtual agent class. Defines the main interface for the agent.
    """
    environment: Environment = field(default=None)
    handler: BatchHandler = field(default=None)
    agent_id: int = field(default=None)

    # File with templates for the prompt and demonstration structure
    base_template_file: str = field(default="src/configs/agent/base_agent.yaml")
    
    # specific to the agent type
    template_file: str = field(default="src/configs/agent/base_agent.yaml")
    prompt_template: dict = field(factory=dict)
    demonstration_template: str = field(factory=str)
    agent_description: str = field(factory=str)
    user_template: str = field(factory=str)

    fewshot: int = field(default=5)
    demonstrations_file: str = field(default=None)
    prompt: list[BaseMessage] = field(factory=list)

    # data about a run
    num_actions_taken: int = field(default=0)
    run_df = field(factory=pd.DataFrame)


    def __attrs_post_init__(self):
        self.prompt_template = self._load_from_template_file(self.base_template_file, 'prompt')
        if not self.demonstration_template:
            self.demonstration_template = self._load_from_template_file(self.template_file, 'demonstration_template')
        if not self.agent_description:
            self.agent_description = self._load_from_template_file(self.template_file, 'agent_description')
        if not self.user_template:
            self.user_template = self._load_from_template_file(self.template_file, 'user_template')

    def to_dict(self):
        # Create a dictionary manually
        attributes = {}
        for attr in dir(self):
            if not callable(getattr(self, attr)) and not attr.startswith("__"):
                attributes[attr] = getattr(self, attr)
        return attributes
    
    def _load_from_template_file(self, file: str,  field: str):
        with open(file, "r") as f:
            field = yaml.safe_load(f)[field]
        return field

    def set_environment(self, environment: Environment):
        self.environment = environment
    
    def set_handler(self, handler: BatchHandler):
        self.handler = handler
    
    def set_agent_id(self, agent_id: int):
        self.agent_id = agent_id
        
    def get_system_message(self):
        system_message_content = self.prompt_template['system'].format(
            agent_description=self.agent_description,
            fewshot_demonstrations=self.get_fewshot_demonstrations(),
        )
        return SystemMessage(role='system', content = system_message_content)
    
    def get_fewshot_demonstrations(self) -> str:
        demos_list = self._load_fewshot_demonstrations_data()
        fewshot_demonstrations = ""
        for i in range(self.fewshot):
            user_template_filled = self.user_template.format(replace = demos_list[i])
            fewshot_demonstrations += self.demonstration_template.format(user_template = user_template_filled, replace = demos_list[i])
            fewshot_demonstrations += "\n"
        return fewshot_demonstrations

    def _load_fewshot_demonstrations_data(self) -> list[dict]:
        """
        The fewshot demonstrations file are agent type dependent.
        The output dict should have the same keys as the keywords in the user template.
        """
        raise NotImplementedError("Subclasses must implement the _load_fewshot_demonstrations method.")
    
    def observe_environment(self):
        """
        Agent observes the environment.
        The output dict should have the same keys as the keywords in the user template.
        """
        raise NotImplementedError("Subclasses must implement the observe_environment method.")
    
    def build_prompt(self):
        """
        Agent observes the environment and builds a prompt, that will be passed to the model.
        """
        system_message = self.get_system_message()
        env_data = self.observe_environment()

        user_message_content = self.user_template.format(replace = env_data)
        user_message = HumanMessage(content = user_message_content)
        assistant_prefix = AIMessagePrefix(self.prompt_template['assistant_prefix'])

        prompt = [system_message, user_message, assistant_prefix]
        self.prompt = prompt
        return prompt
        
    def extract_actions(self, response: BaseMessage) -> dict:
        """ Processes the model output and returns a dict with the agent's choices."""
        extracted_json = extract_json_from_response(response.content)
        if extracted_json:
            actions = json.loads(extracted_json)
        else:
            actions = {}
        return actions
    
    def apply_actions(self, actions: dict):
        """ Applies the actions to the environment."""
        raise NotImplementedError("Subclasses must implement the apply_actions method.")

    def update_run_df(self):
        """ Depends on the agent type."""
        pass

    async def act(self):
        """ One turn of the agent."""
        printc(f"Agent {self.agent_id} on page {self.environment.current_page}.", c="r")
        start = time()
        prompt = self.build_prompt()
        printc(f"Agent {self.agent_id} says:\n{prompt[-2].content}", c="r")
        response = await self.handler.send_prompt(self.agent_id, prompt)
        printc(f"Agent {self.agent_id} received response in {time() - start} seconds. Response:\n{response}")
        actions = self.extract_actions(response)
        self.apply_actions(actions)
        self.num_actions_taken += 1
        # self.update_run_df(response.content, actions)
    
    def is_finished(self):
        """ Finishing condition is agent type dependent."""
        raise NotImplementedError("Subclasses must implement the is_finished method.")
    
    async def run(self):
        """ Run the agent until game is over or error."""
        while not self.is_finished():
            try:
                await self.act()
            except Exception as e:
                printc(f"Error in agent {self.agent_id}: {e}", c="r")
                break
        if self.is_finished():
            printc(f"Agent {self.agent_id} finished in {self.num_actions_taken} actions.", c="r")
        


@define
class WikiNavAgent(Agent):
    """
    Virtual class for agents that navigate Wikipedia with a target page.
    """
    target_page: str = field(default=None)
    max_steps: int = field(default=10)

    def set_target_page(self, target_page: str):
        self.target_page = wikify_article_name(target_page)
    
    def is_finished(self, force=False):
        if force:
            return True
        else:
            return un_wikify_article_name(self.environment.current_page) == self.target_page or self.num_actions_taken >= self.max_steps
        
    def update_run_df(self, response: str, click: str):
        row = {"agent_id": self.agent_id, 
                "src": self.environment.current_page,
                "tgt": self.target_page,
                "cands": ";".join(self.environment.links),
                "click": click,
                "step_num": self.num_actions_taken,
                "prompt": self.prompt[-2].content,
                "response": response,
        }
        self.run_df = self.run_df.append(row, ignore_index=True)

                

        

@define
class LinkClickAgent(WikiNavAgent):
    """
    Agent that clicks knowing only the list of links in a certain page.
    """
    template_file: str = field(default="src/configs/agent/link_click_agent.yaml")
    fewshot_demonstrations_file: str = field(default="data/wikispeedia/wikispeedia_paths-and-graph/click_demonstrations.tsv")
    
    def _load_fewshot_demonstrations_data(self) -> list[dict]:
        dicts = load_tsv_to_dict_list(self.fewshot_demonstrations_file)
        #rename some keys
        for d in dicts:
            d['links'] = d.pop('cands')
            d['click'] = d.pop('output')

            d['links'] = ';'.join([un_wikify_article_name(link) for link in d['links'].split(';')])
            d['click'] = un_wikify_article_name(d['click'])
        return dicts

    def observe_environment(self):
        observed_data = {
            "src": un_wikify_article_name(self.environment.current_page),
            "tgt": un_wikify_article_name(self.target_page),
            "links": ";".join([un_wikify_article_name(link) for link in self.environment.links]),
        }
        return observed_data

    def click(self, link: str):
        """
        Clicks on a link.
        """
        try:
            self.environment.change_page(link)
            return True
        except AssertionError as e:
            raise e
        

    def apply_actions(self, actions: dict):
        """
        Apply actions from the model.
        """
        assert 'click' in actions, "Invalid actions for LinkClickAgent."
        self.click(actions['click'])

