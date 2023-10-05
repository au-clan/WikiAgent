# Base Agent class and subclasses
# Some notes:
# - Prompt = a list of BaseMessages
# - Template = a string with placeholders for the agent to fill in
import attr
from attr import define, field
import sys
import yaml
import asyncio
from model_utils import SystemMessage, HumanMessage, AIMessage, BaseMessage
from environments import Environment
from batch_handler import BatchHandler
from utils import load_tsv_to_dict_list

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

    # Filled in templates
    prompt: list[BaseMessage] = field(factory=list)
    agent_description: str = field(factory=str)
    user_template: str = field(factory=str)

    fewshot: int = field(default=5)
    demonstrations_file: str = field(default=None)

    def __attrs_post_init__(self):
        self.prompt_template = self._load_from_template_file(self.base_template_file, 'prompt')

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
        user_message = HumanMessage(role='user', content = user_message_content)

        prompt = [system_message, user_message]
        return prompt
        
    def process_response(self, response: BaseMessage):
        """ Decide what to do based on the model output."""
        raise NotImplementedError("Subclasses must implement the process_response method.")

    async def act(self):
        """ Main logic for the agent."""
        prompt = self.build_prompt()
        response = await self.handler.send_prompt(self.agent_id, prompt)
        processed_response = self.process_response(response)
        return processed_response


@define
class WikiNavAgent(Agent):
    """
    Virtual class for agents that navigate Wikipedia with a target page.
    """
    target_page: str = field(default=None)

    def set_target_page(self, target_page: str):
        self.target_page = target_page

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
        return dicts

    def observe_environment(self):
        observed_data = {
            "src": self.environment.current_page,
            "tgt": self.target_page,
            "links": self.environment.links,
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
            print(e)
            return False
        

    def process_response(self, response: BaseMessage):
        return response.content
