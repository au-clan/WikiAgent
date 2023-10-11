# Base Agent class and subclasses
# Some notes:
# - Prompt = a list of BaseMessages
# - Template = a string with placeholders for the agent to fill in
import attr
from attr import define, field
import sys
import yaml
import asyncio
from model_utils import SystemMessage, HumanMessage, AIMessage, AIMessagePrefix, BaseMessage, printc_prompt
from environments import Environment
from batch_handler import BatchHandler
from utils import load_tsv_to_dict_list, printc, extract_json_from_response, wikify_article_name
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

    #features that are common to all agents
    use_feedback: bool = field(default=False)
    multiturn_demonstrations: bool = field(default=False)

    # File with templates for the prompt and demonstration structure
    base_template_file: str = field(default="src/configs/agent/base_agent.yaml")
    
    # specific to the agent type
    template_file: str = field(default="src/configs/agent/base_agent.yaml")
    prompt_template: dict = field(factory=dict)
    demonstration_template: str = field(factory=str)
    agent_description: str = field(factory=str)
    user_template: str = field(factory=str)
    assistant_prefix: str = field(factory=str)

    fewshot: int = field(default=5)
    demonstrations_file: str = field(default=None)
    prompt: list[BaseMessage] = field(factory=list)

    # data about a run
    num_actions: int = field(default=0)
    num_errors: int = field(default=0)
    run_df = field(factory=pd.DataFrame)
    user_prompt_history = field(factory=list)
    response_history = field(factory=list)


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
        
    def get_fewshot_demonstrations(self) -> list[str]:
        demos_list = self._load_fewshot_demonstrations_data()
        fewshot_demonstrations = []
        for i in range(self.fewshot):
            user_template_filled = self.user_template.format(replace = demos_list[i])
            fewshot_demonstrations.append(self.demonstration_template.format(user_template = user_template_filled, replace = demos_list[i]))
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
    
    def build_prompt(self, feedback: str = None) -> list[BaseMessage]:
        """
        Agent observes the environment and builds a prompt, that will be passed to the model.
        
        args:
            feedback: str = None
                Feedback from the user.
                TODO: change run() to pass feedback to act(). Right now it retries
        """
        if not feedback:
            if self.multiturn_demonstrations:
                system_message_content = self.agent_description
                demos_list = self._load_fewshot_demonstrations_data()
                
                prompt = [SystemMessage(system_message_content)]
                for i in range(self.fewshot):
                    prompt.append(HumanMessage(self.user_template.format(replace = demos_list[i])))         
                    prompt.append(AIMessage(self.demonstration_template.format(replace = demos_list[i]))) 
            else: # demos on system message 
                system_message_content = self.prompt_template['system'].format(
                    agent_description=self.agent_description,
                    fewshot_demonstrations="\n".join(self.get_fewshot_demonstrations()),
                )
                system_message = SystemMessage(system_message_content)
                prompt = [system_message]
            env_data = self.observe_environment()

            user_message_content = self.user_template.format(replace = env_data)
            user_message = HumanMessage(content = user_message_content)
            assistant_prefix = AIMessagePrefix(self.assistant_prefix)

            prompt.append(user_message); prompt.append(assistant_prefix)
            self.prompt = prompt
        else:
            # append bad response and feedback to the prompt
            self.prompt.append(AIMessage(self.response_history[-1]))
            self.prompt.append(HumanMessage(feedback))
            self.prompt.append(AIMessagePrefix(self.assistant_prefix))
            prompt = self.prompt    
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

    async def act(self, feedback: str = None):
        """ One turn of the agent. -Build prompt; Send prompt to model; Apply actions
        args:
            feedback: str = None
                Feedback from the user. If not None, it means the agent made a mistake and needs to be corrected.
        """
        printc(f"Agent {self.agent_id} on page {self.environment.current_page}.", c="r")
        prompt = self.build_prompt(feedback)

        printc(f"Agent {self.agent_id} says: {prompt[-2].content}")
        # printc_prompt(∂isself.prompt, c='p')

        self.user_prompt_history.append(prompt[-2].content)
        start = time()
        response = await self.handler.send_prompt(self.agent_id, prompt)

        printc(f"Agent {self.agent_id} received response. Time in queue: {time() - start} seconds. Response:\n{response}")
        self.response_history.append(response.content)
        actions = self.extract_actions(response)
        self.apply_actions(actions)

    def is_finished(self):
        """ Finishing condition is agent type dependent."""
        raise NotImplementedError("Subclasses must implement the is_finished method.")
    
    async def run(self):
        """Agent type dependent."""
        raise NotImplementedError("Subclasses must implement the run method.")
        


@define
class WikiNavAgent(Agent):
    """
    Virtual class for agents that navigate Wikipedia with a target page.
    """
    target_page: str = field(default=None)

    # features
    use_retry: bool = field(default=False)
    use_reasoning: bool = field(default=False)
    use_backtrack: bool = field(default=False) # backtracking

    # finishing conditions
    max_errors: int = field(default=2) # for feedback
    max_steps: int = field(default=15)


    def set_target_page(self, target_page: str):
        self.target_page = wikify_article_name(target_page)
    
    def is_finished(self, force=False):
        if force:
            return True
        else:
            # printc(f"is_finished: {self.environment.current_page} == {self.target_page} or {self.num_actions} >= {self.max_steps} or {self.num_errors} >= {self.max_errors}", c="r")
            return self.environment.current_page == self.target_page or self.num_actions >= self.max_steps or self.num_errors >= self.max_errors
        
    def get_run_df(self):
        df = pd.DataFrame({
            'agent_id': self.agent_id,
            'agent_type': self.__class__.__name__,
            'src': self.environment.page_history[0],
            'tgt': self.target_page,
            'path': ";".join(self.environment.page_history),
            'fewshot': self.fewshot,
            'num_actions': self.num_actions,
            'num_errors': self.num_errors,
            'user_prompt_history': [self.user_prompt_history],
            'response_history': [self.response_history],
        })
        return df
    
    async def run(self):
        """ Run the agent until game is over or error."""
        while not self.is_finished():
            try:
                await self.act()
                self.num_actions += 1
                self.num_errors = 0
            except Exception as e:
                printc(f"Error in agent {self.agent_id}: {e}", c="r")
                self.num_errors += 1
                if self.use_retry:
                    pass
                else:
                    break

        if self.is_finished() and self.environment.current_page == self.target_page:
            # printc(f"Agent {self.agent_id} finished in {self.num_actions} actions.", c="r")
            return True
        else:
            return False


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

            d['src'] = d['src']
            d['tgt'] = d['tgt']
            d['links'] = ';'.join([link for link in d['links'].split(';')])
            d['click'] = d['click']
        return dicts

    def observe_environment(self):
        observed_data = {
            "src": self.environment.current_page,
            "tgt": self.target_page,
            "links": ";".join([link for link in self.environment.links]),
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
        # if 'click' not in actions:
        if 'next_page' not in actions:
            raise AssertionError(f"Invalid action: {actions}")
        else:
            try:
                # self.click(actions['click'])
                self.click(actions['next_page'])
            except Exception as e:
                raise e
        

