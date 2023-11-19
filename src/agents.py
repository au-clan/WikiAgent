# Base Agent class and subclasses
# Some notes:
# - Prompt = a list of BaseMessages
# - Template = a string with placeholders for the agent to fill in
import attr
from attr import define, field
import os
import yaml
import asyncio
from model_utils import SystemMessage, HumanMessage, AIMessage, AIMessagePrefix, BaseMessage, AIMessageCompletion, printc_prompt, prepare_chat
from transformers import AutoTokenizer
from models.huggingface_model import preprocess_llama_chat
from environments import *
from batch_handler import BatchHandler
from utils import filter_links_from_text, printc, wikify_article_name
from time import time
import pandas as pd
import json
from typing import Any, Coroutine, List
import lmql
from selenium import webdriver

@define
class Agent:
    """
    Virtual agent class. Defines the main interface for the agent.
    """
    environment: Environment = field(default=None)
    agent_name: str = field(default=None)
    source_page: str = field(default=None)
    agent_id: int = field(default=0)
    debug: bool = field(default=False)

    #features that are common to all agents
    use_feedback: bool = field(default=False)
    multiturn_demonstrations: bool = field(default=True)

    # File with templates for the prompt and demonstration structure
    base_template_file: str = field(default="src/configs/agent/base_agent.yaml")
    
    # specific to the agent type
    template_file: str = field(default="src/configs/agent/base_agent.yaml")
    prompt_template: dict = field(factory=dict)
    demonstration_template: str = field(factory=str)
    agent_description: str = field(factory=str)
    user_initial_prefix: str = field(factory=str)
    user_template: str = field(factory=str)
    assistant_prefix: str = field(factory=str)
    instructions: str = field(factory=str)

    fewshot: int = field(default=5)
    demonstrations_file: str = field(default=None)
    prompt: List[BaseMessage] = field(factory=list)
    prepared_prompt: str = field(factory=str)

    # data about a run
    num_actions: int = field(default=0)
    num_errors: int = field(default=0)
    num_tries: int = field(default=0)
    run_df = field(factory=pd.DataFrame)
    user_prompt_history = field(factory=list)
    response_history = field(factory=list)
    raw_response_history = field(factory=list) # without assistant prefix
    action_history = field(factory=list)
    agent_runs_file: str = field(default=None) #should be hydra

    # for LMQL
    model: lmql.model = field(default=None)
    model_cfg: dict = field(default=None)
    tokenizer: AutoTokenizer = field(default=None)
    lmql_query_str: str = field(default=None)


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

    def initialize_environment(self, **kwargs):
        """
        Initializes the environment specific to the agent type.
        """
        raise NotImplementedError("Subclasses must implement the initialize_environment method.")
    
    def set_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg['model_name'])
        
    def get_fewshot_demonstrations(self) -> List[str]:
        demos_list = self._load_fewshot_demonstrations_data()
        fewshot_demonstrations = []
        for demo in demos_list:
            user_template_filled = self.user_template.format(replace = demo)
            demo['user_template'] = user_template_filled
            fewshot_demonstrations.append(self.demonstration_template.format(replace = demo))
        return fewshot_demonstrations

    def _load_fewshot_demonstrations_data(self) -> List[dict]:
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
    
    def build_prompt(self, feedback: str = None) -> List[BaseMessage]:
        """
        Agent observes the environment and builds a prompt, that will be passed to the model.
        
        args:
            feedback: str = None
                Feedback from the user.
                
        """
        replace_dict = {"instructions": self.instructions, "feedback_prefix": self.prompt_template['feedback_prefix']}
        user_message_content = ""
        if not feedback:
            if self.multiturn_demonstrations:
                replace_dict['agent_description'] = self.agent_description.format(replace=replace_dict)
                system_message_content = self.prompt_template['system'].format(replace=replace_dict)                
                self.prompt = [SystemMessage(system_message_content)]
                user_message_content += self.user_initial_prefix.format(replace=replace_dict)
                ai_message_content = ""

                if self.fewshot > 0:
                    demos_list = self._load_fewshot_demonstrations_data()
                    # fewshot examples
                    for demo in demos_list:
                        user_message_content += self.user_template.format(replace = demo)
                        ai_message_content = self.demonstration_template.format(replace = demo)
                        self.prompt.append(HumanMessage(user_message_content))
                        self.prompt.append(AIMessage(ai_message_content))
                        user_message_content = ""
                        ai_message_content = ""
                    
            else: # non-multiturn demonstrations 
                replace_dict['fewshot_demonstrations'] = "\n".join(self.get_fewshot_demonstrations())
                replace_dict['agent_description'] = self.agent_description.format(replace=replace_dict)
                system_message_content = self.prompt_template['system'].format(replace=replace_dict)
                user_message_content += self.user_initial_prefix.format(replace=replace_dict)
                system_message = SystemMessage(system_message_content)
                self.prompt = [system_message]

            env_data = self.observe_environment()

            # current user prompt
            user_message_content += self.user_template.format(replace = env_data)
            user_message = HumanMessage(content = user_message_content)
            self.prompt.append(user_message)
        else: #got feedback
            # append bad response and feedback to the prompt
            self.prompt.append(AIMessage(self.raw_response_history[-1]))
            self.prompt.append(HumanMessage(feedback))
    
        
    def extract_actions_from_lmql_variables(self, variables: dict) -> dict:
        """ Processes the model output and returns a dict with the agent's choices."""
        raise NotImplementedError("Subclasses must implement the extract_actions method.")
    
    def apply_actions(self, actions: dict):
        """ Applies the actions to the environment."""
        raise NotImplementedError("Subclasses must implement the apply_actions method.")

    def update_runs_df(self):
        """ Depends on the agent type."""
        pass

    def get_lmql_query(self):
        prompt = self.prepared_prompt
        # substitute [ and ] with [[ and ]] to avoid lmql formatting
        prompt = re.sub(r'(?<!\[)\[(?!\[)', '[[', prompt)
        prompt = re.sub(r'(?<!\])\](?!\])', ']]', prompt)
        
        formatted_query = self.lmql_query_str.format(prompt=prompt, model_cfg=self.model_cfg)
        # because lmql formats the query again, we have to turn single occurrences of { into {{ and } into }}
        formatted_query = re.sub(r'(?<!\{)\{(?!\{)', '{{', formatted_query)
        formatted_query = re.sub(r'(?<!\})\}(?!\})', '}}', formatted_query)
        
        return lmql.query(formatted_query)
    
    async def send_query_lmql(self, query, **kwargs) -> lmql.LMQLResult:
        """Receives the prepared prompt along with other info to fill in prompt and sends it to the model.
        Outputs the response."""
        raise NotImplementedError("Subclasses must implement the query_lmql method.")
        

    async def act(self, feedback: str = None):
        """ One turn of the agent. -Build prompt; Send prompt to model; Apply actions
        args:
            feedback: str = None
                Feedback from the user. If not None, it means the agent made a mistake and needs to be corrected.
        """
        printc(f"Agent {self.agent_id} on page {self.environment.current_page}, num_actions={self.num_actions}, num_errors={self.num_errors}.", c="r")
        
        # Prepare prompt
        self.build_prompt(feedback)
        prepared_prompt = prepare_chat(self.tokenizer,self.prompt)
        self.prepared_prompt = prepared_prompt
        printc(f"Agent {self.agent_id} says: {self.prompt[-1].content}")
        self.user_prompt_history.append(self.prompt[-1].content)

        # Send prompt to model
        query = self.get_lmql_query()
        start = time()
        if not self.debug:
            response = await self.send_query_lmql(query)
        else:
            input_ = input("Response from model:")
            response = AIMessage(input_)
        # get only the new text added by the model in response.prompt
        self.prompt.append(AIMessage(response.prompt.split(prepared_prompt)[-1]))
        self.raw_response_history.append(response.prompt.split(prepared_prompt)[-1])
        printc(f"Agent {self.agent_id} received response. Time in queue: {time() - start} seconds. Response:\n{response.prompt}")
        self.response_history.append(response.variables)
        printc(f"Variables: {response.variables}", c="y")
        # Apply actions
        actions = self.extract_actions_from_lmql_variables(response.variables)
        self.apply_actions(actions)
        self.action_history.append(actions)

    def is_finished(self):
        """ Finishing condition is agent type dependent."""
        raise NotImplementedError("Subclasses must implement the is_finished method.")
    
    async def run(self):
        """Agent type dependent."""
        raise NotImplementedError("Subclasses must implement the run method.")

@define
class BaseWikiAgent(Agent):
    """
    Virtual class for agents that navigate Wikipedia with a target page.
    """
    target_page: str = field(default=None)
    visited_pages: List[str] = field(factory=list)

    # features
    use_retry: bool = field(default=False)
    use_reasoning: bool = field(default=False)
    use_back: bool = field(default=True)

    # finishing conditions
    max_tries: int = field(default=3) # retry same prompt
    max_actions: int = field(default=15)

    raise_error: bool = field(default=False)

    def set_target_page(self, target_page: str):
        self.target_page = wikify_article_name(target_page)

    def set_environment(self, environment: Environment):
        self.visited_pages.append(environment.current_page)
        return super().set_environment(environment)
    
    def update_runs_df(self):
        row = {
            'agent_name': self.agent_name,
            'agent_type': self.__class__.__name__,
            'cur': self.environment.current_page,
            'src': self.source_page,
            'tgt': self.target_page,
            'response': self.response_history[-1],
            'action': self.action_history[-1],
            'num_actions': self.num_actions,
            'num_tries': self.num_tries,
            'num_errors': self.num_errors,
            'is_finished': self.is_finished()[0],
            'is_finished_reason': self.is_finished()[1],
            'path_so_far': ";".join(self.environment.full_page_history),
            'user_prompt': self.user_prompt_history[-1],
        }
        self.run_df = pd.concat([self.run_df, pd.DataFrame([row])], ignore_index=True)

        # check if all_runs_df file exists
        if os.path.isfile(self.agent_runs_file):
            all_runs_df = pd.read_csv(self.agent_runs_file, sep="\t")
        else:
            all_runs_df = pd.DataFrame(columns=[key for key in row.keys()])

        #concat new row to all_runs_df
        all_runs_df = pd.concat([all_runs_df, pd.DataFrame([row])], ignore_index=True)
        all_runs_df.to_csv(self.agent_runs_file, sep="\t", index=False)
    
    
    def is_finished(self) -> (bool, str):
        if self.target_page in self.environment.links:
            return True, "success"
        elif self.num_actions >= self.max_actions:
            return True, "max_actions"
        elif self.num_tries >= self.max_tries:
            return True, "max_tries"
        else:
            return False, "none"
        
    async def run(self):
        """ Run the agent until game is over or error."""
        feedback = None
        while not self.is_finished()[0]:
            try:
                await self.act(feedback=feedback)
                feedback = None
                self.num_actions += 1
                self.num_tries = 0
            except Exception as e:
                if self.raise_error:
                    raise e
                printc(f"Error in agent {self.agent_id}: {e}", c="r")
                self.action_history.append(str(e))
                self.num_actions += 1
                self.num_errors += 1
                self.num_tries += 1
                if self.use_retry:
                    feedback = str(e) if self.use_feedback else None
                    pass
                else:
                    self.update_runs_df()
                    break
            self.update_runs_df()

        finished, reason = self.is_finished()
        if reason=='success':
            printc(f"Agent {self.agent_id} reached the target page {self.target_page} in {self.num_actions} actions.", c="g")
            return True
        else:
            printc(f"Agent {self.agent_id} failed to reach the target page {self.target_page} because of {reason}.", c="r")
            return False

@define
class LinkClickAgent(BaseWikiAgent):
    """
    Agent that clicks knowing only the list of links in a certain page.
    """
    template_file: str = field(default="src/configs/agent/link_click_agent.yaml")
    demonstrations_file: str = field(default="data/wikispeedia/wikispeedia_paths-and-graph/fewshot_click_demonstrations.tsv")
    demonstration_index: int = field(default=-1) # for testing different demos
    fewshot_click_types: str = field(default='all_no_back')
    sample_random_demos: bool = field(default=False)

    default_lmql_query_str = '''lmql
sample(n=1, temperature={model_cfg[temperature]}, top_p={model_cfg[top_p]}, max_len={model_cfg[max_len]})
"""{prompt}"""
"""{{"next_page": "[next_page]"}}""" where stops_before(next_page, '"')
'''
    lmql_query_str: str = field(default=default_lmql_query_str)

    def _load_fewshot_demonstrations_data(self) -> List[dict]:
        demos_df = pd.read_csv(self.demonstrations_file, sep='\t').fillna('NA')
        # remove src column
        demos_df = demos_df.drop(columns=['src'])
        #rename cur column to src
        demos_df = demos_df.rename(columns={'cur': 'src'})
        dicts = []
        if self.fewshot_click_types == 'first-only':
            demos_df = demos_df[demos_df['click_type'] == 'first']
            dicts = [row.to_dict() for _, row in demos_df.iterrows()]
        elif self.fewshot_click_types == 'all_no_back':
            demos_df = demos_df[demos_df['click_type'] != 'back']
            dicts = [row.to_dict() for _, row in demos_df.iterrows()]
        elif self.fewshot_click_types == 'all':
            if self.sample_random_demos:
                demos_df = demos_df.sample(n=self.fewshot)
            dicts = [row.to_dict() for _, row in demos_df.iterrows()]
        
        if self.demonstration_index != -1:
            return [dicts[self.demonstration_index]]
        else:
            return dicts[:self.fewshot]
        
    def initialize_environment(self, **kwargs):
        self.environment = LinkWikiEnviroment(**kwargs)
        self.source_page = self.environment.current_page
    
    def observe_environment(self):
        observed_data = {
            "src": self.environment.current_page,
            "tgt": self.target_page,
            "links": ";".join([link for link in self.environment.links]),
        }
        return observed_data
    
    async def send_query_lmql(self, query):
        candidates = self.environment.links
        return await query(candidates=candidates, model=self.model)
    
    def extract_actions_from_lmql_variables(self, variables: dict) -> dict:
        """ Processes the model output and returns a dict with the agent's choices."""
        actions = {}
        if 'next_page' in variables:
            actions['next_page'] = variables['next_page']
        else:
            raise AssertionError(f"Invalid action: {variables}")
        return actions

    def apply_actions(self, actions: dict):
        """
        Apply actions from the model.
        """
        self.environment.change_page(actions['next_page'])

@define
class LinkClickDFSAgent(LinkClickAgent):
    """
    This class allows backtracking and excludes previously visited pages from candidates.
    Makes playing Wikispeedia similar to a DFS.
    """
    def __attrs_post_init__(self):
        if self.fewshot_click_types != 'first-only':
            self.fewshot_click_types = 'all_no_back' if not self.use_back else 'all'
        return super().__attrs_post_init__()

    def observe_environment(self):
        self.environment.remove_pages_from_links(self.visited_pages)
        return super().observe_environment()
    
    def apply_actions(self, actions: dict):
        """
        Apply actions from the model.
        """
        if self.use_back and actions['next_page'] == "NA":
            self.environment.back()
        else:
            self.environment.change_page(actions['next_page'])
            self.visited_pages.append(self.environment.current_page)

@define
class WikiTextAgent(BaseWikiAgent):
    action_list: List[str] = field(default=["scroll_down", "click", "back"])
    default_lmql_query_str = '''lmql
sample(n=1, temperature={model_cfg[temperature]}, top_p={model_cfg[top_p]}, max_len={model_cfg[max_len]})
"""{prompt}"""
"Action:\n[next_action]" where stops_before(next_action, '(')
if next_action=='click':
    "([link])"
'''

    """Still does DFS, but uses the text of the page to make decisions."""
    def _load_fewshot_demonstrations_data(self) -> List[dict]:
        return [{}]
    
    def initialize_environment(self, **kwargs):
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')  # Necessary for running in headless mode on some systems
        options.add_argument('--no-sandbox')
        driver = webdriver.Chrome(options=options)
        #reading directly from the wpcd folder and there names are url encoded 1 time
        nav = WebNavigatorHTML(driver, pages=[quote(kwargs['current_page'])], downloads_folder="")
        env = WebWikiEnvironment(nav=nav, **kwargs)
        self.environment = env
        self.source_page = self.environment.current_page

    def observe_environment(self):
        ## for DFS
        self.environment.remove_pages_from_links(self.visited_pages)
        self.environment.content = filter_links_from_text(self.environment.content, self.environment.links)
        prev_page = self.environment.page_history[-2] if len(self.environment.page_history) > 1 else "none"

        observed_data = {
            "prev": prev_page,
            "src": self.environment.current_page,
            "tgt": self.target_page,
            "links": ";".join(self.environment.links),
            "content": self.environment.content,
            "remaining_actions": f"{self.max_actions - self.num_actions}/{self.max_actions}",
            "remaining_scrolls": self.environment.position[1] - self.environment.position[0],
        }
        return observed_data
    
    async def send_query_lmql(self, query):
        links = self.environment.links
        return await query(action_list=self.action_list, links=links, model=self.model)
        
    def extract_actions_from_lmql_variables(self, variables: dict) -> dict:
        """ Processes the model output and returns a dict with the agent's choices."""
        assert 'next_action' in variables, f"Invalid variables: {variables}"
        assert variables['next_action'] in self.action_list, f"Invalid action: {variables['next_action']}"
        return variables
    
    def apply_actions(self, actions: dict):
        if actions['next_action'] == "scroll_down":
            self.environment.move("down")
            # we don't count scrolls as actions, and there's +1 num_actions in the agent.run
            self.num_actions -= 1
        elif actions['next_action'] == "click":
            self.environment.change_page(actions['link'])
        else: # actions['next_action'] == "back":
            self.environment.back()

