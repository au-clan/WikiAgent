# Base Agent class and subclasses
# Some notes:
# - Prompt = a list of BaseMessages
# - Template = a string with placeholders for the agent to fill in
import attr
from attr import define, field
import os
import yaml
import asyncio
from model_utils import SystemMessage, HumanMessage, AIMessage, AIMessagePrefix, BaseMessage, AIMessageCompletion, printc_prompt
from environments import Environment
from batch_handler import BatchHandler
from utils import load_tsv_to_dict_list, printc, extract_json_from_response, wikify_article_name
from time import time
import pandas as pd
import json
from typing import List
import lmql

@define
class Agent:
    """
    Virtual agent class. Defines the main interface for the agent.
    """
    environment: Environment = field(default=None)
    handler: BatchHandler = field(default=None)
    agent_id: int = field(default=None)
    source_page: str = field(default=None)

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
        self.source_page = environment.current_page
    
    def set_handler(self, handler: BatchHandler):
        self.handler = handler
    
    def set_agent_id(self, agent_id: int):
        self.agent_id = agent_id
        
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

                demos_list = self._load_fewshot_demonstrations_data()
                
                prompt = [SystemMessage(system_message_content)]
                user_message_content += self.user_initial_prefix.format(replace=replace_dict)
                ai_message_content = ""
                # fewshot examples
                for demo in demos_list:
                    user_message_content += self.user_template.format(replace = demo)
                    ai_message_content = self.demonstration_template.format(replace = demo)
                    prompt.append(HumanMessage(user_message_content))
                    prompt.append(AIMessage(ai_message_content))
                    user_message_content = ""
                    ai_message_content = ""
                    
            else: # non-multiturn demonstrations 
                replace_dict['fewshot_demonstrations'] = "\n".join(self.get_fewshot_demonstrations())
                replace_dict['agent_description'] = self.agent_description.format(replace=replace_dict)
                system_message_content = self.prompt_template['system'].format(replace=replace_dict)
                user_message_content += self.user_initial_prefix.format(replace=replace_dict)
                system_message = SystemMessage(system_message_content)
                prompt = [system_message]

            env_data = self.observe_environment()

            # current user prompt
            user_message_content += self.user_template.format(replace = env_data)
            user_message = HumanMessage(content = user_message_content)
            assistant_prefix = AIMessagePrefix(self.assistant_prefix)

            prompt.append(user_message); prompt.append(assistant_prefix)
            self.prompt = prompt
        else: #got feedback
            # append bad response and feedback to the prompt
            self.prompt.append(AIMessageCompletion(self.raw_response_history[-1]))
            self.prompt.append(HumanMessage(feedback))
            self.prompt.append(AIMessagePrefix(self.assistant_prefix))
            prompt = self.prompt    
        return prompt
    
        
    def extract_actions(self, response: BaseMessage) -> dict:
        """ Processes the model output and returns a dict with the agent's choices."""
        extracted_json = extract_json_from_response(self.assistant_prefix + response.content)
        if extracted_json:
            actions = json.loads(extracted_json)
        else:
            raise Exception(f"Invalid output format. Please follow the correct format.")
        return actions
    
    def apply_actions(self, actions: dict):
        """ Applies the actions to the environment."""
        raise NotImplementedError("Subclasses must implement the apply_actions method.")

    def update_runs_df(self):
        """ Depends on the agent type."""
        pass

    async def act(self, feedback: str = None, debug=False):
        """ One turn of the agent. -Build prompt; Send prompt to model; Apply actions
        args:
            feedback: str = None
                Feedback from the user. If not None, it means the agent made a mistake and needs to be corrected.
        """
        printc(f"Agent {self.agent_id} on page {self.environment.current_page}, num_actions={self.num_actions}, num_errors={self.num_errors}.", c="r")
        prompt = self.build_prompt(feedback)

        printc(f"Agent {self.agent_id} says: {prompt[-2].content}")
        #printc_prompt(self.prompt, c='p')
    
        self.user_prompt_history.append(prompt[-2].content)
        start = time()
        if not debug:
            response = await self.handler.send_prompt(self.agent_id, prompt)
        else:
            input_ = input("Response from model:")
            response = AIMessage(input_)
        self.raw_response_history.append(response.content)
        response.content = self.assistant_prefix + response.content

        printc(f"Agent {self.agent_id} received response. Time in queue: {time() - start} seconds. Response:\n{response}")
        self.response_history.append(response.content)
        actions = self.extract_actions(response)
        self.apply_actions(actions)
        self.action_history.append(actions)

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
    visited_pages: List[str] = field(factory=list)

    # features
    use_retry: bool = field(default=False)
    use_reasoning: bool = field(default=False)
    use_backtrack: bool = field(default=False) # backtracking

    # finishing conditions
    max_tries: int = field(default=3) # retry same prompt
    max_actions: int = field(default=15)

    def set_target_page(self, target_page: str):
        self.target_page = wikify_article_name(target_page)

    def set_environment(self, environment: Environment):
        self.visited_pages.append(environment.current_page)
        return super().set_environment(environment)
    
    def update_runs_df(self):
        row = {
            'agent_id': self.agent_id,
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
        
    async def run(self, debug=False):
        """ Run the agent until game is over or error."""
        feedback = None
        while not self.is_finished()[0]:
            try:
                await self.act(feedback=feedback, debug=debug)
                feedback = None
                self.num_actions += 1
                self.num_tries = 0
            except Exception as e:
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
class LinkClickAgent(WikiNavAgent):
    """
    Agent that clicks knowing only the list of links in a certain page.
    """
    template_file: str = field(default="src/configs/agent/link_click_agent.yaml")
    demonstrations_file: str = field(default="data/wikispeedia/wikispeedia_paths-and-graph/fewshot_click_demonstrations.tsv")
    fewshot_click_types: str = field(default='all_no_back')

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
            dicts = [row.to_dict() for _, row in demos_df.iterrows()]
        return dicts[:self.fewshot]

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
        except Exception as e:
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

@define
class LinkClickDFSAgent(LinkClickAgent):
    """
    This class allows backtracking and excludes previously visited pages from candidates.
    Makes playing Wikispeedia similar to a DFS.
    """
    use_back: bool = field(default=True)

    def __attrs_post_init__(self):
        if self.fewshot_click_types != 'first-only':
            self.fewshot_click_types = 'all_no_back' if not self.use_back else 'all'
        return super().__attrs_post_init__()

    def observe_environment(self):
        self.environment.remove_pages_from_links(self.visited_pages)
        return super().observe_environment()
    
    def back(self):
        """
        Backtracks to the previous page.
        """
        try:
            self.environment.back()
            return True
        except Exception as e:
            raise e
        
    def apply_actions(self, actions: dict):
        """
        Apply actions from the model.
        """
        if 'next_page' not in actions:
            raise AssertionError(f"Invalid action: {actions}")
        else:
            if self.use_back and actions['next_page'] == "NA":
                try:
                    self.back()
                except Exception as e:
                    raise e
            else:
                try:
                    self.click(actions['next_page'])
                    self.visited_pages.append(self.environment.current_page)
                except Exception as e:
                    raise e
    
@define 
class LMQLAgent(WikiNavAgent):

    @lmql.query
    async def lmql_query(prompt):
        '''lmql
        {prompt}
        """{{"next_page": "[NEXT_PAGE]"}}"""
        return NEXT_PAGE
        '''
