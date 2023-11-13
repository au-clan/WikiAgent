from attr import define, field
from typing import List
from models.huggingface_model import preprocess_llama_chat
from model_utils import BaseMessage, printc_prompt, prepare_chat
from utils import printc, filter_links_from_text
from time import time
import lmql
from agents import LinkClickDFSAgent, Agent, BaseWikiAgent
from transformers import AutoTokenizer

@define 
class LMQLLinkClickDFSAgent(LinkClickDFSAgent):
    model= field(default=None)
    model_cfg: dict = field(default=None)
    tokenizer = field(default=None)

    def set_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg['model_name'])
        return super().__attrs_post_init__()
    
    @lmql.query(output_writer=lmql.printing)
    async def lmql_query_reason(self, prompt: str, candidates: List[str]):
        #
        '''lmql
        sample(n=1, temperature=self.model_cfg['temperature'], top_p=self.model_cfg['top_p'], max_len=2048)
            "{prompt}"
            """{{"reason": "The 5 most relevant pages to {self.target_page} are: """

            relevant_links = []
            for i in range(5):
                '{i+1}.[relevant_link]: [explanation]' where relevant_link in set(candidates) and stops_at(explanation, '.') and stops_before(explanation, '"')
                relevant_links.append(relevant_link)
                if i < 4:
                    ';'

            """ Therefore, [reason], "next_page": "[next_page]"}}""" where next_page in set(relevant_links) and len(TOKENS(reason)) < 150 and stops_before(reason, '"') and stops_before(reason, '\n')
            
        
        from
            lmql.model(f"local:{self.model_cfg['model_path']}" if self.model_cfg['model_path'] else f"local:{self.model_cfg['model_name']}", 
            cuda=True if self.model_cfg['device']=='cuda' else False, 
            batch_size=self.model_cfg['batch_size'],
            dtype='float16' if self.model_cfg['half_precision'] else 'float32',)
        '''

    @lmql.query
    async def lmql_query(self, prompt: str, candidates: List[str]):
        #
        '''lmql
        sample(n=1, temperature=self.model_cfg['temperature'], top_p=self.model_cfg['top_p'], max_len=4096)
            "{prompt}"
            """{{"next_page": "[next_page]"}}""" where next_page in set(candidates)
        '''

    def extract_actions(self, response: dict) -> dict:
        """ Processes the model output and returns a dict with the agent's choices."""
        return response['next_page']
    
    async def act(self, feedback: str = None):
        """ One turn of the agent. -Build prompt; Send prompt to model; Apply actions
        args:
            feedback: str = None
                Feedback from the user. If not None, it means the agent made a mistake and needs to be corrected.
        """
        printc(f"Agent {self.agent_id} on page {self.environment.current_page}, num_actions={self.num_actions}, num_errors={self.num_errors}.", c="r")
        prompt = self.build_prompt(feedback)

        if self.model_cfg['model_type'] == 'chat':
            prepared_prompt = prepare_chat(self.tokenizer,prompt)
        else:
            prepared_prompt = prompt
        links = self.environment.links
        if self.use_back and self.environment.current_page != self.source_page:
            links.append("NA")

        printc(f"Agent {self.agent_id} says: {prompt[-1].content}")
        printc_prompt(self.prompt, c='p')
    
        self.user_prompt_history.append(prompt[-1].content)
        start = time()
        if not self.debug:
            response = await self.lmql_query(prepared_prompt, candidates=links, model=self.model)
        else:
            input_ = input("Response from model:")
            response = input_
        ### TODO: fix this for non llama
        self.raw_response_history.append(response.prompt.split("[/INST]")[-1])
        printc(response, c='y')

        printc(f"Agent {self.agent_id} received response. Time in queue: {time() - start} seconds. Response:\n{response.prompt.split('[/INST]')[-1]}")
        self.response_history.append(response.variables)
        actions = response.variables
        self.apply_actions(actions)
        self.action_history.append(actions)

@define
class WikiNavAgent(BaseWikiAgent):
    model: lmql.model = field(default=None)
    model_cfg: dict = field(default=None)
    action_list: List[str] = field(default=["scroll_down", "click", "back"])
    tokenizer = field(default=None)

    def __attrs_post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg['model_name'])
        return super().__attrs_post_init__()
    
    @lmql.query
    async def lmql_query(self, prompt: str):
        '''lmql
        sample(n=1, temperature=self.model_cfg['temperature'], top_p=self.model_cfg['top_p'], max_len=4096)
            "{prompt}"
            """Thought: [thought]\n""" where stops_before(thought, "\n") and len(TOKENS(thought)) < 50
            """Final answer: {{"next_action": "[next_action]",""" where next_action in set(self.action_list)
            if next_action=="click":
                """ "arguments": "[arguments]"}}""" where arguments in set(self.environment.links)
            else:
                """" "arguments": ""}}"""
        '''
    def observe_environment(self):
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
            "scroll_position": f"{self.environment.position[0]}/{self.environment.position[1]}"
        }
        return observed_data

    def _load_fewshot_demonstrations_data(self) -> List[dict]:
        return [{}]
    
    def apply_actions(self, actions: dict):
        if actions['next_action'] not in self.action_list:
            raise Exception(f"Invalid action {actions['next_action']}")
        if actions['next_action'] == "scroll_down":
            self.environment.move("down")
        elif actions['next_action'] == "click":
            self.environment.change_page(actions['arguments'])
        else: # actions['next_action'] == "back":
            self.environment.back()
        
    async def act(self, feedback: str = None):
        """ One turn of the agent. -Build prompt; Send prompt to model; Apply actions
        args:
            feedback: str = None
                Feedback from the user. If not None, it means the agent made a mistake and needs to be corrected.
        """
        printc(f"Agent {self.agent_id} on page {self.environment.current_page}, num_actions={self.num_actions}, num_errors={self.num_errors}.", c="r")
        prompt = self.build_prompt(feedback)

        if self.model_cfg['model_type'] == 'chat':
            prepared_prompt = prepare_chat(self.tokenizer,prompt)
        else:
            prepared_prompt = prompt

        links = self.environment.links
        if self.use_back and self.environment.current_page != self.source_page:
            links.append("NA")

        printc(f"Agent {self.agent_id} says: {prompt[-1].content}")
        printc_prompt(self.prompt, c='p')
    
        self.user_prompt_history.append(prompt[-1].content)
        start = time()
        if not self.debug:
            response = await self.lmql_query(prepared_prompt, candidates=links, model=self.model)
        else:
            input_ = input("Response from model:")
            response = input_
        ### TODO: fix this for non llama
        self.raw_response_history.append(response.prompt.split("[/INST]")[-1])

        printc(f"Agent {self.agent_id} received response. Time in queue: {time() - start} seconds. Response:\n{response.prompt.split('[/INST]')[-1]}")
        self.response_history.append(response.variables)
        actions = response.variables
        self.apply_actions(actions)
        self.action_history.append(actions)
