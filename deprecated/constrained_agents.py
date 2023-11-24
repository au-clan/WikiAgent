from attr import define, field
from typing import List
from models.huggingface_model import preprocess_llama_chat
from model_utils import BaseMessage, printc_prompt, prepare_chat
from utils import printc, filter_links_from_text
from time import time
import lmql
from agents import LinkClickDFSAgent, Agent, BaseWikiAgent, LinkClickAgent
from transformers import AutoTokenizer

@define
class WikiTextAgent(BaseWikiAgent):
    model: lmql.model = field(default=None)
    model_cfg: dict = field(default=None)
    action_list: List[str] = field(default=["scroll_down", "click", "back"])
    tokenizer = field(default=None)

    def set_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg['model_name'])

    @lmql.query
    async def lmql_query(self, prompt: str):
        '''lmql
        sample(n=1, temperature=self.model_cfg['temperature'], top_p=self.model_cfg['top_p'], max_len=4096)
        "{prompt}"
        "Thought:\n[thought]" where stops_before(thought, 'Action:')
        "Action:\n[next_action]" where next_action in set(self.action_list)
        if next_action=='click':
            "([link])" where link in set(self.environment.links)
        '''
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

    def _load_fewshot_demonstrations_data(self) -> List[dict]:
        return [{}]
    
    def apply_actions(self, actions: dict):
        if actions['next_action'] not in self.action_list:
            raise Exception(f"Invalid action {actions['next_action']}")
        if actions['next_action'] == "scroll_down":
            self.environment.move("down")
        elif actions['next_action'] == "click":
            self.environment.change_page(actions['link'])
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
        ###printc(f"Agent {self.agent_id} says: {prompt[-1].content}")
        printc(prepared_prompt, c='p')

        self.user_prompt_history.append(prompt[-1].content)
        start = time()
        if not self.debug:
            response = await self.lmql_query(prepared_prompt, candidates=links, model=self.model)
        else:
            input_ = input("Response from model:")
            response = input_
        ### TODO: fix this for non llama
        self.raw_response_history.append(response.prompt.split("[/INST]")[-1])
        printc(f"Agent {self.agent_id} received response. Time in queue: {time() - start} seconds. Response:\n{response.variables}")
        self.response_history.append(response.variables)
        actions = response.variables
        self.apply_actions(actions)
        self.action_history.append(actions)
