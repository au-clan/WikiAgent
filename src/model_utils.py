# Classes adapted from github.com/epfl-dlab/GPTeam
from typing import List, Union, Any
import tiktoken
import yaml
from datetime import datetime as dt
import attr
from attr import define, field
import sys
sys.path.append("src/")
from utils import get_api_key, printc

@define
class BaseMessage:
    role: str = attr.ib()
    content: str = attr.ib()
    ext_visible: bool = field(default=True)
    alt_role: str = field(default=None)

    def format_prompt(self, before, to):
        self.content = self.content.replace("{" + before + "}", f"{to}")
        return self

    def prepare_for_generation(self, provider=None):
        """Google chat models do not support ongoing 'system' type messages. They allow for custom role naming."""
        _role = self.role
        if provider == 'google':
            if self.role == 'assistant':
                _role = 'bot'
            if self.alt_role is not None:
                _role = self.alt_role
            # google uses 'author' instead of 'role'
            return {"author": _role, "content": self.content}

        return {"role": _role, "content": self.content}

    def text(self):
        return self.content

    def copy(self):
        if isinstance(self, SystemMessage):
            c = SystemMessage
        elif isinstance(self, AIMessage):
            c = AIMessage
        elif isinstance(self, HumanMessage):
            c = HumanMessage
        else:
            c = BaseMessage

        return c(role=self.role, content=self.content, alt_role=self.alt_role)

    def __str__(self):
        return self.content

    def __getitem__(self, key):
        return self.__dict__[key]


@define
class AIMessage(BaseMessage):
    role: str = "assistant"
    
    def prepare_for_completion(self):
        return f"\nAssistant: {self.content}"

@define
class AIMessagePrefix(BaseMessage):
    role: str = "assistant_prefix"

    def prepare_for_completion(self):
        return f"\nAssistant: {self.content}"
    
@define
class AIMessageCompletion(BaseMessage):
    role: str = "assistant_completion"

    def prepare_for_completion(self):
        return f"{self.content}"
    
@define
class HumanMessage(BaseMessage):
    role: str = "user"

    def prepare_for_completion(self):
        return f"\nUser: {self.content}"


@define
class SystemMessage(BaseMessage):
    role: str = "system"

    def __add__(self, otherSystem):
        # i think this is the only class that will require adding.
        return SystemMessage(self.content + "\n" + otherSystem.content)
    
    def prepare_for_completion(self):
        return f"{self.content}"





@define
class ChatModel:
    """
    Basic LLM API model wrapper class

    # TODO: (1) error handling
    #       (2) fix token counts and history for batched inputs
    """
    model_provider: str = field(default='openai')
    model_name: str = field(default='gpt-3.5-turbo')
    model_key: str = field(default=None)
    model_key_path: Any = field(default='secrets.json')
    model_key_name: Any = field(default=None)
    model: Any = field(default=None)

    debug_mode: bool = field(default=False)
    temperature: float = field(default=0.0)
    generation_params: dict = field(factory=dict)
    context_max_tokens: int = field(default=128)
    prompt_cost: float = field(default=0)
    completion_cost: float = field(default=0)
    tpm: int = field(default=0)
    rpm: int = field(default=0)
    api_info: dict = field(factory=dict)
    messages: list = field(factory=list)
    response: str = field(default="")
    prompt_tokens: int = field(default=0)
    completion_tokens: int = field(default=0)
    device: str = field(default='cpu')

    # keep track of costs
    budget: float = field(default=10.)
    session_prompt_costs: float = field(default=0.)
    session_completion_costs: float = field(default=0.)

    def __attrs_post_init__(self):

        if self.model_key is None:
            self.model_key = get_api_key(fname=self.model_key_path, provider=self.model_provider,
                                            key=self.model_key_name)
        # get model api details
        model_details = get_model_details(self.model_name)
        self.context_max_tokens = model_details['max_tokens']
        self.prompt_cost = model_details['prompt_cost']
        self.completion_cost = model_details['completion_cost']
        self.tpm = model_details['tpm']
        self.rpm = model_details['rpm']
        # get api info
        self.api_info = get_api_settings(self.model_provider)

        # only single generations currently implemented
        if self.generation_params.get("n", 1) >= 2:
            raise NotImplementedError("Need to implement for more than one generation.")


    def __call__(self, messages: Union[List[BaseMessage], List[List[BaseMessage]]], keep_history=False):
        """
        Generate tokens.
        If keep_history is True, then the model will keep track of the message history throughout all calls.
        """
        #printc(f"ChatModel received messages: {messages}", c="g")
        if self.debug_mode:
            # time.sleep(0.2)  # small wait to see sensible msg timestamps
            return f"<{dt.strftime(dt.now(), '%H%M%S_%f')}> lorem ipsum dolor sit amet"
        self.prompt_tokens, self.completion_tokens = -1, -1
        response = self._generate(messages)

        # TODO: Make the code below work for batches
        # # update internal books:
        # self._update_token_counts(prompt_messages=messages, completion=response.content)

        # # update message history
        # if not keep_history:
        #     self.messages = messages.copy()
        # else:
        #     self.messages += messages
        # self.messages.append(response)
        return response

    def _generate(self, data) -> BaseMessage:
        pass

    def _preprocess(self, data: List[BaseMessage]):
        pass

    def _postprocess(self, data) -> BaseMessage:
        pass

    def _update_token_counts(self, prompt_messages, completion):
        self.prompt_tokens = self.estimate_tokens(messages=prompt_messages) if self.prompt_tokens < 0 else self.prompt_tokens
        self.completion_tokens = self.estimate_tokens(completion) if self.completion_tokens < 0 else self.completion_tokens

        # keep track of budget and costs
        pc, cc, _ = self.estimate_cost(input_tokens=self.prompt_tokens, output_tokens=self.completion_tokens)
        self.session_prompt_costs += pc
        self.session_completion_costs += cc
        self.budget -= (pc + cc)

    def history(self):
        # optionally keep a history of interactions
        return self.messages

    def estimate_cost(self, input_tokens: int, output_tokens: int):
        """
        Basic cost estimation
        """
        input_cost = (input_tokens / 1000) * self.prompt_cost
        output_cost = (output_tokens / 1000) * self.completion_cost
        total_cost = input_cost + output_cost

        return input_cost, output_cost, total_cost

    def estimate_tokens(self, messages: Union[List[BaseMessage], str]):
        if "hf" in self.model_name:
            enc = self.tokenizer
        try:
            enc = tiktoken.encoding_for_model(self.model_name)
        except KeyError as e:
            enc = tiktoken.encoding_for_model("gpt-4")
        msg_token_penalty = 8
        str_to_estimate = ''

        if isinstance(messages, list):
            for m in messages:
                if isinstance(m, dict):
                    str_to_estimate += m["content"]
                elif isinstance(m, BaseMessage):
                    str_to_estimate += m.text()
                else:
                    str_to_estimate += ''
        elif isinstance(messages, str):
            str_to_estimate = messages
        else:
            str_to_estimate = ''

        tokens = len(enc.encode(str_to_estimate)) + msg_token_penalty
        ###
        return tokens

    def check_context_len(self, context: List[BaseMessage], max_gen_tokens: int) -> bool:
        # TODO: currently not checking context length anymore!!!
        """Calculate how many tokens we have left. 
        
        messages: List[system_msg, msg_history, note_history, prev_game_history) + note/msg]
        
        Returns:
            got_space (bool)
        """
        # 1. enough context token space to generate note?
        # 2. enough context token space to generate msg?

        # openai adds 8 default tokens per request¨
        msg_token_penalty = 8
        context_tokens = self.estimate_tokens(context)
        tokens_left = self.context_max_tokens - (context_tokens + msg_token_penalty + max_gen_tokens)
        got_space = tokens_left > 0

        return got_space

    def __repr__(self):
        return f'ChatModel("model_name"={self.model_name}, "model_provider"={self.model_provider})'

    def get_history_str(self) -> str:
        history_str = ""
        for msg in self.messages:
            history_str += f"{msg.role}: {msg.content}\n"
        return history_str

def get_model_pricing(model_name):
    model_details = get_model_details(model_name=model_name)
    return model_details['prompt_cost'], model_details['completion_cost']

# TODO: change paths
def get_model_details(model_name, fpath='data/llm_model_details.yaml'):
    try:
        with open(fpath) as f:
            details = yaml.safe_load(f)
    except Exception as e:
        print(f'error: unable to load model details - {e}')
        details = {}

    models = details.keys()
    if model_name not in models:
        raise KeyError(f'error: no details available for model {model_name} - pick one of {models}')

    return details[model_name]

# TODO: change paths
def get_api_settings(api_provider, fpath='data/api_settings/apis.yaml'):
    try:
        with open(fpath) as f:
            details = yaml.safe_load(f)
    except Exception as e:
        print(f'error: unable to load model details - {e}')
        details = {}

    models = details.keys()
    if api_provider not in models:
        raise KeyError(f'error: no details available for model {api_provider} - pick one of {models}')

    return details[api_provider]

def printc_prompt(prompt: List[BaseMessage], c="r"):
    for msg in prompt:
        printc(f"[{msg.role.upper()}]: {msg.content}", c=c)