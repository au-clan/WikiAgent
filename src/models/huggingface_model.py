# Adapted from github.com/epfl-dlab/GPTeam

from typing import List, Union, Any
from attr import define
import os 
import sys 
sys.path.append("src/")
# fix for importing pytorch: https://github.com/pytorch/pytorch/issues/78490
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from transformers import pipeline, StoppingCriteria, StoppingCriteriaList
from model_utils import BaseMessage, ChatModel, AIMessage, SystemMessage, HumanMessage
from utils import printc
import torch

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
LLAMA_PROMPT_FORMAT = {
    "user": "<s>"+B_INST+" {content} "+E_INST+" ",
    "assistant": "{content} </s>",
    "assistant_prefix": "{content}",
}

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


@define
class HuggingFaceModel(ChatModel):
    """
    This class is a wrapper for HuggingFace models, providing methods for initialization, 
    preprocessing, generation, postprocessing, tokenization, and model loading.
    """
    model_provider: str = 'huggingface'  # The provider of the model
    model_name: str = 'meta-llama/Llama-2-7b-chat-hf'  # The name of the model
    model_path: str = None  # The path to the model
    generator: pipeline = None  # The generator to be used
    tokenizer: Any = None  # The tokenizer to be used
    batch_size: int = 1  # The batch size to be used
    half_precision: bool = True
    top_p: float = 0.95 # top-p sampling parameter
    do_sample: bool = True # whether to sample or not
    num_threads: int = 16 # number of CPU cores to use
    raw_prompt: str = None # the raw prompt
    stop: str = None # the stop character (besides the eos token)

    def __attrs_post_init__(self):
        """
        This method is called after the instance has been initialized. 
        It initializes the model.
        """
        self.init_model()

    def _preprocess(self, data: Union[List[BaseMessage], List[List[BaseMessage]]]) -> List[str]:
        """
        This method preprocesses the data based on the model name.
        Handles single and batched inputs.
        """
        if isinstance(data[0], BaseMessage):
            # turn single input into a batch
            data = [data]

        if "chat" in self.model_name:
            msges = [[k.prepare_for_generation() for k in data]
                        for data in data]
            concat_msges = [preprocess_llama_chat(msg) for msg in msges]
            self.raw_prompt = concat_msges

            # return self.tokenizer(concat_msges)
            return concat_msges

        # TODO: fix this for non chat models
        # msges = [k.prepare_for_completion() for k in messages]
        # concat_msges = ""
        # for msg in msges:
        #     concat_msges += msg + "\n"

        # self.prompt_tokens = self.estimate_tokens(concat_msges)
        # self.raw_prompt = concat_msges
        # return self.tokenize(concat_msges)
        return None

    def _generate(self, data):
        """
        This method generates the output using the model.
        """
        data = self._preprocess(data)
        # printc(f"data: {data}", c="g")
        # outputs = List[dict{"generated_text": str}], for text generation models
        if self.stop:
            stopping_criteria = self.get_stopping_criteria()
        else:
            stopping_criteria = None
        for d in data:
            printc(f"\ninput: {d}", c="g")
        outputs = self.generator(data,
                                 max_new_tokens=self.context_max_tokens,
                                 pad_token_id=self.tokenizer.eos_token_id,
                                 do_sample=self.do_sample, 
                                 top_p=self.top_p,
                                 temperature=self.temperature,
                                 stopping_criteria=stopping_criteria,)#[:, data.shape[1]:]
        # outputs = self.model.generate(data,
        #                               max_new_tokens=self.context_max_tokens,
        #                               pad_token_id=self.tokenizer.eos_token_id,
        #                               temperature=self.temperature,
        #                               do_sample=self.do_sample, top_p=self.top_p,)[:, data.shape[1]:]
        # printc(f"model outputs: {outputs}", c="g")
        for output in outputs:
            printc(f"\noutput: {output}", c="p")
        output = self._postprocess(outputs)
        token_list = self.tokenizer.encode(output[0].content)
        # printc(f"output tokens: {[(token_list[i], self.tokenizer.decode(token_list[i])) for i in range(len(token_list))]}", c="g")
        return output

    def _postprocess(self, outputs) -> AIMessage:
        """
        This method postprocesses the outputs, decoding them into AIMessages.
        """
        processed_outputs = []
        if "chat" in self.model_name:
            for output in outputs:
                output = output[0]["generated_text"].split(E_INST)[-1]
                processed_outputs.append(AIMessage(output.strip()))

        return processed_outputs
    
    def tokenize(self, message: str):
        """
        This method tokenizes the input message.
        """
        token_input = self.tokenizer.encode(message,
                                            return_tensors='pt',
                                            padding=True,
                                            max_length=700,
                                            truncation=True)
        if self.device == "cuda":
            token_input = token_input.cuda()
        
        return token_input
    
    def init_model(self):
        """ 
        This method initializes the model and tokenizer.
        """
        if self.device=="cpu":
            self.generator = pipeline("text-generation", model=self.model_path if self.model_path else self.model_name,
                                                            tokenizer=self.model_path if self.model_path else self.model_name,
                                                            batch_size=self.batch_size, 
                                                            device= -1)
            torch.set_num_threads(self.num_threads)
        else: # cuda device
            self.generator = pipeline("text-generation", model=self.model_path if self.model_path else self.model_name,
                                                    tokenizer=self.model_path if self.model_path else self.model_name,
                                                    batch_size=self.batch_size, 
                                                    torch_dtype = torch.float16 if self.half_precision else torch.float32,
                                                    device_map="auto")
        if "llama" in self.model_name.lower():
            self.generator.tokenizer.pad_token = "[PAD]"
            self.generator.tokenizer.padding_side = "left"
        else:
            self.generator.tokenizer.pad_token_id = self.tokenizer.eos_token_id            
        self.tokenizer = self.generator.tokenizer

    def get_stopping_criteria(self):
        # printc(f"stop_word: {(self.stop_token, self.tokenizer.decode(self.stop_token))}", c="g")
        
        return [CustomStoppingCriteria(stop=self.stop, tokenizer=self.tokenizer)]

def preprocess_llama_chat(dialog):
    """
    This function preprocesses a chat dialog by joining the dialog into a single formatted string.
    
    Args:
    dialog (list): A list of dictionaries, each containing a 'role' and a 'content' key.
    
    Returns:
    str: a formatted string containing the chat dialog.
    """
    # If the first role is a system role, join the system content and the next content
    # And replace the first two items in dialog with joined content (list of dict)
    if dialog[0]["role"] == "system":
        formatted_content = B_SYS + dialog[0]["content"] + E_SYS + dialog[1]["content"]
        dialog = [{"role": dialog[1]["role"], "content": formatted_content}] + dialog[2:]
    
    dialog_list = []
    for msg in dialog:
        if msg["role"] in LLAMA_PROMPT_FORMAT:
            dialog_list.append(LLAMA_PROMPT_FORMAT[msg["role"]].format(content=msg["content"]))
        else:
            raise ValueError(f"Invalid role: {msg['role']}")
    dialog_text = "".join(dialog_list)
    return dialog_text


class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop: str = None, tokenizer=None):
        self.stop = stop
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops = []) -> bool:
        last_id = input_ids[:,-1].tolist()
        # we have to check if the string is in the stops cause there's multiple tokens that have }
        last_str = self.tokenizer.decode(last_id)
        #printc(f"last_id: {last_id}, stop: {self.stop}, stop in last_str:{self.stop in last_str}", c="g")     
        return self.stop in last_str

if __name__ == "__main__":
    # test that this module works
    messages = [SystemMessage("You're a nice assistant"), HumanMessage("Hey my name is Mariama! How are you?")]

    model = HuggingFaceModel()
    print(model(messages))
