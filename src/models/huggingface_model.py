# Adapted from github.com/epfl-dlab/GPTeam

from typing import List, Union
from attr import define
import os 
import sys 
sys.path.append("src/")
# fix for importing pytorch: https://github.com/pytorch/pytorch/issues/78490
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from transformers import AutoTokenizer, AutoModelForCausalLM
from model_utils import BaseMessage, ChatModel, AIMessage, SystemMessage, HumanMessage
import torch

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

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
    tokenizer: AutoTokenizer = None  # The tokenizer to be used
    model: AutoModelForCausalLM = None  # The model to be used
    half_precision: bool = True
    top_p: float = 0.95 # top-p sampling parameter
    do_sample: bool = True # whether to sample or not
    num_threads: int = 16 # number of CPU cores to use
    raw_prompt: str = None # the raw prompt

    def __attrs_post_init__(self):
        """
        This method is called after the instance has been initialized. 
        It initializes the model.
        """
        self.init_model()

    def _preprocess(self, data):
        """
        This method preprocesses the data based on the model name.
        """
        if "chat" in self.model_name:
            msges = [k.prepare_for_generation() for k in data]
            concat_msges = preprocess_llama_chat(msges)
            self.raw_prompt = concat_msges
            return self.tokenize(concat_msges)

        msges = [k.prepare_for_completion() for k in messages]
        concat_msges = ""
        for msg in msges:
            concat_msges += msg + "\n"

        self.prompt_tokens = self.estimate_tokens(concat_msges)
        self.raw_prompt = concat_msges
        return self.tokenize(concat_msges)

    def _generate(self, data):
        """
        This method generates the output using the model.
        """
        data = self._preprocess(data)
        outputs = self.model.generate(data,
                                      max_new_tokens=self.context_max_tokens,
                                      pad_token_id=self.tokenizer.eos_token_id,
                                      temperature=self.temperature,
                                      do_sample=self.do_sample, top_p=self.top_p,)[:, data.shape[1]:]

        output = self._postprocess(outputs)
        return output

    def _postprocess(self, outputs) -> AIMessage:
        """
        This method postprocesses the outputs, decoding them into AIMessages.
        """
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # .generate method returns the full string.
        if "chat" in self.model_name:
            decoded = decoded.split(E_INST)[-1]

        output = AIMessage(decoded.strip())

        return output

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
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                          #device_map="auto",
                                                          torch_dtype=torch.float16 if self.half_precision else torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        torch.set_num_threads(self.num_threads)
    

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

    # If dialog only contains one item, handle it separately
    if len(dialog) == 1:
        dialog_list = [f"<s>{B_INST} {(dialog[0]['content']).strip()} {E_INST}"]
    else:
        # Create a formatted string for each paired prompt and answer in the dialog
        dialog_list = [f"<s>{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()}</s>"
                       for prompt, answer in zip(dialog[::2], dialog[1::2])]
        # Append the last dialog's content separately if the number of dialog items is odd
        dialog_list += [f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"]

    # combine all dialog strings into a single string separated by newlines
    dialog_text = "\n".join(dialog_list)

    return dialog_text


if __name__ == "__main__":
    # test that this module works
    messages = [SystemMessage("You're a nice assistant"), HumanMessage("Hey my name is Mariama! How are you?")]

    model = HuggingFaceModel()
    print(model(messages))