from typing import List
import numpy as np
import torch
import regex
from transformers import PreTrainedModel, PreTrainedTokenizer, LogitsProcessor

from rellm.re_token_filter import ReTokenFilter

class LogitsMask(LogitsProcessor):
    """
    LogitsMask is a LogitsProcessor that masks logits for tokens that are 
    not in the allowed token ids set.
    """
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        mask = torch.ones_like(scores, device=scores.device) * -1e10
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = 0
        scores = scores + mask 
        return scores

def complete_re(prompt:str, pattern: regex.Pattern | List[regex.Pattern], tokenizer: PreTrainedTokenizer, 
                model: PreTrainedModel, max_new_tokens: int = 3, 
                stop_after_match: bool = True,
                debug: bool = False,
                **model_kwargs):
    """
    Complete a prompt with a regex pattern.
    """
    if isinstance(pattern, regex.Pattern):
        pattern = [pattern]
        
    gen_tokens = 0
    partial_completion = ""
    prompt_plus_completion = prompt + partial_completion

    token_filter = ReTokenFilter(tokenizer)

    while gen_tokens < max_new_tokens:
        prompt_token_ids = tokenizer.encode(prompt_plus_completion, return_tensors="pt").to(model.device)
        prompt_length = prompt_token_ids.shape[1]

        allowed_token_ids = token_filter.filter_tokens(partial_completion, pattern)
        
        if "llama" in model.name_or_path.lower():
            allowed_token_ids.discard(29871)

        allowed_token_ids_list = list(allowed_token_ids)
        custom_mask_processor = LogitsMask(allowed_token_ids)
        print("allowed_token_ids", [(allowed_token_ids_list[i], tokenizer.decode(allowed_token_ids_list[i])) for i in range(len(allowed_token_ids))])

        output_ids = model.generate(input_ids=prompt_token_ids,
                                    max_new_tokens=1,
                                    pad_token_id=tokenizer.eos_token_id,
                                    logits_processor=[custom_mask_processor],
                                    **model_kwargs
        )
        new_token_ids = output_ids[0, prompt_length:]
        output_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        partial_completion += output_text
        prompt_plus_completion = prompt_plus_completion + output_text

        if debug:
            print("step={} new_token{} completion={}"
                  .format(gen_tokens, (new_token_ids.item(), output_text), partial_completion))

        if stop_after_match:
            for p in pattern:
                m = p.match(partial_completion)
                if m and m.start() == 0:
                    return m[0]
        gen_tokens += 1

    return partial_completion
