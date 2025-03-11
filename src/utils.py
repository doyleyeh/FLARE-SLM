from typing import Any, List, Dict
import random
import time
import os
import logging
import copy
import string
import asyncio
import openai
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

logging.basicConfig(level=logging.INFO)

class FrequencyPenaltyProcessor(LogitsProcessor):
    """
    This custom LogitsProcessor penalizes tokens based on how many times they
    have already appeared in the generated sequence (including or excluding the prompt).
    It matches the idea behind OpenAI’s 'frequency_penalty'.
    """
    def __init__(self, penalty: float, consider_prompt: bool = False):
        """
        Args:
            penalty: Strength of the penalty, e.g. 0.5 is moderate.
            consider_prompt: Whether to also penalize tokens that appeared
                             in the *initial prompt*. If True, then any token
                             that also appears in the prompt is penalized from
                             the start.
        """
        self.penalty = penalty
        self.consider_prompt = consider_prompt
        # For each batch item, we store a frequency dictionary
        self.token_freqs = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        input_ids shape: [batch_size, seq_len]
        scores    shape: [batch_size, vocab_size]

        We update 'scores' in-place, subtracting penalty * (count of token so far).
        """
        batch_size = input_ids.shape[0]

        # If first step, initialize self.token_freqs
        if len(self.token_freqs) < batch_size:
            # create one frequency dict per batch item
            for _ in range(batch_size - len(self.token_freqs)):
                self.token_freqs.append({})

        for b in range(batch_size):
            seq = input_ids[b].tolist()

            # Count frequencies for this step
            freqs = self.token_freqs[b]
            # Clear + rebuild from scratch (or increment).
            # We'll rebuild from scratch for clarity:
            new_freqs = {}
            for tok in seq:
                new_freqs[tok] = new_freqs.get(tok, 0) + 1
            self.token_freqs[b] = new_freqs

            # Subtract penalty from the tokens that appear more frequently
            #   new_freqs[tok] is how many times 'tok' has shown up
            for tok, count in new_freqs.items():
                # The more times it’s appeared, the bigger the penalty
                # (Similar to OpenAI: new_logit = old_logit - penalty * count)
                # Make sure to index [b, tok] in 'scores'
                # Note that if you do not want a linear penalty, you can scale differently.
                scores[b, tok] -= self.penalty * float(count)

        return scores

class Utils:
    punctuations = set(string.punctuation)

    @classmethod
    def is_chat(cls, model: str):
        # return 'llama' in model #### mark any model as a chat model if needed
        return False
    # @classmethod
    # def is_code(cls, model: str):
    #     return 'code' in model
    @classmethod
    def no_stop(cls, model: str):
        # return 'turbo' in model
        return False


# class NoKeyAvailable(Exception):
#     pass


# def retry_with_exponential_backoff(
#     func,
#     max_reqs_per_min: int = 0,
#     initial_delay: float = 1,
#     exponential_base: float = 2,
#     jitter: bool = True,
#     max_retries: int = 5,
#     errors_to_catch: tuple = (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError, openai.error.Timeout, NoKeyAvailable),
#     errors_to_raise: tuple = (openai.error.APIConnectionError, openai.error.InvalidRequestError, openai.error.AuthenticationError),
# ):
#     """Retry a function with exponential backoff."""
#     def wrapper(*args, **kwargs):
#         # initialize variables
#         is_code_model = Utils.is_chat(kwargs['model'])
#         mrpm = max_reqs_per_min
#         mrpm = mrpm or (15 if is_code_model else 1000)
#         const_delay = 60 / mrpm
#         delay = initial_delay
#         num_retries = 0

#         # loop until a successful response or max_retries is hit or an exception is raised
#         while True:
#             # initialize key-related variables
#             api_key = get_key_func = return_key_func = None
#             forbid_key = False

#             try:
#                 # get key
#                 _kwargs = copy.deepcopy(kwargs)
#                 if 'api_key' in kwargs:
#                     ori_api_key = kwargs['api_key']
#                     if type(ori_api_key) is tuple:  # get a key through a call
#                         get_key_func, return_key_func = ori_api_key
#                         api_key = get_key_func()
#                     else:  # a specified key
#                         api_key = ori_api_key or os.getenv('OPENAI_API_KEY')
#                     _kwargs['api_key'] = api_key

#                 # query API
#                 start_t = time.time()
#                 logging.info(f'API call start: {_kwargs.get("api_key", "")[-5:]}')
#                 results = func(*args, **_kwargs)
#                 logging.info(f'API call end: {_kwargs.get("api_key", "")[-5:]}')
#                 return results

#             # retry on specific errors
#             except errors_to_catch as e:
#                 # check if the key is useless
#                 if hasattr(e, 'json_body') and e.json_body is not None and 'error' in e.json_body and 'type' in e.json_body['error'] and e.json_body['error']['type'] == 'insufficient_quota':  # quota error
#                     logging.info(f'NO QUOTA: {api_key[-5:]}')
#                     forbid_key = True
#                 if hasattr(e, 'json_body') and e.json_body is not None and 'error' in e.json_body and 'type' in e.json_body['error'] and e.json_body['error']['code'] == 'account_deactivated':  # ban error
#                     logging.info(f'BAN: {api_key[-5:]}')
#                     forbid_key = True

#                 # check num of retries
#                 num_retries += 1
#                 if num_retries > max_retries:
#                     raise Exception(f'maximum number of retries ({max_retries}) exceeded.')

#                 # incremental delay
#                 delay *= exponential_base * (1 + jitter * random.random())
#                 logging.info(f'retry on {e}, sleep for {const_delay + delay}')
#                 time.sleep(const_delay + delay)

#             # raise on specific errors
#             except errors_to_raise as e:
#                 raise e

#             # raise exceptions for any errors not specified
#             except Exception as e:
#                 raise e

#             finally:  # return key if necessary
#                 if api_key is not None and return_key_func is not None:
#                     end_t = time.time()
#                     return_key_func(api_key, time_spent=end_t - start_t, forbid=forbid_key)

#     return wrapper


# async def async_chatgpt(
#     *args,
#     messages: List[List[Dict[str, Any]]],
#     **kwargs,
# ) -> List[str]:
#     async_responses = [
#         openai.ChatCompletion.acreate(
#             *args,
#             messages=x,
#             **kwargs,
#         )
#         for x in messages
#     ]
#     return await asyncio.gather(*async_responses)


# @retry_with_exponential_backoff
# def openai_api_call(*args, **kwargs):
#     model = kwargs['model']
#     is_chat_model = Utils.is_chat(model)
#     if is_chat_model:
#         if len(kwargs['messages']) <= 0:
#             return []
#         if type(kwargs['messages'][0]) is list:  # batch request
#             return asyncio.run(async_chatgpt(*args, **kwargs))
#         else:
#             return openai.ChatCompletion.create(*args, **kwargs)
#     else:
#         return openai.Completion.create(*args, **kwargs)
    
############################################################################################################

# Define a function to dynamically load the correct model
_modelcache = {}
def load_model_and_tokenizer(model_name):
    model_mapping = {
        "llama3.1-8b": "meta-llama/Llama-3.1-8B",
        "mamba2": "mistralai/Mamba-Codestral-7B-v0.1",
    }
    if model_name not in model_mapping:
        raise ValueError(f"Unsupported model: {model_name}. Available models: {list(model_mapping.keys())}")
    
    if model_name in _modelcache:
        return _modelcache[model_name]
    
    hf_token = os.getenv("HF_token")
    if hf_token is None:
        raise ValueError("Hugging Face token not found. Please set the HF_token environment variable.")
    
    model_path = model_mapping[model_name]
    logging.info(f"Loading model+tokenizer for {model_name} from {model_path}")

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    
    if torch.cuda.is_available():
        model.to("cuda")
        model = torch.nn.DataParallel(model)
    
    _modelcache[model_name] = (model, tokenizer)
    return model, tokenizer

def HFmodel_call(*args, **kwargs):
    """
    A local 'generate' wrapper that takes in:
      - model name
      - prompt or messages
      - max_tokens, temperature, top_p, etc.
      - frequency_penalty (OpenAI-like)
      - echo (bool): if True, return the prompt+generated text in .text
                     and token-wise data from the beginning of the prompt
                     all the way through the generation.
    And returns an OpenAI-style dict with "model" and "choices" keys.
    """

    # 1) Parse arguments
    model_name = kwargs.get('model', 'llama3.1-8b')
    echo = kwargs.get('echo', False)                     # <--- The new echo parameter
    max_tokens = kwargs.get('max_tokens', 128)
    temperature = kwargs.get('temperature', 1.0)
    top_p = kwargs.get('top_p', 1.0)
    freq_penalty = kwargs.get('frequency_penalty', 0.0)
    return_logprobs = kwargs.get('logprobs', 1)          # If >0, we’ll collect token-level data

    # 2) Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_for_generate = model.module if hasattr(model, 'module') else model

    # 3) Get text to generate from. Could be 'messages' or 'prompt'.
    messages = kwargs.get('messages', None)
    if messages is not None:
        # Chat-style usage: gather user-role content
        if len(messages) == 0:
            text_inputs = [""]
        else:
            text_inputs = []
            for msg in messages:
                if msg['role'] == 'user':
                    text_inputs.append(msg['content'])
            # fallback: if no user role found, just use last
            if not text_inputs:
                text_inputs = [messages[-1]['content']]
    else:
        # Normal usage with 'prompt'
        prompt = kwargs.get('prompt', "")
        if isinstance(prompt, str):
            text_inputs = [prompt]
        elif isinstance(prompt, list):
            text_inputs = prompt
        else:
            text_inputs = [""]

    # 4) Build logits processors (for freq_penalty etc.)
    logits_processors = LogitsProcessorList()

    # If freq_penalty > 0, add our custom FrequencyPenaltyProcessor
    if freq_penalty > 0.0:
        freq_proc = FrequencyPenaltyProcessor(penalty=freq_penalty)
        logits_processors.append(freq_proc)
    # Generate for each text input (We will fill this with one entry per prompt)
    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for idx, prompt_text in enumerate(text_inputs):
        # 5) Tokenize (Encode) the prompt
        prompt_enc = tokenizer(
            prompt_text,
            return_tensors='pt',
            add_special_tokens=False,
            return_offsets_mapping=False,   # We don't need it *yet*
        )
        input_ids = prompt_enc["input_ids"].to(device)
        prompt_len = input_ids.shape[1] # how many tokens in the prompt

        # 6) Generate
        with torch.no_grad():
            output = model_for_generate.generate(
                input_ids=input_ids,
                max_length=prompt_len + max_tokens,
                temperature=temperature if temperature > 0.0 else 1.0,
                top_p=top_p,
                repetition_penalty=1.0,
                logits_processor=logits_processors,
                return_dict_in_generate=True,
                output_scores=True
            )

        # "full_ids" is the entire token sequence = prompt + newly-generated
        full_ids = output.sequences[0]

        # If echo=True, we return the entire text (prompt+completion).
        # If echo=False, we return only the newly generated tokens.
        if echo:
            used_ids = full_ids
        else:
            used_ids = full_ids[prompt_len:]  # slice out only new tokens

        # Convert result to text
        generated_text = tokenizer.decode(used_ids, skip_special_tokens=True)

        # 7) Basic usage stats
        prompt_tokens_count = prompt_len
        completion_tokens_count = len(used_ids)
        total_prompt_tokens += prompt_tokens_count
        total_completion_tokens += completion_tokens_count

        # 8) Build token-level arrays if logprobs>0
        token_strs = []
        token_logprobs = []
        token_offsets = None

        if return_logprobs:
            # The raw generate call gives us `scores`, one for each new token
            # that was generated (not including the prompt). We’ll reconstruct
            # each new token’s logprob from these. If echo=True, we have to
            # unify the prompt tokens + new tokens into a single array. So
            # we’ll do a second pass to handle that properly.

            # 8A) Collect the logprobs for the *newly generated* portion
            #     from `scores`.
            # (Hugging Face always returns one entry in `scores` per step in
            #  the *generated* portion, ignoring the prompt.)
            new_tokens_len = len(output.scores)  # how many steps
            # For the portion that is newly generated, the tokens are:
            # full_ids[prompt_len : prompt_len + new_tokens_len]
            # The i-th new token is full_ids[prompt_len + i]
            # We'll store logprobs for them in a local array
            gen_token_ids = full_ids[prompt_len : prompt_len + new_tokens_len].tolist()

            # Collect logprobs for newly generated tokens
            new_token_strs = []
            new_token_logprobs = []
            for i, logits_i in enumerate(output.scores):
                step_logprobs = torch.log_softmax(logits_i[0], dim=-1)   # [vocab_size]
                tok_id = gen_token_ids[i]
                logp = float(step_logprobs[tok_id].item())
                text_tok = tokenizer.decode([tok_id], skip_special_tokens=True)

                new_token_strs.append(text_tok)
                new_token_logprobs.append(logp)

            # 8B) If echo=False, then logprobs[] should just be these new tokens.
            #     If echo=True, we also want the prompt’s tokens in front, with
            #     offsets starting at 0. So we’ll do a re-tokenization pass
            #     over the ENTIRE combined text (prompt + generation).
            #     Then we line up the final tokens array with the new_token_logprobs for the generation portion.
            if echo:
                # The text is prompt_text + generated_text
                # We'll unify them in a single string:
                full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
                # Re-tokenize to get offsets from the beginning
                reenc = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)

                # reenc["input_ids"] = all tokens (prompt + generation) in same subwording
                # We want:
                #  - tokens for the entire prompt + generation
                #  - logprobs = None for the prompt portion (OpenAI does store them as “null”)
                #  - offsets from 0.. for each token

                # We'll do a naive direct alignment approach: one token from reenc
                # for each ID in full_ids. Usually this matches perfectly if the
                # same subword merges apply. We'll store something like:
                offsets = reenc["offset_mapping"]  # list of (start,end) in the combined text
                full_input_ids = reenc["input_ids"]

                # Now we expect len(full_ids) == len(full_input_ids) in most cases,
                # but if you see minor discrepancies, you'd need a more robust matching.
                # We'll assume it lines up exactly:

                # The first `prompt_len` tokens in full_ids are the prompt.
                # The next `new_tokens_len` are the newly generated portion (the same size as `scores`).
                # We'll create parallel arrays of strings + logprobs:
                all_token_strs = []
                all_token_logprobs = []
                all_token_offsets = []
                j = 0
                for j in range(len(full_input_ids)):
                    # text snippet
                    t_str = tokenizer.decode([full_input_ids[j]], skip_special_tokens=True)
                    # offset pair => store just the start offset (like OpenAI)
                    start_offset = offsets[j][0]
                    all_token_strs.append(t_str)
                    all_token_offsets.append(start_offset)
                    # For the prompt portion, logprobs = None
                    if j < prompt_len:
                        # no logprob for the prompt
                        all_token_logprobs.append(None)
                    else:
                        # For newly generated tokens, we match up with new_token_logprobs
                        # The index in new_token_logprobs is j - prompt_len
                        gen_index = j - prompt_len
                        # Check bounds
                        if gen_index < len(new_token_logprobs):
                            all_token_logprobs.append(new_token_logprobs[gen_index])
                        else:
                            # Should not normally happen, but guard anyway
                            all_token_logprobs.append(None)

                token_strs = all_token_strs
                token_logprobs = all_token_logprobs
                token_offsets = all_token_offsets

            else:
                # echo=False => no prompt tokens in `choices[].logprobs`
                # so we do a re-tokenization just for the newly generated text
                # to get offsets from 0.. up to len(generated_text).
                if generated_text.strip():
                    reenc = tokenizer(
                        generated_text,
                        return_offsets_mapping=True,
                        add_special_tokens=False
                    )
                    offsets = reenc["offset_mapping"]
                    gen_input_ids = reenc["input_ids"]

                    # We also have new_token_strs + new_token_logprobs from above
                    # We must line them up. Typically the merges should match 1:1 if the text is short.
                    # We'll do the naive approach: assume lengths match exactly.
                    # If there's a mismatch, you'll need a more robust alignment.
                    if len(gen_input_ids) == len(new_token_strs):
                        token_strs = new_token_strs
                        token_logprobs = new_token_logprobs
                        token_offsets = [off[0] for off in offsets]  # store just start
                    else:
                        # fallback: minimal usage
                        token_strs = new_token_strs
                        token_logprobs = new_token_logprobs
                        token_offsets = [None]*len(new_token_strs)
                else:
                    # no new text
                    token_strs = []
                    token_logprobs = []
                    token_offsets = []

        # 9) Build the choice dictionary
        finish_reason = "stop"  # or "length", etc.
        choice_data = {
            "text": generated_text,
            "index": idx,
            "finish_reason": finish_reason,
            "logprobs": None
        }
        if return_logprobs:
            choice_data["logprobs"] = {
                "tokens": token_strs,
                "token_logprobs": token_logprobs,
                "text_offset": token_offsets
            }

        choices.append(choice_data)

    # 10) Summarize usage across the entire batch
    total_used_tokens = total_prompt_tokens + total_completion_tokens
    response = {
        "object": "text_completion",
        "model": model_name,
        "choices": choices,
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_used_tokens
        }
    }
    return response