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
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

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
      - frequency_penalty (to replicate openAI's concept)

    And returns an OpenAI-style dict with "model" and "choices" keys.
    """
    # 1) Load or get model+tokenizer
    model_name = kwargs.get('model', 'llama3.1-8b')
    model, tokenizer = load_model_and_tokenizer(model_name)

    # 2) Parse generation parameters
    max_tokens = kwargs.get('max_tokens', 128)
    temperature = kwargs.get('temperature', 1.0)
    top_p = kwargs.get('top_p', 1.0)
    freq_penalty = kwargs.get('frequency_penalty', 0.0)
    return_logprobs = kwargs.get('logprobs', 1)
    do_sample = (temperature > 0.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_for_generate = model.module if hasattr(model, 'module') else model

    # 3) Get text to generate from. Could be 'messages' or 'prompt'.
    messages = kwargs.get('messages', None)
    if messages is not None:
        # Very simple approach: gather content of user messages
        if len(messages) == 0:
            text_inputs = [""]
        else:
            text_inputs = []
            for msg in messages:
                if msg['role'] == 'user':
                    text_inputs.append(msg['content'])
            if not text_inputs:
                # fallback: if no user role found
                text_inputs = [messages[-1]['content']]
    else:
        # Possibly used 'prompt' param
        prompt = kwargs.get('prompt', None)
        if isinstance(prompt, str):
            text_inputs = [prompt]
        elif isinstance(prompt, list):
            text_inputs = prompt
        else:
            text_inputs = [""]

    # 4) Build a LogitsProcessorList
    from transformers import LogitsProcessorList
    logits_processors = LogitsProcessorList()

    # If freq_penalty > 0, add our custom FrequencyPenaltyProcessor
    if freq_penalty > 0.0:
        freq_proc = FrequencyPenaltyProcessor(penalty=freq_penalty)
        logits_processors.append(freq_proc)
    
    # 5) Generate for each text input
    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for idx, input_text in enumerate(text_inputs):
        # Tokenize the prompt first
        prompt_enc = tokenizer(
            input_text,
            return_tensors='pt',
            add_special_tokens=False,
            return_offsets_mapping=True
        )

        input_ids = prompt_enc["input_ids"].to(device)
        prompt_len = input_ids.shape[1]  # how many tokens in the prompt
        prompt_offsets = prompt_enc["offset_mapping"][0] if "offset_mapping" in prompt_enc else None

        # The raw generation call
        with torch.no_grad():
            output = model_for_generate.generate(
                input_ids=input_ids,
                max_length=prompt_len + max_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p,
                repetition_penalty=1.0,
                logits_processor=logits_processors,
                return_dict_in_generate=True,
                output_scores=True
            )

        # The entire sequence includes prompt + newly generated tokens
        full_ids = output.sequences[0]
        full_len = full_ids.shape[0]
        # Convert result to text
        gen_ids = full_ids[prompt_len:]  # the new tokens
        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # We decide a "finish_reason":
        finish_reason = "stop"  # or "length" if you want to detect it

        # We'll gather usage stats
        prompt_tokens_count = prompt_len
        completion_tokens_count = gen_ids.shape[0]
        total_prompt_tokens += prompt_tokens_count
        total_completion_tokens += completion_tokens_count

        # -----------------------------
        # 6) Build logprobs + offsets if requested
        # -----------------------------
        # If user wants token-level logprobs:
        token_strs = []
        token_logprobs = []
        token_offsets = []
        top_logprobs = None  # or store if needed

        if return_logprobs:
            scores = output.scores  # list of [batch, vocab] logits for each new token
            for i, logits_i in enumerate(scores):
                # i-th token in the newly generated portion
                step_logprobs = torch.log_softmax(logits_i[0], dim=-1)
                token_id = gen_ids[i].item()
                chosen_logprob = float(step_logprobs[token_id].item())
                token_str = tokenizer.decode([token_id], skip_special_tokens=True)

                token_strs.append(token_str)
                token_logprobs.append(chosen_logprob)

            # Build text_offsets for the newly generated portion.
            # One naive approach is to re-tokenize the entire (prompt + generated_text) 
            # and slice, but let's do an approximate approach:
            # 
            # Re-tokenize the final text, then skip prompt_len tokens to align offsets
            # (Potential minor differences in spacing or merges, but works in many cases.)
            entire_text = input_text + generated_text
            reenc = tokenizer(
                entire_text,
                return_offsets_mapping=True,
                add_special_tokens=False
            )
            re_offsets = reenc["offset_mapping"]
            # skip the first 'prompt_len' tokens
            # but watch out for mismatch if the prompt re-encodes differently
            # We'll do a safer approach: min(len(re_offsets), prompt_len + i)
            new_token_offsets = re_offsets[prompt_len : prompt_len + len(gen_ids)]
            # We only store the *start* offset typically (like OpenAI). 
            # But you could store (start,end).
            token_offsets = [o[0] for o in new_token_offsets]

        # 7) Build the "choice" entry
        choice_data = {
            "text": generated_text,
            "index": idx,
            "logprobs": None,
            "finish_reason": finish_reason
        }
        if return_logprobs:
            choice_data["logprobs"] = {
                "tokens": token_strs,
                "token_logprobs": token_logprobs,
                "text_offset": token_offsets
            }

        choices.append(choice_data)

    # -----------------------------
    # 8) Build final OpenAI-style JSON
    # -----------------------------
    # Summarize usage for entire batch
    total_used_tokens = total_prompt_tokens + total_completion_tokens
    openai_response = {
        "object": "text_completion",
        "model": model_name,
        "choices": choices,
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_used_tokens
        }
    }

    return openai_response
    