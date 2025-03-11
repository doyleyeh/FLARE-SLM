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

    choices = []
    for input_text in text_inputs:
        # 5) Tokenize
        inputs_tok = tokenizer(input_text, return_tensors='pt')
        if torch.cuda.is_available():
            input_ids = inputs_tok.input_ids.to("cuda")  
        else:
            input_ids = inputs_tok.input_ids

        # 6) Generate with your desired parameters, passing in the logits_processors
        with torch.no_grad():
            output = model.module.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + max_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p,
                repetition_penalty=1.0,  # built-in HF param if you want it
                logits_processor=logits_processors,  # our custom logic
                return_dict_in_generate=True,
                output_scores=True
            )

        # 7) Convert result to text
        gen_ids = output.sequences[0]
        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # 8) Build an openAI-like 'choice' entry
        choice_dict = {
            "text": generated_text,
            # set finish_reason as you see fit
            "finish_reason": "length",  # or "stop"
        }

        # If user wants token-level probabilities
        if return_logprobs:
            # We'll parse the newly generated portion from output.scores
            token_logprobs = []
            new_tokens_count = len(output.scores)  # how many new tokens we produced
            base_len = input_ids.shape[1]
            for i, logits in enumerate(output.scores):
                # i-th generation step => position = base_len + i
                lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
                token_id = output.sequences[0][base_len + i].item()
                token_str = tokenizer.decode([token_id])
                token_lp = lprobs[0, token_id].item()
                token_logprobs.append((token_str, token_lp))

            choice_dict["logprobs"] = {
                "tokens": [t for (t, _) in token_logprobs],
                "token_logprobs": [p for (_, p) in token_logprobs],
            }

        choices.append(choice_dict)

    # 9) Return an openAI-like JSON
    return {
        "model": model_name,
        "choices": choices
    }