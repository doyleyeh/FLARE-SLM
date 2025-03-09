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
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
logging.basicConfig(level=logging.INFO)


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

    model = AutoModelForCausalLM.from_pretrained(model_path, use_auth_token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=hf_token)

    _modelcache[model_name] = (model, tokenizer)
    return model, tokenizer

def HFmodel_call(*args, **kwargs):
    model_name = kwargs.get('model', 'llama3.1-8b')  # Default to LLaMA 3.1 8B
    model, tokenizer = load_model_and_tokenizer(model_name)

    # 1) Extract the text to generate from:
    messages = kwargs.get('messages', None)
    if messages is not None:
        # Possibly you have a single list of dicts: [{"role": "user", "content": "..."}]
        # Let's extract all user contents if you want to support multiple messages. 
        # Or just do the last user content. Adjust logic to your liking:
        if len(messages) == 0:
            text_inputs = [""]
        else:
            # For simplicity, let's just concatenate them or pick the last "user" content:
            # The old OpenAI code can be quite flexible. We'll do the simplest approach:
            text_inputs = []
            for msg in messages:
                if msg['role'] == 'user':
                    text_inputs.append(msg['content'])
            if not text_inputs:
                text_inputs = [messages[-1]['content']]
    else:
        # If you had prompt=... usage
        prompt = kwargs.get('prompt', None)
        if isinstance(prompt, str):
            text_inputs = [prompt]
        elif isinstance(prompt, list):
            text_inputs = prompt
        else:
            text_inputs = [""]

    # 2) Extract generation params
    max_tokens = kwargs.get('max_tokens', 50)
    temperature = kwargs.get('temperature', 1.0)
    top_p = kwargs.get('top_p', 1.0)
    frequency_penalty  = kwargs.get('frequency_penalty', 1.0)   # map to repetition penalty
    return_logprobs = kwargs.get('logprobs', 0)  # Whether to return log probabilities
    
    # 3) For each input text, do HF generation
    choices = []
    for input_text in text_inputs:
        # Tokenize
        inputs_tok = tokenizer(input_text, return_tensors='pt')
        input_ids = inputs_tok.input_ids

        # Generate
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=frequency_penalty,
                return_dict_in_generate=True,
                output_scores=True  # needed for logprobs
            )

        # Decode to text
        generated_ids = output.sequences[0]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Build a single "choice" dictionary, like OpenAI
        choice_dict = {
            "text": generated_text,
            "finish_reason": "length",  # or "stop" if you detect a stop condition
        }

        # If user wants logprobs
        if return_logprobs:
            token_logprobs = []
            # The newly generated tokens are in `output.scores`
            # e.g. if we generated N new tokens, we have len(output.scores) = N
            for i, logits in enumerate(output.scores):
                # turn logits -> logprobs
                lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
                # the i-th generated token is at position input_length + i
                gen_token_id = output.sequences[0][input_ids.shape[1] + i].item()
                gen_token_str = tokenizer.decode([gen_token_id])
                gen_token_lp = lprobs[0, gen_token_id].item()
                token_logprobs.append((gen_token_str, gen_token_lp))

            choice_dict["logprobs"] = {
                "tokens": [t for (t, _) in token_logprobs],
                "token_logprobs": [p for (_, p) in token_logprobs],
            }

        choices.append(choice_dict)

    # If you had multiple input_text items, you have multiple choices. 
    # For direct parity with openai, typically you'd have one "choices" item per prompt.
    # This structure is:
    return {
        "model": model_name,
        "choices": choices
    }