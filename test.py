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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        # fallback if eos_token_id is also None
        tokenizer.pad_token_id = 0
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        model.to("cuda")
    
    _modelcache[model_name] = (model, tokenizer)
    return model, tokenizer

def HFmodel_call(*args, **kwargs):
    """
    A local 'generate' wrapper that takes in:
      - model name
      - prompt or messages
      - max_tokens, temperature, top_p, etc.
      - frequency_penalty (OpenAI-like)
      - echo (bool)
    And returns an OpenAI-style dict with "model" and "choices" keys.
    """

    model_name = kwargs.get('model', 'llama3.1-8b')
    echo = kwargs.get('echo', False)
    max_tokens = kwargs.get('max_tokens', 128)
    temperature = kwargs.get('temperature', 0.0)
    top_p = kwargs.get('top_p', 1.0)
    freq_penalty = kwargs.get('frequency_penalty', 0.0)
    return_logprobs = kwargs.get('logprobs', 0)  # If >0, collect token-level data
    if temperature > 0.0:
        do_sample = True
    else:
        do_sample = False

    # 1) Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_for_generate = model.module if hasattr(model, 'module') else model

    # 2) Gather text inputs from `messages` or `prompt`
    messages = kwargs.get('messages', None)
    if messages is not None:
        # Chat style: gather user-role content
        if len(messages) == 0:
            text_inputs = [""]
        else:
            text_inputs = []
            for msg in messages:
                if msg['role'] == 'user':
                    text_inputs.append(msg['content'])
            if not text_inputs:  # fallback if no user content
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

    # 3) Build logits processors
    logits_processors = LogitsProcessorList()
    if freq_penalty > 0.0:
        freq_proc = FrequencyPenaltyProcessor(penalty=freq_penalty)
        logits_processors.append(freq_proc)

    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for idx, prompt_text in enumerate(text_inputs):
        # 4) Tokenize prompt
        prompt_enc = tokenizer(
            prompt_text,
            return_tensors='pt',
            add_special_tokens=False,
            return_offsets_mapping=False  # no need for prompt offsets here
        )
        attention_mask = prompt_enc["attention_mask"].to(device)
        input_ids = prompt_enc["input_ids"].to(device)
        prompt_len = input_ids.shape[1]

        # 5) Generate
        with torch.no_grad():
            output = model_for_generate.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=prompt_len + max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p,
                repetition_penalty=1.0,
                logits_processor=logits_processors,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id
            )

        # "full_ids" includes prompt + newly generated tokens
        full_ids = output.sequences[0]

        # Slice out the newly generated portion if echo=False
        if echo:
            used_ids = full_ids
        else:
            used_ids = full_ids[prompt_len:]  # only new tokens

        # Convert to final text (with special tokens removed so you can print nicely)
        generated_text = tokenizer.decode(used_ids, skip_special_tokens=True)

        prompt_tokens_count = prompt_len
        completion_tokens_count = len(used_ids)
        total_prompt_tokens += prompt_tokens_count
        total_completion_tokens += completion_tokens_count

        # Prepare to fill logprobs if asked
        token_strs = []
        token_logprobs = []
        token_offsets = None

        if return_logprobs:
            # Collect the raw logprobs of newly generated tokens
            # (length = number of generation steps)
            new_tokens_len = len(output.scores)  # how many new tokens
            gen_token_ids = full_ids[prompt_len : prompt_len + new_tokens_len].tolist()

            new_token_logprobs = []
            for i, logits_i in enumerate(output.scores):
                step_logprobs = torch.log_softmax(logits_i[0], dim=-1)  # [vocab_size]
                tok_id = gen_token_ids[i]
                new_token_logprobs.append(float(step_logprobs[tok_id].item()))

            # -----------------------------
            #  EITHER echo=True or echo=False
            #  We do a re-encode of exactly the span we want token-by-token offsets for.
            #  - If echo=True => full text = prompt + generated
            #  - If echo=False => text = only the newly generated portion
            #  BUT we do it WITHOUT skipping special tokens, so the IDs align 1:1.
            # -----------------------------
            if echo:
                # The entire text (prompt + completion), as raw as possible:
                full_text_incl_special = tokenizer.decode(full_ids, skip_special_tokens=False)
                # Re-encode
                reenc = tokenizer(full_text_incl_special,
                                  return_offsets_mapping=True,
                                  add_special_tokens=False)
                reenc_ids = reenc["input_ids"]
                reenc_offsets = reenc["offset_mapping"]

                # We'll iterate over reenc_ids. The first `prompt_len` correspond
                # to the prompt tokens, for which logprobs = None.
                # The next `new_tokens_len` correspond to newly generated tokens.
                # We'll store them in parallel arrays.
                # Since we used skip_special_tokens=False, reenc_ids should align exactly
                # with full_ids (barring unusual mismatch—rare if same tokenizer).
                # If there's a mismatch, you can do a pointer-based approach below.

                # In many cases, len(reenc_ids) == len(full_ids). But let’s do
                # a pointer-based match for safety:
                p_reenc = 0
                p_full = 0
                all_strs = []
                all_logprobs = []
                all_offsets = []

                # Precomputed new-token logprobs are only for the slice [prompt_len : prompt_len+new_tokens_len].
                # We'll track how many generation tokens we've consumed:
                gen_consumed = 0

                while p_reenc < len(reenc_ids) and p_full < len(full_ids):
                    if reenc_ids[p_reenc] == full_ids[p_full]:
                        # Match => gather token text, offset, logprob
                        raw_str = tokenizer.decode([reenc_ids[p_reenc]], skip_special_tokens=False)
                        # For offset, store the start char (like OpenAI)
                        start_char = reenc_offsets[p_reenc][0]

                        if p_full < prompt_len:
                            # This is a prompt token => logprob=None
                            all_strs.append(tokenizer.decode([reenc_ids[p_reenc]], skip_special_tokens=True))
                            all_logprobs.append(None)
                            all_offsets.append(start_char)
                        else:
                            # This is one of the newly generated tokens
                            gen_index = gen_consumed
                            if gen_index < len(new_token_logprobs):
                                all_strs.append(tokenizer.decode([reenc_ids[p_reenc]], skip_special_tokens=True))
                                all_logprobs.append(new_token_logprobs[gen_index])
                                all_offsets.append(start_char)
                            else:
                                # Safety fallback
                                all_strs.append(raw_str)
                                all_logprobs.append(None)
                                all_offsets.append(start_char)
                            gen_consumed += 1

                        p_reenc += 1
                        p_full += 1
                    else:
                        # If there's a mismatch, we move forward in reenc_ids
                        # until we find a match or exhaust them. This can happen
                        # if the HF tokenizer merges things differently than
                        # the raw internal IDs. It's rare with a consistent model/tokenizer.
                        p_reenc += 1

                token_strs = all_strs
                token_logprobs = all_logprobs
                token_offsets = all_offsets

            else:
                # echo=False => we only want offsets for the newly generated text
                # Step 1: decode the newly generated tokens WITHOUT skip_special_tokens
                gen_text_incl_special = tokenizer.decode(used_ids, skip_special_tokens=False)
                # Step 2: re-encode that chunk
                reenc = tokenizer(gen_text_incl_special,
                                  return_offsets_mapping=True,
                                  add_special_tokens=False)
                reenc_ids = reenc["input_ids"]
                reenc_offsets = reenc["offset_mapping"]

                # We align reenc_ids with used_ids. The new_token_logprobs array
                # has one entry per newly generated token. We'll do pointer-based
                # matching:
                p_reenc = 0
                p_used = 0
                matched_strs = []
                matched_logprobs = []
                matched_offsets = []

                while p_reenc < len(reenc_ids) and p_used < len(used_ids):
                    if reenc_ids[p_reenc] == used_ids[p_used]:
                        # Match => gather text + offset + logprob
                        raw_str = tokenizer.decode([reenc_ids[p_reenc]], skip_special_tokens=True)
                        start_char = reenc_offsets[p_reenc][0]
                        matched_strs.append(raw_str)
                        matched_logprobs.append(new_token_logprobs[p_used])  # same index
                        matched_offsets.append(start_char)

                        p_reenc += 1
                        p_used += 1
                    else:
                        # Move forward in reenc_ids until we find a match
                        p_reenc += 1

                token_strs = matched_strs
                token_logprobs = matched_logprobs
                token_offsets = matched_offsets

        # Build final choice data
        finish_reason = "stop"
        choice_data = {
            "text": generated_text,  # your nicely printed text (skip_special_tokens=True)
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

    # Usage
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



if __name__ == '__main__':
    # Simple prompt completion
    response = HFmodel_call(
        prompt="Explain the theory of relativity in simple terms:",       # Single-string prompt
        model="llama3.1-8b",         # Which model to use (from your defined mappings)
        max_tokens=128                # Limit generation length
    )

    # Print the result
    print("Response object:\n", response)
    print("\nGenerated text:\n", response["choices"][0]["text"])
    print("########################################################")

    # ##########################
    # # Chat-style completion
    # messages = [
    #     {"role": "system", "content": "You are a friendly, helpful AI assistant."},
    #     {"role": "user", "content": "Hello! How are you today?"}
    # ]

    # response = HFmodel_call(
    #     model="llama3.1-8b",
    #     messages=messages,    # Provide messages instead of a direct 'prompt'
    #     max_tokens=60
    # )
    # print("Response object:\n", response)
    # print("Chat response:\n", response["choices"][0]["text"])
    ###################################
    # Advanced usage with logprobs
    response = HFmodel_call(
        model="llama3.1-8b",
        prompt="Explain the theory of relativity in simple terms:",
        max_tokens=100,
        temperature=0,       # Sampling temperature
        top_p=1.0,             # Top-p nucleus sampling
        frequency_penalty=0.5, # Apply frequency penalty (OpenAI-like)
        echo=True,             # Include the prompt in the returned text
        logprobs=1             # Return token-level logprobs
    )
    print("Response object:\n", response)
    print("Generated text (including prompt):\n", response["choices"][0]["text"])

    # If logprobs=1, you can inspect the tokens and their log probabilities:
    log_probs_info = response["choices"][0]["logprobs"]
    print("Tokens:", log_probs_info["tokens"])
    print("Token logprobs:", log_probs_info["token_logprobs"])
    print("Text offsets:", log_probs_info["text_offset"])
    print("########################################################")
    ##############################
    # Multiple prompts
    prompts = [
        "Write a short poem about the sunrise.",
        "What is the capital of France?"
    ]

    response = HFmodel_call(
        prompt=prompts,       # List of multiple prompts
        model="llama3.1-8b",
        max_tokens=30
    )

    # Each prompt has its own choice object
    for idx, choice in enumerate(response["choices"]):
        print(f"Prompt {idx+1}: {prompts[idx]}")
        print("Completion:", choice["text"])
        print("Finish reason:", choice["finish_reason"])
        print("-"*40)


    print("Usage stats:")
    print("Prompt tokens used:", response["usage"]["prompt_tokens"])
    print("Completion tokens used:", response["usage"]["completion_tokens"])
    print("Total tokens used:", response["usage"]["total_tokens"])