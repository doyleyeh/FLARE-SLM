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
import pdb
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList, BitsAndBytesConfig, AutoConfig, Gemma3ForCausalLM

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
            for _ in range(batch_size - len(self.token_freqs)):
                self.token_freqs.append({})

        for b in range(batch_size):
            seq = input_ids[b].tolist()

            # Count frequencies for this step
            freqs = self.token_freqs[b]
            new_freqs = {}
            for tok in seq:
                new_freqs[tok] = new_freqs.get(tok, 0) + 1
            self.token_freqs[b] = new_freqs

            # Subtract penalty
            for tok, count in new_freqs.items():
                scores[b, tok] -= self.penalty * float(count)

        return scores


############################################################################################################
def format_chat_prompt(messages, model_name):
    is_chat_style = False
    prompt_text = ""
    if 'qwen' in model_name or 'mamba' in model_name:
        system_started = False
        user_text = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                if not system_started:
                    prompt_text += "<|im_start|>system\n"
                    system_started = True
                prompt_text += f"{content} "
            elif role == "user":
                user_text += f"<|im_start|>user\n{content}<|im_end|>\n"
        prompt_text += f"<|im_end|>\n{user_text}<|im_start|>assistant\n"
    elif 'llama' in model_name:
        system_started = False
        user_text = ""
        prompt_text = "<|begin_of_text|>"
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                if not system_started:
                    prompt_text += "<|start_header_id|>system<|end_header_id|>\n\n"
                    system_started = True
                prompt_text += f"{content} "
            elif role == "user":
                user_text += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
        prompt_text += f"<|eot_id|>\n{user_text}<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif 'phi' in model_name:
        system_started = False
        user_text = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                if not system_started:
                    prompt_text += "<|system|>\n"
                    system_started = True
                prompt_text += f"{content} "
            elif role == "user":
                user_text += f"<|user|>\n{content}<|end|>"
        prompt_text += f"<|end|>\n{user_text}\n<|assistant|>\n"
    else:
        # Chat-style completion
        is_chat_style = True
        formatted_messages = []
        if 'gemma' in model_name:
            for message in messages:
                if message["role"] == "system" and formatted_messages and formatted_messages[-1]["role"] == "system":
                    formatted_messages[-1]["content"].append({"type": "text", "text": message["content"]})
                else:
                    formatted_messages.append({"role": message["role"], "content": [{"type": "text", "text": message["content"]}]})
        else:
            # Fallback format (original messages format)
            for message in messages:
                if message["role"] == "system" and formatted_messages and formatted_messages[-1]["role"] == "system":
                    formatted_messages[-1]["content"] += " " + message["content"]
                else:
                    formatted_messages.append({"role": message["role"], "content": message["content"]})
        return [formatted_messages], is_chat_style  # return as list for consistency
    return [prompt_text], is_chat_style

# Define a function to dynamically load the correct model
_modelcache = {}
def load_model_and_tokenizer(model_name):
    model_mapping = {
        "llama3.1-8b-i": "meta-llama/Llama-3.1-8B-Instruct",
        "llama3.1-8b": "meta-llama/Llama-3.1-8B",
        "llama3.2-3b-i": "meta-llama/Llama-3.2-3B-Instruct",
        "llama3.2-1b": "meta-llama/Llama-3.2-1B",
        "llama3.2-1b-i": "meta-llama/Llama-3.2-1B-Instruct",
        "mamba2": "tiiuae/falcon-mamba-7b",
        "mamba2-i": "tiiuae/Falcon3-Mamba-7B-Instruct",
        "gemma3-12b-i": "google/gemma-3-12b-it",
        "gemma3-12b": "google/gemma-3-12b-pt",
        "qwen2.5-7b-i": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
        "phi4-4b-i":"microsoft/Phi-4-mini-instruct",
        "phi3.5-4b-i":"microsoft/Phi-3.5-mini-instruct",
        "xlstm7b": "NX-AI/xLSTM-7b"
    }
    if model_name not in model_mapping:
        raise ValueError(f"Unsupported model: {model_name}. Available models: {list(model_mapping.keys())}")
    
    if model_name in _modelcache:
        return _modelcache[model_name]
    
    hf_token = os.getenv("HF_token")
    if hf_token is None:
        raise ValueError("Hugging Face token not found. Please set the HF_token environment variable.")
    
    model_path = model_mapping[model_name]
    if "xlstm" in model_name:
        xlstm_config = AutoConfig.from_pretrained("NX-AI/xLSTM-7b")
        xlstm_config.step_kernel = "native"
        xlstm_config.chunkwise_kernel = "chunkwise--native_autograd"
        xlstm_config.sequence_kernel = "native_sequence__native"

    logging.info(f"Loading model+tokenizer for {model_name} from {model_path}")
    if "gemma" in model_name:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = Gemma3ForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, token=hf_token).eval()
    elif "xlstm" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_path, config=xlstm_config, torch_dtype=torch.float16, token=hf_token)
    else:    
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        # fallback if eos_token_id is also None
        tokenizer.pad_token_id = 0
        
    if torch.cuda.is_available():
        model.to("cuda:0")
        model = torch.nn.DataParallel(model)
    
    _modelcache[model_name] = (model, tokenizer)
    return model, tokenizer


def HFmodel_call(*args, **kwargs):
    """
    A local 'generate' wrapper that takes in (OpenAI-style parameters):
      - model (str) : which local model to load
      - messages (List[Dict]) or prompt (str or List[str]): the text input
      - max_tokens (int)
      - temperature (float)
      - top_p (float)
      - frequency_penalty (float)
      - stop (str or List[str]): stop string(s) for post-hoc truncation
      - echo (bool)
      - logprobs (int) => if > 0, return token-level info
      - logit_bias (Dict[str, float])
        e.g. {"foo": 0.5, "bar": 2.0}

    Returns an OpenAI-style dict with "model", "choices", "usage".
    """
    model_name = kwargs.get('model', 'llama3.1-8b-i')
    echo = kwargs.get('echo', False)
    messages = kwargs.get('messages', None)
    prompt = kwargs.get('prompt', "")
    max_tokens = kwargs.get('max_tokens', 256)
    temperature = kwargs.get('temperature', 0.0)
    top_p = kwargs.get('top_p', 1.0)
    freq_penalty = kwargs.get('frequency_penalty', 0.0)
    return_logprobs = kwargs.get('logprobs', 0)  # 0 or None => no logprobs
    stop = kwargs.get('stop', None)
    logit_bias = kwargs.get('logit_bias', None)
    chat_style_tokenizer = False

    # Convert a single string stop to list
    if isinstance(stop, str):
        stop = [stop]
    elif not isinstance(stop, list) and stop is not None:
        raise ValueError("`stop` must be None, a string, or a list of strings.")

    do_sample = (temperature > 0.0)

    # 1) Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_for_generate = model.module if hasattr(model, 'module') else model

    # 2) Gather text inputs from `messages` or `prompt`
    if messages is not None:
        assert 'i' in model_name, "Chat-style completion requires a instruction model."
        text_inputs, chat_style_tokenizer = format_chat_prompt(messages, model_name)
        # Chat style
        # if 'qwen' in model_name or 'mamba' in model_name:
        #     fsys = True
        #     prompt_text = ""
        #     temp = ""
        #     for message in messages:
        #         if message["role"] == "system":
        #             if fsys: # first system message
        #                 fsys = False
        #                 prompt_text += "<|im_start|>system\n"
        #             prompt_text += f"{message['content']} "
        #         elif message["role"] == "user":
        #             temp += f"<|im_start|>user\n{message['content']}<|im_end|>\n"
        #     prompt_text += f"<|im_end|>\n{temp}<|im_start|>assistant\n"
        #     text_inputs = [prompt_text]
        # elif 'llama' in model_name:
        #     fsys = True
        #     prompt_text = "<|begin_of_text|>"
        #     temp = ""
        #     for message in messages:
        #         if message["role"] == "system":
        #             if fsys: # first system message
        #                 fsys = False
        #                 prompt_text += "<|start_header_id|>system<|end_header_id|>\n\n"
        #             prompt_text += f"{message['content']} "
        #         elif message["role"] == "user":
        #             temp += f"<|start_header_id|>user<|end_header_id|>\n\n{message['content']}<|eot_id|>"
        #     prompt_text += f"<|eot_id|>\n{temp}<|start_header_id|>assistant<|end_header_id|>\n\n"
        #     text_inputs = [prompt_text]
        # elif 'phi' in model_name:
        #     fsys = True
        #     prompt_text = ""
        #     temp = ""
        #     for message in messages:
        #         if message["role"] == "system":
        #             if fsys: # first system message
        #                 fsys = False
        #                 prompt_text += "<|system|>\n"
        #             prompt_text += f"{message['content']} "
        #         elif message["role"] == "user":
        #             temp += f"<|user|>\n{message['content']}<|end|>"
        #     prompt_text += f"<|end|>\n{temp}\n<|assistant|>\n"
        #     text_inputs = [prompt_text]
        # # # elif 'gemma' in model_name and '-i' in model_name:
        # #     text_inputs = [[]]
        # #     assistant_message = []
        # #     for message in messages:
        # #         if message["role"] == "system":
        # #             if len(text_inputs[0]) == 0 or text_inputs[0][-1]["role"] != "system":
        # #                 text_inputs[0].append({"role": "system", "content": [{"type": "text", "text": message["content"]}]})
        # #             else:
        # #                 text_inputs[0][-1]["content"].append({"type": "text", "text": message["content"]})
        # #         elif message["role"] == "user":
        # #             text_inputs[0].append({"role": "user", "content": [{"type": "text", "text": message["content"]}]})
        # #         elif message["role"] == "assistant":
        # #             assistant_message.append({"role": "assistant", "content": [{"type": "text", "text": message["content"]}]})
        # #     if len(assistant_message) > 0:
        # #         text_inputs[0].extend(assistant_message)
        # else:
        #     # Normal usage with 'messages' (keep the original format)
        #     text_inputs = [[]]
        #     for message in messages:
        #         if message["role"] == "system":
        #             if len(text_inputs[0]) == 0 or text_inputs[0][-1]["role"] != "system":
        #                 text_inputs[0].append({"role": "system", "content": message["content"]})
        #             else:
        #                 text_inputs[0][-1]["content"] += message["content"]
        #         elif message["role"] == "user":
        #             text_inputs[0].append({"role": "user", "content": message["content"]})
        #     chat_style_tokenizer = True
    else:
        # Normal usage with 'prompt'
        if isinstance(prompt, str):
            text_inputs = [prompt]
        elif isinstance(prompt, list):
            text_inputs = prompt
        else:
            text_inputs = [""]

    # 3) Build logits processors
    logits_processors = LogitsProcessorList()
    # 3a) If a frequency penalty is specified, add it
    if freq_penalty > 0.0:
        freq_proc = FrequencyPenaltyProcessor(penalty=freq_penalty)
        logits_processors.append(freq_proc)

    # 3b) If a logit_bias dict is provided, construct LogitsBiasProcessor
    #     Here we map each word -> factor, by encoding the word to token IDs.
    if logit_bias is not None:
        token_ids_to_bias = {}
        for word, factor in logit_bias.items():
            # Convert the word to a tuple of token ids
            encoded = tokenizer.encode(word, add_special_tokens=False)
            token_ids = tuple(encoded)
            token_ids_to_bias[token_ids] = factor

        bias_proc = LogitsBiasProcessor(token_ids_to_bias)
        logits_processors.append(bias_proc)
    

    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for idx, prompt_text in enumerate(text_inputs):
        # pdb.set_trace()
        # 4) Tokenize prompt
        if chat_style_tokenizer:
            prompt_enc = tokenizer.apply_chat_template(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
                return_offsets_mapping=False
            )
            # pdb.set_trace()
        else:
            prompt_enc = tokenizer(
                prompt_text,
                return_tensors='pt',
                add_special_tokens=False,
                return_offsets_mapping=False
            )
        print('prompt_enc:', prompt_enc)
        print('prompt_text:', prompt_text)
        if chat_style_tokenizer:
            attention_mask = torch.ones_like(prompt_enc).to(device)
            input_ids = prompt_enc.to(device)
        else:
            attention_mask = prompt_enc["attention_mask"].to(device)
            input_ids = prompt_enc["input_ids"].to(device)
        prompt_len = input_ids.shape[1]
        # pdb.set_trace()

        # 5) Generate
        with torch.no_grad():
            output = model_for_generate.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=prompt_len + max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p,
                logits_processor=logits_processors,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id
            )

        # The full (prompt + generated) tokens
        full_ids = output.sequences[0]

        # Separate out just the newly generated portion
        gen_ids = full_ids[prompt_len:]  # newly generated tokens
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        print('gen_text:', gen_text)

        # 6) Apply post-hoc stop logic only on the *newly generated text*,
        #    ignoring leading newlines in that portion (to handle LLaMA's \n\n).
        truncated_gen_text = gen_text
        # print('pdb debug for HFmodel_call before stop function')
        # pdb.set_trace()
        if stop:
            # We'll strip leading newlines from the generated text for matching only
            cleaned_for_stop = gen_text.lstrip("\n")
            leading_removed = len(gen_text) - len(cleaned_for_stop)

            earliest_index = None
            for s in stop:
                i = cleaned_for_stop.find(s)
                if i != -1:
                    if earliest_index is None or i < earliest_index:
                        earliest_index = i

            if earliest_index is not None:
                # Map back to the original (unstripped) gen_text index
                actual_index_in_gen_text = earliest_index + leading_removed
                truncated_gen_text = gen_text[:actual_index_in_gen_text]
        # print('pdb debug for HFmodel_call after stop function')
        # pdb.set_trace()
        # 7) Combine prompt + newly generated portion if echo=True,
        #    otherwise just the truncated_gen_text.
        if echo:
            # Decode prompt in raw form
            prompt_text_decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            final_text = prompt_text_decoded + truncated_gen_text
        else:
            final_text = truncated_gen_text

        # 8) Count tokens
        prompt_tokens_count = prompt_len
        # For the completion tokens, we consider how many tokens remain after truncation (if any)
        if truncated_gen_text == gen_text:
            # No stop truncation
            completion_tokens_count = len(gen_ids)
        else:
            # Must re-encode truncated_gen_text to see how many tokens remain
            truncated_enc = tokenizer(truncated_gen_text, add_special_tokens=False)
            completion_tokens_count = len(truncated_enc["input_ids"])

        total_prompt_tokens += prompt_tokens_count
        total_completion_tokens += completion_tokens_count

        # 9) Handle logprobs if requested
        #    We'll reconstruct carefully, matching the truncated text if there's a stop.
        token_strs = []
        token_logprobs = []
        token_offsets = None

        if return_logprobs:
            # Number of newly generated tokens before any truncation
            new_tokens_len = len(output.scores)  # one score per generated token
            # The token IDs actually generated (not truncated)
            gen_token_ids = full_ids[prompt_len : prompt_len + new_tokens_len].tolist()

            # Compute the logprobs for each new token
            new_token_logprobs = []
            for i, logits_i in enumerate(output.scores):
                step_logprobs = torch.log_softmax(logits_i[0], dim=-1)  # [vocab_size]
                tok_id = gen_token_ids[i]
                new_token_logprobs.append(float(step_logprobs[tok_id].item()))

            # We'll define a helper to re-encode and align logprobs with tokens
            def align_logprobs(text_str, all_ids, all_lps, is_echo):
                """
                text_str: the final text portion we want logprobs for
                all_ids: the newly generated token ids (in sequence)
                all_lps: the logprobs for those newly generated token ids
                is_echo: if True, we need to include prompt tokens with None logprobs
                """
                # If echo=True, text_str = (prompt + truncated_gen_text)
                # If echo=False, text_str = truncated_gen_text
                # We re-encode text_str to determine final tokens.
                reenc = tokenizer(text_str, return_offsets_mapping=True, add_special_tokens=False)
                reenc_ids = reenc["input_ids"]
                reenc_offsets = reenc["offset_mapping"]

                matched_strs = []
                matched_lps = []
                matched_offsets = []

                # Pointers
                p_reenc = 0
                gen_consumed = 0

                # If echo=True, the first `prompt_len` tokens in the final text are from prompt => None logprobs
                # Then the subsequent tokens come from all_ids/all_lps
                if is_echo:
                    # Re-encode the prompt separately, just to see how many tokens are in the prompt
                    prompt_part_ids = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                    prompt_enc2 = tokenizer(prompt_part_ids, add_special_tokens=False)
                    prompt_num_toks = len(prompt_enc2["input_ids"])

                    # We'll iterate through all re-encoded tokens and set None for the prompt portion
                    # Then fill logprobs for the generated portion
                    while p_reenc < len(reenc_ids):
                        token_txt = tokenizer.decode([reenc_ids[p_reenc]], skip_special_tokens=True)
                        start_char = reenc_offsets[p_reenc][0]

                        if p_reenc < prompt_num_toks:
                            # This is part of the prompt => logprob=None
                            matched_strs.append(token_txt)
                            matched_lps.append(None)
                            matched_offsets.append(start_char)
                        else:
                            # Generated portion
                            if gen_consumed < len(all_ids):
                                # If it matches the ID, assign the logprob
                                if reenc_ids[p_reenc] == all_ids[gen_consumed]:
                                    matched_strs.append(token_txt)
                                    matched_lps.append(all_lps[gen_consumed])
                                    matched_offsets.append(start_char)
                                    gen_consumed += 1
                                else:
                                    # If mismatch, just skip forward
                                    # (This can happen if the tokenizer merges or splits differently.)
                                    matched_strs.append(token_txt)
                                    matched_lps.append(None)
                                    matched_offsets.append(start_char)
                            else:
                                # No more generated logprobs left
                                matched_strs.append(token_txt)
                                matched_lps.append(None)
                                matched_offsets.append(start_char)

                        p_reenc += 1

                else:
                    # echo=False => we only have the newly generated portion
                    while p_reenc < len(reenc_ids) and gen_consumed < len(all_ids):
                        token_txt = tokenizer.decode([reenc_ids[p_reenc]], skip_special_tokens=True)
                        start_char = reenc_offsets[p_reenc][0]

                        # If the ID matches, assign the logprob
                        if reenc_ids[p_reenc] == all_ids[gen_consumed]:
                            matched_strs.append(token_txt)
                            matched_lps.append(all_lps[gen_consumed])
                            matched_offsets.append(start_char)
                            gen_consumed += 1
                            p_reenc += 1
                        else:
                            # If mismatch, skip forward in reenc
                            p_reenc += 1

                return matched_strs, matched_lps, matched_offsets

            final_strs, final_lps, final_offsets = align_logprobs(
                final_text, gen_token_ids, new_token_logprobs, echo
            )
            token_strs = final_strs
            token_logprobs = final_lps
            token_offsets = final_offsets

        finish_reason = "stop"  # or "length" if we consumed all tokens without seeing a stop
        choice_data = {
            "text": final_text,
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

    # 10) Assemble final response
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
    # print('pdb debug for HFmodel_call end')
    # pdb.set_trace()
    return response


if __name__ == '__main__':
    # Simple prompt completion
    response = HFmodel_call(
        prompt="What is LLM?",      # Single-string prompt
        model="gemma3-12b",         # Which model to use (from your defined mappings)
        max_tokens=128                # Limit generation length
    )

    # Print the result
    print("Response object:\n", response)
    print("\nGenerated text:\n", response["choices"][0]["text"])
    # print("########################################################")

    ##########################
    # Chat-style completion
    # messages = [
    #     {"role": "system", "content": "You are a friendly, helpful AI assistant."},
    #     {"role": "user", "content": "What is the capital of Taiwan?"}
    # ]
    # messages=[  # system
    #     {'role': 'system', 'content': 'You are a friendly, helpful AI assistant.'},
    #     {'role': 'system', 'content': 'Answer the question with three sentences.'},
    # ] + [  # current example
    #     {'role': 'user', 'content': "What is the capital of France?"},
    # ]

    # response = HFmodel_call(
    #     model="gemma3-12b-i",
    #     messages=messages,    # Provide messages instead of a direct 'prompt'
    #     max_tokens=60,
    #     echo=False, 
    #     temperature=0           # Include the prompt in the returned text
    #     # stop=["\n\n"]           # Stop at the first newline
    # )
    # print("Response object:\n", response)
    # print("Chat response:\n", response["choices"][0]["text"])
    
    # ###################################
    # # Advanced usage with logprobs
    # response = HFmodel_call(
    #     model="gemma3-12b-i",
    #     prompt="Explain the theory of relativity in simple terms:",
    #     max_tokens=100,
    #     temperature=0,       # Sampling temperature
    #     top_p=1.0,             # Top-p nucleus sampling
    #     frequency_penalty=0.5, # Apply frequency penalty (OpenAI-like)
    #     echo=True,             # Include the prompt in the returned text
    #     logprobs=1             # Return token-level logprobs
    # )
    # print("Response object:\n", response)
    # print("Generated text (including prompt):\n", response["choices"][0]["text"])

    # # If logprobs=1, you can inspect the tokens and their log probabilities:
    # log_probs_info = response["choices"][0]["logprobs"]
    # print("Tokens:", log_probs_info["tokens"])
    # print("Token logprobs:", log_probs_info["token_logprobs"])
    # print("Text offsets:", log_probs_info["text_offset"])
    # print("########################################################")
    # ##############################
    # # Multiple prompts
    # prompts = [
    #     "Write a short poem about the sunrise.",
    #     "What is the capital of France?"
    # ]

    # response = HFmodel_call(
    #     prompt=prompts,       # List of multiple prompts
    #     model="llama3.1-8b",
    #     max_tokens=30
    # )

    # # Each prompt has its own choice object
    # for idx, choice in enumerate(response["choices"]):
    #     print(f"Prompt {idx+1}: {prompts[idx]}")
    #     print("Completion:", choice["text"])
    #     print("Finish reason:", choice["finish_reason"])
    #     print("-"*40)


    # print("Usage stats:")
    # print("Prompt tokens used:", response["usage"]["prompt_tokens"])
    # print("Completion tokens used:", response["usage"]["completion_tokens"])
    # print("Total tokens used:", response["usage"]["total_tokens"])