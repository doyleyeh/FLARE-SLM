# from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
# import torch
# import os
# hf_token = os.getenv("HF_token")

# model_id = "google/gemma-3-12b-pt"

# quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# model = Gemma3ForCausalLM.from_pretrained(
#     model_id, token=hf_token, device_map="cpu"
# ).eval()

# tokenizer = AutoTokenizer.from_pretrained(model_id)

# prompt = "Question: Which film has the director who died first, Promised Heaven or Fire Over England? Answer: "
# inputs = tokenizer(
#     prompt,
#     return_tensors="pt",
#     add_special_tokens=False,
#     return_offsets_mapping=False
# ).to(model.device)
# print("inputs: ", inputs)
# prompt_token = inputs["input_ids"]

# with torch.inference_mode():
#     outputs = model.generate(prompt_token, max_new_tokens=128, do_sample=True)

# outputs = tokenizer.batch_decode(outputs)
# print("outputs: ", outputs)

import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM
import os
hf_token = os.getenv("HF_token")
ckpt = "google/gemma-3-12b-pt"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = Gemma3ForCausalLM.from_pretrained(
    ckpt,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    token=hf_token
)

# prompt = "Explain the theory of relativity in simple terms:"
# prompt = "Question: Which film has the director who died first, Promised Heaven or Fire Over England? Answer: "
prompt = "Question: What is the capital of Taiwan?"
model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

input_len = model_inputs["input_ids"].shape[-1]
print("model_inputs: ", model_inputs)
with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=50, do_sample=False)
    generation = generation[0][input_len:]

decoded = tokenizer.decode(generation, skip_special_tokens=True)
print(decoded)
