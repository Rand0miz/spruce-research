import os, sys, torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval.harness import run_needle_test

name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(
    name, torch_dtype="auto", device_map="auto"
)

def generate(prompt):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    # Return only the generated text, not the echoed part.
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)

for depth in [0.0, 0.5, 1.0]:
    result = run_needle_test(generate, tok, target_tokens=2000, depth=depth)
    print(result)