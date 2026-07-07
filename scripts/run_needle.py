import os, sys, torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval.harness import run_needle_test

load_dotenv()  # load HF_TOKEN (and friends) from .env into os.environ
hf_token = os.environ.get("HF_TOKEN")

name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    name, torch_dtype="auto", device_map="auto", token=hf_token,
)

def generate(prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tok(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    new_tokens = out[0][input_len:]
    return tok.decode(new_tokens, skip_special_tokens=True)

for depth in [0.0, 0.5, 1.0]:
    result = run_needle_test(generate, tok, target_tokens=2000, depth=depth)
    print(result)