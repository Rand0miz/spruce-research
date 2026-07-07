import os, sys, json, argparse, glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
import vllm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from eval.harness import run_needle_test
from eval.score import score_retrival

load_dotenv()  # load HF_TOKEN (and friends) from .env into os.environ
HF_TOKEN = os.environ.get("HF_TOKEN")

# config
MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ORIG_MAX_POS = 32768          # Qwen2.5 native context 
FACTOR = 4.0                  # YaRN factor  -> 4x
ROPE_THETA = 1000000          # Qwen2.5-Coder rope_theta 
MAX_MODEL_LEN = int(ORIG_MAX_POS * FACTOR)   # 131072

LENGTHS = [32000, 64000, 128000]             # "32K / 64K / 128K"
DEPTHS  = [0.0, 0.25, 0.5, 0.75, 1.0]                                                                                                                   
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Repo-QA items: facts whose answer lives in the repo source.
QA_ITEMS = [
    {"question": "What is the value of PAD_VALUE in interfaces/validator.py?", "expected": "PAD_VALUE"},
    {"question": "What is the value of K in interfaces/validator.py?", "expected": "4"}
]

def build_yarn_overrides():
    """
    YaRN override. Current vLLM uses `rope_parameters`; older builds use
    `rope_scaling`. Detect and pick the right key so max_model_len is accepted.
    """ 
    rope = {
        "rope_type": "yarn",
        "factor": FACTOR,
        "original_max_position_embeddings": ORIG_MAX_POS,
        "rope_theta": ROPE_THETA,
    }
    ver = tuple(int(x) for x in vllm.__version__.split("."))
    key = "rope_parameters" if ver >= (0, 11) else "rope_scaling"
    print (f"vLLM version {vllm.__version__} -> using key {key} for YaRN override")
    return {key: rope, "max_model_len": MAX_MODEL_LEN}

def build_llm(tp):
    if tp == 1:
        print("[ks0] WARNING: tp=1. 126k bucket needs ~114GB KV. Use H200 or --tp 2 for the 128k run.")
    return LLM(model=MODEL,
               hf_overrides=build_yarn_overrides(),
               max_model_len=MAX_MODEL_LEN,
               tensor_parallel_size=tp,
               gpu_memory_utilization=0.9,
               dtype="auto",
               hf_token=HF_TOKEN,
               # enfroce_egar=True, # unncomment if CUDA-graph capture OOMs at 128k
    )

def make_generate(llm):
    """ wrap vLLM chat() to return just the text, for run_needle_test() """
    def generate(prompt, max_tokens=64):
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        messages = [{"role": "user", "content": prompt}]
        out = llm.chat([messages], sampling_params=params)
        return out[0].outputs[0].text
    return generate

def run_needles(generate, tokenizer):
    results = []
    for target in LENGTHS:
        for depth in DEPTHS:
            r = run_needle_test(generate, tokenizer, target_tokens=target, depth=depth)
            results.append(r)
            print(f"Needle test: target={target:>7}, depth={depth:<4}"
                  f"exact={r['exact']} fuzzy={r['fuzzy']:.2f} total={r['total']}")
    return results

def build_repo_context(tokenizer, target_tokens):
    """ Concatinate repo source files into a single prompt, up to target_tokens. """
    parts = []
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "**/*.py"), recursive=True)):
        if ".venv" in path:
            continue
        rel = os.path.relpath(path, REPO_ROOT)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            parts.append(f"# FILE: {rel}\n{f.read()}\n")
    
    ids = tokenizer("\n".join(parts))["input_ids"][:target_tokens]
    return tokenizer.decode(ids)

def run_repo_qa(generate, tokenizer):
    results = []
    for target in LENGTHS:
        context = build_repo_context(tokenizer, target)
        for item in QA_ITEMS:
            prompt = (f"{context}\n\nUsing ONLY the repository above, answer.\n"                     
                       f"Question: {item['question']}\nAnswer:")
            answer = generate(prompt, max_tokens=64)
            score = score_retrival(answer, item["expected"])
            results.append({
                "target_tokens": target,
                "question": item["question"],
                "expected": item["expected"],
                "retrieved": score["exact"],
                "fuzzy": score["fuzzy"],
                "answer": answer
            })
            print(f"[repo-qa] {target:>7} retrieved={score['exact']} "                               
                f":: {item['question'][:50]}")
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=int(os.environ.get("KS0_TP", "1")),                    
                      help="tensor_parallel_size (use 2 for 128K on 2xA100)")                          
    ap.add_argument("--out", default="ks0_results.json")                                             
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN)
    llm = build_llm(args.tp)
    generate = make_generate(llm)

    needle_results = run_needles(generate, tokenizer)
    repo_qa_results = run_repo_qa(generate, tokenizer)

    payload = {
        "model": MODEL,
        "yarn": {
            "factor": FACTOR,
            "original_max_position_embeddings": ORIG_MAX_POS,
            "max_model_len": MAX_MODEL_LEN,
        },
        "needle": needle_results,
        "repo_qa": repo_qa_results
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[ks0] wrote {args.out} "                                                                 
            f"({len(needle_results)} needle + {len(repo_qa_results)} repo-qa rows)")

if __name__ == "__main__":
    main()
