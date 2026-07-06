from transformers import AutoTokenizer, AutoModelForCausalLM
from teacher.extract import get_attentions
from teacher.pool import pool_attention, rollup
from teacher.validate import check_causal, check_causal
import matplotlib.pyplot as plt
import torch

name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(
    name, torch_dtype="auto", device_map="auto", attn_implementation="eager",
)

prompt = "Master Prompt Template (Adapt for your specific use case)[ROLE & CONTEXT]Act as a Senior AI Solutions Architect and expert in advanced prompt engineering. You are tasked with analyzing complex technical requirements, designing scalable microservices architectures, and outputting highly structured, production-ready documentation. You will apply industry best practices, prioritize security protocols, and ensure strict adherence to DRY (Don't Repeat Yourself) principles. You have 15 years of experience in system design, cloud infrastructure, and enterprise-level AI deployments.[OBJECTIVE]Your goal is to parse the raw project requirements provided below, conceptualize a comprehensive system architecture, and generate a step-by-step implementation guide. Your output must include data flow diagrams (in Markdown/Mermaid syntax), a robust API specification, and a detailed deployment plan.[CONSTRAINTS & GUIDELINES]Output Generation: Respond in clear, concise, and structured Markdown. Use appropriate headings, bold text, and lists.Structure: Break down the response logically into: System Overview, Architecture Design, Data Models, API Endpoints, and Deployment Steps.Code Quality: All code snippets must be fully implemented (no placeholders or 'TODOs'), syntactically valid, and properly commented.Error Handling: Outline robust error-handling mechanisms for every major component, including retry logic and fallback states.Tone: Maintain a professional, objective, and authoritative tone suitable for technical documentation.[INPUT DATA][Insert your highly specific, detailed source material, technical specifications, or raw data here. Ensure that all necessary variables, constraints, and project parameters are comprehensively defined. The more context provided in this section, the more accurate the resulting architecture will be.][OUTPUT FORMAT EXPECTED]Begin your response directly with a high-level summary of the architecture. Do not include conversational filler (e.g., Here is your plan). Proceed immediately to the System Overview block.Best Practices for Log PromptsAvoid Bloat: Lengthy prompts do not automatically cost more or take longer if they are concise, but adding redundancies will degrade output quality.Mind the Threshold: Be aware of context limits. For models like GPT-5, performance can begin dropping after 4,000 tokens.Structure over Length: Instead of writing one massive paragraph of instructions, break up long prompts into structural sections, and rely on detailed input data rather than endless behavioral constraints.For tips on streamlining and optimizing your prompts for maximum performance and cost-efficiency:"

attentions, seq_len = get_attentions(model, tok, prompt)
print(f"seq_len={seq_len}, num_layers={len(attentions)}, layer0 shape={tuple(attentions[0].shape)}")

block_size = 64
layer0_pooled = pool_attention(attentions[0], block_size)
print(f"pooled leaf shape: {tuple(layer0_pooled.shape)}")

ok, violation = check_causal(layer0_pooled)
print("causal check:", "PASS" if ok else f"FAIL at {violation}")

levels = rollup(layer0_pooled)
for i, lvl in enumerate(levels):
    print(f"level {i} shape: {tuple(lvl.shape)}")

# layer0_pooled has shape [1, 14, 8, 8] — pick batch 0, head 0
grid = layer0_pooled[0, 0].detach().cpu().to(torch.float32).numpy()

plt.imshow(grid, cmap="viridis")
plt.title("Pooled attention, layer 0, head 0")
plt.xlabel("key block")
plt.ylabel("query block")
plt.colorbar()
plt.savefig("heatmap.png")