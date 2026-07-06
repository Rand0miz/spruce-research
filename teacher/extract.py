import torch

def get_attentions(model, tok, prompt, device='cuda'):
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    attentions = outputs.attentions # tuple, length = num_layers
    # Fail loud
    assert isinstance(attentions, tuple) and len(attentions) > 0, "Attentions not returned by the model."
    sample = attentions[0] # shape: (batch_size, num_heads, seq_len, seq_len)
    assert sample.dim() == 4, f"Unexpected attention tensor shape. got shape {tuple(sample.shape)}, expected 4D tensor."

    return attentions, inputs["input_ids"].shape[1]
