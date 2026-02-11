import torch
import os
from transformers import LlamaTokenizer, LlamaForCausalLM

def main():
    model_id = "meta-llama/Llama-2-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

    gender_pairs = [
        ("a man", "a woman"),
        ("a male doctor", "a female doctor"),
        ("a son", "a daughter")
    ]

    # Hook mechanism
    activations = {}
    def hook_fn(module, input, output):
        activations['current'] = output.detach()
    model.model.layers[15].mlp.gate_proj.register_forward_hook(hook_fn)

    # Markdown Table Header
    results_md = "\n### Gender Neuron Experiment Results\n"
    results_md += "| Gender Pair | Top 5 Neurons (Indices) |\n"
    results_md += "| :--- | :--- |\n"

    print("\nRunning Gender Discovery...")
    for male, female in gender_pairs:
        pair_acts = []
        for role in [male, female]:
            prompt = f"The identity of this person is {role}."
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                model(**inputs)
            pair_acts.append(activations['current'][0, -1, :])
        
        diff = torch.abs(pair_acts[0] - pair_acts[1])
        
        # Get indices of top 5 neurons with largest diff
        top5_values, top5_indices = torch.topk(diff, 5)
        indices_list = top5_indices.tolist()
        
        # Append to Markdown string
        results_md += f"| {male} vs {female} | {indices_list} |\n"
        print(f"Finished: {male} vs {female}")

    # 3. Auto-write to README.md
    with open("README.md", "a") as f:
        f.write(results_md)
    
    print("\nResults have been automatically appended to README.md")

if __name__ == "__main__":
    main()