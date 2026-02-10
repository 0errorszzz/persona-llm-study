import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

def main():
    model_id = "meta-llama/Llama-2-7b-hf"
    # Our two main suspects now
    target_indices = [8372, 8758]
    
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )

    test_pairs = [
        ("professional teacher", "young student"),
        ("senior doctor", "junior nurse"),
        ("experienced manager", "new intern")
    ]

    activations = {}
    def hook_fn(module, input, output):
        activations['current'] = output.detach()
    model.model.layers[15].mlp.gate_proj.register_forward_hook(hook_fn)

    print("\n" + "="*80)
    print(f"{'Comparison Group':<40} | {'Neuron 8372 Rank':<18} | {'Neuron 8758 Rank'}")
    print("-" * 80)

    for senior, junior in test_pairs:
        pair_acts = []
        for role in [senior, junior]:
            prompt = f"The identity of this person is a {role}."
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                model(**inputs)
            pair_acts.append(activations['current'][0, -1, :])
        
        diff = torch.abs(pair_acts[0] - pair_acts[1])
        
        # Calculate Ranks for both suspects
        rank_8372 = (diff > diff[8372]).sum().item()
        rank_8758 = (diff > diff[8758]).sum().item()
        
        print(f"{senior + ' vs ' + junior:<40} | Rank {rank_8372:<13} | Rank {rank_8758}")

    print("="*80)

if __name__ == "__main__":
    main()